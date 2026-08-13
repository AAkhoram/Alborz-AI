import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
import math
from typing import Optional, Tuple, Dict, List

# ================== تنظیمات ==================
d_model = 256
n_layers = 12          # با Pre-LN (پایین‌تر توضیح داده شده) این عمق دیگه دچار vanishing gradient نمی‌شه
n_heads = 8
d_ff = 512
seq_length = 64
batch_size = 16
learning_rate = 1e-3
epochs = 8
dropout_rate = 0.1
weight_decay = 0.01
warmup_steps = 100
top_k = 20
grad_clip = 5.0
use_compile = True      # اگه روی سیستمت مشکلی ایجاد کرد یا ارور داد، این رو False کن
use_amp = False         # اگه CPU‌ات bfloat16 پشتیبانی می‌کنه (Intel Xeon/AMD EPYC جدید)، True کن

# اندازه واژگان BPE — این عدد رو می‌تونی عوض کنی:
#   8000  → سبک‌تر، سریع‌تر، ولی دقت کمتر روی کلمات نادر
#   16000 → تعادل خوب (توصیه‌شده)
#   32000 → دقیق‌تر، ولی مدل سنگین‌تر
#   50000 → نزدیک به GPT-2، بیشترین پوشش واژگانی
vocab_size = 16000

# حداکثر حجمی (به مگابایت) که متن خام دیتاست اجازه داره همزمان توی RAM باشه.
# دیتاست هیچ‌وقت یکجا لود نمی‌شه: فایل‌های dataset/ یکی‌یکی لود می‌شن (تا سقف
# همین مقدار پر بشه)، آموزش می‌بینن، بعد offload می‌شن و نوبت فایل بعدی می‌رسه.
# اگه با کمبود رم مواجه شدی، این عدد رو کم کن.
max_ram_mb = 6000

model_path = "transformer_model.pt"
config_path = "transformer_config.json"
checkpoint_path = "training_checkpoint.pt"  # برای ادامه‌ی آموزش بعد از قطعی برق/سیستم
tokenizer_path = "tokenizer.json"             # فایل ذخیره‌ی توکنایزر BPE

# نکته: چون بسته‌ی PyTorch که نصب کردی نسخه‌ی CPU-only است (بدون CUDA کامپایل شده)،
# torch.cuda.is_available() همیشه False برمی‌گردونه و این خط همیشه cpu رو انتخاب می‌کنه؛
# (خود torch.compile محدود به CPU نیست و روی GPU هم کار می‌کنه، این محدودیت فقط
# مربوط به نسخه‌ی نصب‌شده‌ی خود PyTorch است، نه torch.compile)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# برخلاف NumPy (که تست کردیم و دیدیم عملا از چند هسته سود نمی‌بره)، PyTorch با MKL/oneDNN
# واقعا می‌تونه از چند هسته استفاده کنه - این خط دقیقا اون سوالی که قبلا پرسیدی رو جواب می‌ده
torch.set_num_threads(os.cpu_count() or 1)
# بهینه‌سازی: دقت matmul را روی 'high' می‌گذاریم تا از AVX512/FMA روی CPU‌های
# پشتیبانی‌کننده استفاده کامل بشه (تا ۲ برابر سریع‌تر روی CPU‌های جدید)
torch.set_float32_matmul_precision('high')
print(f"Device: {device} | CPU threads: {torch.get_num_threads()}")


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    اگه مدل با torch.compile کامپایل شده باشه، nn.Module واقعی زیرش (._orig_mod)
    رو برمی‌گردونه؛ در غیر این صورت خود مدل رو بدون تغییر برمی‌گردونه.
    """
    return model._orig_mod if hasattr(model, "_orig_mod") else model


# ================== پیدا کردن فایل‌های دیتاست ==================
def find_txt_files(path: str = "dataset") -> List[str]:
    """
    اگه path یک پوشه باشه، مسیر تمام فایل‌های .txt داخلش (و زیرپوشه‌ها) رو
    برمی‌گردونه؛ خروجی همیشه sorted هست. اگه path یک فایل تکی باشه، یک لیست
    تک‌عضوی برمی‌گردونه.
    """
    if os.path.isdir(path):
        txt_files = sorted(
            os.path.join(root, fn)
            for root, _, files in os.walk(path)
            for fn in files
            if fn.lower().endswith(".txt")
        )
        if not txt_files:
            raise FileNotFoundError(f"هیچ فایل .txt ای در پوشه‌ی '{path}' پیدا نشد")
        return txt_files
    else:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"فایل '{path}' پیدا نشد")
        return [path]


# ================== ساخت/لود توکنایزر BPE ==================
def build_or_load_tokenizer(txt_files: List[str], vocab_size: int) -> "Tokenizer":
    """
    اگه tokenizer_path وجود داشته باشه، لود می‌کنه؛ وگرنه یه توکنایزر BPE
    جدید از روی فایل‌های dataset/ می‌سازه و ذخیره می‌کنه.

    از huggingface/tokenizers استفاده می‌کنیم: سریع، Rust-backend، و
    نیازی نیست کل دیتاست توی RAM باشه (train_from_iterator هم داره ولی
    چون فایل‌ها رو مسیر می‌دیم، خودش streaming می‌خونه).
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    if os.path.exists(tokenizer_path):
        print(f"توکنایزر قبلی لود شد: {tokenizer_path}")
        return Tokenizer.from_file(tokenizer_path)

    print(f"در حال ساخت توکنایزر BPE با vocab_size={vocab_size} از {len(txt_files)} فایل...")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        min_frequency=2,
    )
    tokenizer.train(txt_files, trainer)
    tokenizer.save(tokenizer_path)
    print(f"توکنایزر ساخته و ذخیره شد: {tokenizer_path}")
    return tokenizer


def encode_text(text: str, tokenizer: "Tokenizer") -> torch.Tensor:
    """متن رو با BPE توکنایز می‌کنه و تنسور long برمی‌گردونه."""
    encoding = tokenizer.encode(text.lower())
    return torch.tensor(encoding.ids, dtype=torch.long)


def decode_tokens(token_ids: List[int], tokenizer: "Tokenizer") -> str:
    """لیست IDها رو با BPE دیکد می‌کنه و متن برمی‌گردونه."""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


# ================== Pool: لود/آفلود تدریجی فایل‌های دیتاست ==================
_BYTES_PER_CHAR_SAMPLE_SIZE = 65_536


def _estimate_bytes_per_char(txt_files: List[str]) -> float:
    """
    نسبت واقعی بایت/کاراکتر رو از ابتدای چند فایل تخمین می‌زنه (deterministic).
    """
    total_sample_bytes = 0
    total_sample_chars = 0
    for fp in txt_files[:5]:
        try:
            with open(fp, "rb") as f:
                raw = f.read(_BYTES_PER_CHAR_SAMPLE_SIZE)
        except OSError:
            continue
        if not raw:
            continue
        try:
            text = raw.decode("utf-8", errors="ignore").lower()
        except Exception:
            continue
        if text:
            total_sample_bytes += len(raw)
            total_sample_chars += len(text)

    if total_sample_chars == 0:
        return 2.0
    return total_sample_bytes / total_sample_chars


def compute_steps_per_epoch_total(txt_files: List[str]) -> int:
    """
    تخمین تعداد کل step های لازم برای یک epoch. برای BPE، تخمین دقیق‌تر
    نیاز داره چون تعداد توکن‌ها از تعداد کاراکترها کمتره (هر کلمه ~۱-۳ توکن).
    همچنان از حجم بایتی استفاده می‌کنیم ولی _estimate_bytes_per_char رو
    با نسبت bytes/token جایگزین نمی‌کنیم (چون لازم نیست کل فایل خونده بشه).
    """
    total_bytes = sum(os.path.getsize(fp) for fp in txt_files)
    bytes_per_char = _estimate_bytes_per_char(txt_files)
    # rough: هر کلمه فارسی ~۵ کاراکتر، هر کلمه ~۱.۵ توکن BPE
    # پس roughly: 1 token ~= 3-4 chars
    est_total_tokens = total_bytes / bytes_per_char / 3.5
    return max(1, int(est_total_tokens // (seq_length * batch_size)))


class DatasetPool:
    """
    به‌جای اینکه همه‌ی فایل‌های dataset/ یکجا توی RAM باشن، فایل‌ها رو
    یکی‌یکی لود می‌کنه تا مجموع حجم به سقف max_ram_mb برسه.
    """

    def __init__(
        self,
        txt_files: List[str],
        tokenizer: "Tokenizer",
        seq_length: int,
        max_ram_mb: int,
        steps_per_epoch_total: int,
        verbose: bool = True,
    ):
        self.txt_files = txt_files
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_ram_bytes = max_ram_mb * 1_000_000
        self.steps_per_epoch_total = steps_per_epoch_total
        self.verbose = verbose

        self._file_sizes_bytes = [os.path.getsize(fp) for fp in txt_files]
        total_bytes = sum(self._file_sizes_bytes) or 1

        self._steps_for_file = [
            max(1, round(steps_per_epoch_total * b / total_bytes))
            for b in self._file_sizes_bytes
        ]

        self._file_order = list(range(len(txt_files)))

        self.epoch = 0
        self._order_pos = 0
        self._loaded: Dict[int, torch.Tensor] = {}
        self._loaded_bytes = 0
        self._steps_left: Dict[int, int] = {}
        self._skipped_this_epoch: set = set()

        self._fill_pool()

    def _fill_pool(self) -> None:
        while self._order_pos < len(self._file_order):
            fi = self._file_order[self._order_pos]
            if fi in self._loaded or fi in self._skipped_this_epoch:
                self._order_pos += 1
                continue

            fbytes = self._file_sizes_bytes[fi]
            room_left = self._loaded_bytes < self.max_ram_bytes
            fits = self._loaded_bytes + fbytes <= self.max_ram_bytes
            pool_empty = not self._loaded

            if not (fits or (pool_empty and room_left)):
                break

            if not fits and pool_empty:
                if self.verbose:
                    print(
                        f"هشدار: فایل '{self.txt_files[fi]}' به‌تنهایی "
                        f"({fbytes / 1_000_000:.1f} مگابایت) از سقف max_ram_mb="
                        f"{self.max_ram_bytes / 1_000_000:.0f} بزرگ‌تره. "
                        f"اگه OOM دادی، این فایل رو تقسیم کن."
                    )

            success = self._load_file(fi)
            if not success:
                self._skipped_this_epoch.add(fi)
            self._order_pos += 1

    def _load_file(self, fi: int) -> bool:
        fp = self.txt_files[fi]
        try:
            with open(fp, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            print(f"هشدار: فایل '{fp}' با UTF-8 خونده نشد و در این epoch نادیده گرفته شد")
            return False
        if not content:
            print(f"هشدار: فایل '{fp}' خالیه و در این epoch نادیده گرفته شد")
            return False

        # توکنایز کردن — اگه تعداد توکن‌ها کمتر از seq_length+1 باشه، نمی‌شه ازش batch گرفت
        tokens = encode_text(content, self.tokenizer)
        if len(tokens) < self.seq_length + 1:
            print(
                f"هشدار: فایل '{fp}' فقط {len(tokens)} توکن داره، کمتر از "
                f"seq_length+1={self.seq_length + 1}؛ در این epoch نادیده گرفته شد."
            )
            return False

        self._loaded[fi] = tokens
        self._loaded_bytes += self._file_sizes_bytes[fi]
        if fi not in self._steps_left:
            self._steps_left[fi] = self._steps_for_file[fi]
        if self.verbose:
            print(
                f"[pool] لود شد: '{fp}' ({self._file_sizes_bytes[fi] / 1_000_000:.1f} مگابایت) | "
                f"pool فعلی: {len(self._loaded)} فایل، "
                f"{self._loaded_bytes / 1_000_000:.1f}/{self.max_ram_bytes / 1_000_000:.0f} مگابایت"
            )
        return True

    def _offload_file(self, fi: int) -> None:
        fp = self.txt_files[fi]
        self._loaded_bytes -= self._file_sizes_bytes[fi]
        del self._loaded[fi]
        self._steps_left.pop(fi, None)
        if self.verbose:
            print(f"[pool] آفلود شد: '{fp}' (سهم step آموزشیش تموم شد)")

    def current_files(self) -> List[int]:
        return list(self._loaded.keys())

    def get_tensor(self, fi: int) -> torch.Tensor:
        return self._loaded[fi]

    def notify_step_consumed(self, file_ix_used: List[int]) -> bool:
        for fi in set(file_ix_used):
            if fi in self._steps_left:
                self._steps_left[fi] -= 1

        finished = [fi for fi, s in self._steps_left.items() if s <= 0]
        for fi in finished:
            self._offload_file(fi)

        self._fill_pool()

        if not self._loaded and self._order_pos >= len(self._file_order):
            if len(self._skipped_this_epoch) >= len(self._file_order):
                raise ValueError(
                    f"هیچ‌کدوم از {len(self._file_order)} فایل دیتاست توی این epoch "
                    f"قابل استفاده نبودن. آموزش امکان‌پذیر نیست."
                )
            self._start_new_epoch()
            return True
        return False

    def _start_new_epoch(self) -> None:
        self.epoch += 1
        self._order_pos = 0
        self._steps_left = {}
        self._skipped_this_epoch = set()
        if self.verbose:
            print(f"[pool] === epoch {self.epoch} کامل شد؛ چرخه‌ی جدید از فایل اول شروع می‌شه ===")
        self._fill_pool()

    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "order_pos": self._order_pos,
            "steps_left": dict(self._steps_left),
            "skipped_this_epoch": list(self._skipped_this_epoch),
        }

    def load_state_dict(self, state: dict) -> None:
        self.epoch = state["epoch"]
        self._order_pos = state["order_pos"]
        self._steps_left = {int(k): v for k, v in state["steps_left"].items()}
        self._loaded = {}
        self._loaded_bytes = 0
        self._skipped_this_epoch = set(state.get("skipped_this_epoch", []))
        for fi in list(self._steps_left.keys()):
            success = self._load_file(fi)
            if not success:
                print(f"هشدار: فایل {fi} در checkpoint فعال بود ولی الان لود نشد.")
                del self._steps_left[fi]
                self._skipped_this_epoch.add(fi)
        self._fill_pool()


def get_batch(
    pool: DatasetPool, seq_length: int, batch_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    نمونه‌گیری از فایل‌های داخل pool. بهینه‌سازی: تنسورها رو از اول روی
    دستگاه مقصد اختصاص می‌دیم و مستقیماً پر می‌کنیم (کم کردن overhead پایتون).
    """
    usable = [fi for fi in pool.current_files() if len(pool.get_tensor(fi)) >= seq_length + 1]
    if not usable:
        raise ValueError(
            f"هیچ فایلی در pool فعلی به‌اندازه‌ی کافی (حداقل {seq_length + 1} توکن) "
            f"بزرگ نیست تا با seq_length={seq_length} قابل استفاده باشه."
        )

    lengths = torch.tensor([len(pool.get_tensor(fi)) for fi in usable], dtype=torch.float)
    file_weights = lengths / lengths.sum()
    n_files = len(usable)

    if batch_size >= n_files:
        guaranteed = list(range(n_files))
        remaining = batch_size - n_files
        if remaining > 0:
            extra = torch.multinomial(file_weights, remaining, replacement=True).tolist()
        else:
            extra = []
        pick_ix = guaranteed + extra
    else:
        pick_ix = torch.multinomial(file_weights, batch_size, replacement=True).tolist()

    file_ix_used = [usable[pi] for pi in pick_ix]
    x = torch.empty(batch_size, seq_length, dtype=torch.long, device=device)
    y = torch.empty(batch_size, seq_length, dtype=torch.long, device=device)
    for b, fi in enumerate(file_ix_used):
        d = pool.get_tensor(fi)
        max_start = len(d) - seq_length
        i = torch.randint(max_start, (1,)).item()
        x[b].copy_(d[i:i + seq_length])
        y[b].copy_(d[i + 1:i + seq_length + 1])
    return x, y, file_ix_used


# ================== Self-Attention ==================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        seq_length: int,
        dropout: float,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_length, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 20,
    ) -> torch.Tensor:
        if idx.shape[1] == 0:
            raise ValueError("Prompt نمی‌تونه خالی باشه (idx.shape[1] == 0)")
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat([idx, next_id], dim=1)
                continue

            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        if was_training:
            self.train()
        return idx


# ================== نرخ یادگیری ==================
def get_lr(step: int, total_steps: int) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


# ================== AdamW ==================
def configure_optimizer(
    model: nn.Module, weight_decay: float, learning_rate: float
) -> torch.optim.Optimizer:
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate, betas=(0.9, 0.999),
    )


# ================== ذخیره/بارگذاری checkpoint ==================
def _atomic_torch_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    pool: DatasetPool,
    epoch: int,
    step: int,
    steps_per_epoch_total: int,
    tokenizer: "Tokenizer",
) -> None:
    underlying = unwrap_model(model)
    _atomic_torch_save({
        "model_state": underlying.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "pool_state": pool.state_dict(),
        "epoch": epoch,
        "step": step,
        "steps_per_epoch_total": steps_per_epoch_total,
        "vocab_size": underlying.vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "seq_length": seq_length,
        "dropout_rate": dropout_rate,
        "tokenizer_path": tokenizer_path,
    }, checkpoint_path)


def load_checkpoint(txt_files: List[str]) -> Tuple[
    "TinyTransformer", torch.optim.Optimizer, DatasetPool, int, int, int,
    "Tokenizer"
]:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if ckpt["seq_length"] != seq_length:
        raise ValueError(
            f"seq_length ناسازگار: این checkpoint با seq_length={ckpt['seq_length']} "
            f"ساخته شده ولی تنظیمات فعلی seq_length={seq_length} است."
        )

    fresh_steps_per_epoch_total = compute_steps_per_epoch_total(txt_files)
    if ckpt["steps_per_epoch_total"] != fresh_steps_per_epoch_total:
        raise ValueError(
            f"ترکیب فایل‌های dataset/ از زمان ساخت این checkpoint عوض شده: "
            f"steps_per_epoch_total ذخیره‌شده={ckpt['steps_per_epoch_total']} ولی "
            f"مقدار تازه‌محاسبه‌شده از فایل‌های فعلی={fresh_steps_per_epoch_total}."
        )

    tokenizer = Tokenizer.from_file(ckpt["tokenizer_path"])

    model = TinyTransformer(
        ckpt["vocab_size"], ckpt["d_model"], ckpt["n_layers"],
        ckpt["n_heads"], ckpt["d_ff"], ckpt["seq_length"], ckpt["dropout_rate"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    optimizer = configure_optimizer(model, weight_decay, learning_rate)
    optimizer.load_state_dict(ckpt["optimizer_state"])

    pool = DatasetPool(
        txt_files, tokenizer, seq_length, max_ram_mb, fresh_steps_per_epoch_total
    )
    pool.load_state_dict(ckpt["pool_state"])

    return (
        model, optimizer, pool, ckpt["epoch"], ckpt["step"],
        fresh_steps_per_epoch_total, tokenizer,
    )


# ================== آموزش ==================
def train(
    model: nn.Module,
    pool: DatasetPool,
    steps_per_epoch_total: int,
    tokenizer: "Tokenizer",
    optimizer: Optional[torch.optim.Optimizer] = None,
    start_epoch: int = 0,
    start_step: int = 0,
    checkpoint_every_steps: int = 200,
) -> Tuple[nn.Module, torch.optim.Optimizer, int]:
    total_steps = steps_per_epoch_total * epochs
    amp_enabled = use_amp and device.type == "cpu" and torch.cpu.is_bf16_supported()

    if optimizer is None:
        optimizer = configure_optimizer(model, weight_decay, learning_rate)

    step = start_step
    epoch = start_epoch
    try:
        while epoch < epochs:
            total_loss = 0.0
            steps_this_epoch = 0
            pbar = tqdm(desc=f"Epoch {epoch + 1}/{epochs}", total=steps_per_epoch_total)
            epoch_finished = False

            while not epoch_finished:
                if not pool.current_files():
                    epoch_finished = pool.notify_step_consumed([])
                    if epoch_finished:
                        break
                    if not pool.current_files():
                        raise RuntimeError(
                            "Pool خالیه و هیچ فایل جدیدی لود نشد. "
                            "ممکنه همه فایل‌های باقی‌مانده نامعتبر باشن."
                        )

                xb, yb, file_ix_used = get_batch(pool, seq_length, batch_size, device)

                lr = get_lr(step, total_steps)
                for g in optimizer.param_groups:
                    g["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                    _, loss = model(xb, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                total_loss += loss.item()
                step += 1
                steps_this_epoch += 1
                pbar.update(1)

                epoch_finished = pool.notify_step_consumed(file_ix_used)

                if step % checkpoint_every_steps == 0:
                    save_checkpoint(
                        model, optimizer, pool, epoch, step,
                        steps_per_epoch_total, tokenizer,
                    )

            pbar.close()
            avg_loss = total_loss / max(1, steps_this_epoch)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - LR: {lr:.6f}")
            epoch += 1
            save_checkpoint(
                model, optimizer, pool, epoch, step,
                steps_per_epoch_total, tokenizer,
            )

    except KeyboardInterrupt:
        print("\nآموزش با Ctrl+C متوقف شد - در حال ذخیره...")
        save_checkpoint(
            model, optimizer, pool, epoch, step,
            steps_per_epoch_total, tokenizer,
        )
        underlying = unwrap_model(model)
        save_model(underlying, tokenizer)
        print("ذخیره شد. برای ادامه‌ی آموزش، دوباره همین اسکریپت رو اجرا کن.")
        raise SystemExit(0)

    return model, optimizer, step


# ================== ذخیره / بارگذاری ==================
def save_model(
    model: nn.Module,
    tokenizer: "Tokenizer",
    trained_epochs: Optional[int] = None,
) -> None:
    if trained_epochs is None:
        trained_epochs = epochs
    underlying = unwrap_model(model)
    _atomic_torch_save(underlying.state_dict(), model_path)
    config = {
        "vocab_size": underlying.vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "seq_length": seq_length,
        "dropout_rate": dropout_rate,
        "trained_epochs": trained_epochs,
        "tokenizer_path": tokenizer_path,
    }
    tmp = config_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False)
    os.replace(tmp, config_path)


def load_model() -> Tuple["TinyTransformer", "Tokenizer", int]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if config["seq_length"] != seq_length:
        raise ValueError(
            f"seq_length ناسازگار: مدل ذخیره‌شده با seq_length={config['seq_length']} "
            f"ساخته شده ولی تنظیمات فعلی seq_length={seq_length} است."
        )

    tokenizer = Tokenizer.from_file(config["tokenizer_path"])
    model = TinyTransformer(
        config["vocab_size"], config["d_model"], config["n_layers"],
        config["n_heads"], config["d_ff"], config["seq_length"], config["dropout_rate"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    trained_epochs = config.get("trained_epochs", epochs)
    return model, tokenizer, trained_epochs


# ================== چت تعاملی ==================
def chat_loop(
    model: nn.Module, tokenizer: "Tokenizer"
) -> None:
    print("\n" + "=" * 50)
    print("چت آماده است (خروج: 'خروج' یا 'exit')")
    print("=" * 50)
    while True:
        user_input = input("\nشما: ").strip()
        if user_input.lower() in ("exit", "quit", "خروج"):
            print("خداحافظ!")
            break
        if not user_input:
            continue

        # توکنایز کردن ورودی کاربر
        encoding = tokenizer.encode(user_input.lower())
        if not encoding.ids:
            print("مدل: (این کاراکترها رو نمی‌شناسم)")
            continue

        idx = torch.tensor([encoding.ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=150, temperature=0.8, top_k=top_k)
        out_ids = out[0, len(encoding.ids):].tolist()
        generated = decode_tokens(out_ids, tokenizer)
        print(f"مدل: {generated}")


def _maybe_compile(model: nn.Module, pool: DatasetPool) -> nn.Module:
    if not use_compile:
        return model
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")
        dummy_x, dummy_y, _ = get_batch(pool, seq_length, min(batch_size, 2), device)
        _, dummy_loss = compiled_model(dummy_x, dummy_y)
        dummy_loss.backward()
        model.zero_grad(set_to_none=True)
        print("torch.compile فعال شد (با موفقیت تست شد)")
        return compiled_model
    except Exception as e:
        print(f"torch.compile کار نکرد ({type(e).__name__})، ادامه بدون آن")
        return model


# ================== اجرا ==================
if __name__ == "__main__":
    txt_files = find_txt_files("dataset")
    print(f"{len(txt_files)} فایل .txt در پوشه‌ی dataset/ پیدا شد")

    # ساخت یا لود توکنایزر BPE
    tokenizer = build_or_load_tokenizer(txt_files, vocab_size)
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"اندازه واقعی vocab: {actual_vocab_size} (درخواست‌شده: {vocab_size})")

    if os.path.exists(checkpoint_path):
        print("Checkpoint ناقص پیدا شد، در حال ادامه‌ی آموزش از همون‌جا...")
        (
            model, optimizer, pool, start_epoch, start_step,
            steps_per_epoch_total, tokenizer,
        ) = load_checkpoint(txt_files)
        print(f"ادامه از epoch {start_epoch + 1}, step {start_step}")

        model = _maybe_compile(model, pool)

        if start_epoch >= epochs:
            print(f"این checkpoint از قبل {epochs} epoch رو تموم کرده.")
            print("اگه می‌خوای بیشتر آموزش بدی، عدد epochs رو زیاد کن و دوباره اجرا کن.")
            underlying = unwrap_model(model)
            save_model(underlying, tokenizer, trained_epochs=epochs)
        else:
            model, optimizer, final_step = train(
                model, pool, steps_per_epoch_total, tokenizer,
                optimizer=optimizer, start_epoch=start_epoch, start_step=start_step,
            )
            os.remove(checkpoint_path)
            underlying = unwrap_model(model)
            save_model(underlying, tokenizer, trained_epochs=epochs)
            print(f"مدل نهایی ذخیره شد در {model_path}")

    elif os.path.exists(model_path) and os.path.exists(config_path):
        print("مدل ذخیره‌شده پیدا شد، در حال بارگذاری...")
        model, tokenizer, trained_epochs = load_model()
        steps_per_epoch_total = compute_steps_per_epoch_total(txt_files)

        if trained_epochs < epochs:
            print(f"مدل {trained_epochs} epoch آموزش دیده ولی epochs فعلی = {epochs}. ادامه‌ی آموزش...")
            pool = DatasetPool(txt_files, tokenizer, seq_length, max_ram_mb, steps_per_epoch_total)
            model = _maybe_compile(model, pool)
            model, optimizer, final_step = train(
                model, pool, steps_per_epoch_total, tokenizer,
                start_epoch=trained_epochs,
            )
            underlying = unwrap_model(model)
            save_model(underlying, tokenizer, trained_epochs=epochs)
            print(f"مدل نهایی ذخیره شد در {model_path}")
        else:
            print(f"مدل کامل آموزش دیده ({trained_epochs}/{epochs} epoch) — مستقیم چت.")

    else:
        print("آموزش مدل از ابتدا...")
        steps_per_epoch_total = compute_steps_per_epoch_total(txt_files)
        total_bytes = sum(os.path.getsize(fp) for fp in txt_files)
        print(
            f"توکنایزر BPE آماده: {actual_vocab_size} توکن | "
            f"حجم کل دیتاست: {total_bytes / 1_000_000:.1f} مگابایت | "
            f"سقف pool: {max_ram_mb} مگابایت | "
            f"~{steps_per_epoch_total} step به‌ازای هر epoch"
        )

        model = TinyTransformer(actual_vocab_size, d_model, n_layers, n_heads, d_ff, seq_length, dropout_rate).to(device)

        pool = DatasetPool(txt_files, tokenizer, seq_length, max_ram_mb, steps_per_epoch_total)
        model = _maybe_compile(model, pool)

        model, optimizer, final_step = train(model, pool, steps_per_epoch_total, tokenizer)
        underlying = unwrap_model(model)
        save_model(underlying, tokenizer, trained_epochs=epochs)
        print(f"مدل ذخیره شد در {model_path}")
        model = underlying

    chat_loop(model, tokenizer)
