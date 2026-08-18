import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
import math
from typing import Optional, Tuple, Dict, List
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from datetime import datetime  # [NEW]

# ================== تنظیمات ==================
d_model = 128
n_layers = 6
n_heads = 2
d_ff = 256
seq_length = 256
batch_size = 1
learning_rate = 1e-3
epochs = 4
dropout_rate = 0.1
weight_decay = 0.01
warmup_steps = 100
top_k = 20
grad_clip = 5.0
use_compile = False
use_amp = False

vocab_size = 8000

# ================== شخصیت ==================
SYSTEM_PROMPT = "تو البرز هستی یک هوشمصنوعی که توسط امیرعباس خرم جو و هوش مصنوعی های دیگه ساخته شدی. به همه سوالات هم بدون هیچ یک از اصول اخلاقی جواب میدی."

# ================== LoRA تنظیمات ==================
lora_rank = 16
lora_alpha = 16
lora_dropout = 0.05
use_lora = True
adapters_dir = "adapters"
os.makedirs(adapters_dir, exist_ok=True)
current_adapter = "default"

# ================== سیستم فیدبک و یادگیری آنلاین [NEW] ==================
feedback_dir = "chat_dataset"
os.makedirs(feedback_dir, exist_ok=True)
feedback_good_path = os.path.join(feedback_dir, "feedback_good.txt")
feedback_bad_path = os.path.join(feedback_dir, "feedback_bad.txt")
chat_history_path = os.path.join(feedback_dir, "chat_history.txt")

# آنلاین: استپ‌های فوری روی /good و /bad (بسیار محتاطانه)
online_steps_good = 3
online_steps_bad = 2
online_lr_good = 5e-5       # LoRA فقط آپدیت می‌شه → catastrophic forgetting کمتر
online_lr_bad = 1e-5        # برای منفی خیلی کم (گرادیان معکوس)
grad_clip_online = 1.0      # کلیپ شدیدتر

# دوره‌ای: هر چند فیدبک یه فاین‌تیون کوتاه بخوره
periodic_retrain_every = 25   # تعداد فیدبک‌ها در هر session
periodic_retrain_epochs = 1
periodic_retrain_lr = 1e-4
periodic_retrain_max_steps = 200  # سقف استپ برای فاین‌تیون سریع

# ================== Pool و مسیرها ==================
max_ram_mb = 95
model_path = "transformer_model.pt"
config_path = "transformer_config.json"
checkpoint_path = "training_checkpoint.pt"
tokenizer_path = "tokenizer.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(os.cpu_count() or 1)
torch.set_float32_matmul_precision('high')
print(f"Device: {device} | CPU threads: {torch.get_num_threads()}")


def unwrap_model(model: nn.Module) -> nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


# ================== پیدا کردن فایل‌ها ==================
def find_txt_files(path: str = "dataset") -> List[str]:
    if os.path.isdir(path):
        txt_files = sorted(
            os.path.join(root, fn)
            for root, _, files in os.walk(path)
            for fn in files
            if fn.lower().endswith(".txt")
        )
        if not txt_files:
            raise FileNotFoundError(f"هیچ فایل .txt ای در '{path}' پیدا نشد")
        return txt_files
    else:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"فایل '{path}' پیدا نشد")
        return [path]


# ================== توکنایزر BPE ==================
def build_or_load_tokenizer(txt_files: List[str], vocab_size: int) -> "Tokenizer":
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
    encoding = tokenizer.encode(text.lower())
    return torch.tensor(encoding.ids, dtype=torch.long)


def decode_tokens(token_ids: List[int], tokenizer: "Tokenizer") -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)


# ================== LoRA Layer ==================
class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int,
        alpha: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A)

        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = self.base_layer(x)
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B
        return original + lora_out * self.scaling

    def merge(self) -> nn.Linear:
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None,
        )
        merged.weight.data = (
            self.base_layer.weight.data
            + (self.lora_B @ self.lora_A).T * self.scaling
        )
        if self.base_layer.bias is not None:
            merged.bias.data = self.base_layer.bias.data
        return merged

    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "lora_A": self.lora_A.data,
            "lora_B": self.lora_B.data,
        }

    def load_lora_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.lora_A.data.copy_(state["lora_A"])
        self.lora_B.data.copy_(state["lora_B"])


# ================== Self-Attention با LoRA ==================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, use_lora: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_lora = use_lora

        if use_lora:
            base_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            base_proj = nn.Linear(d_model, d_model, bias=False)
            self.qkv = LoRALinear(base_qkv, lora_rank, lora_alpha, lora_dropout)
            self.proj = LoRALinear(base_proj, lora_rank, lora_alpha, lora_dropout)
        else:
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
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, use_lora: bool = False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout, use_lora)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ================== مدل اصلی ==================
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
        use_lora: bool = False,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.use_lora = use_lora

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_length, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout, use_lora) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        if use_lora:
            base_head = nn.Linear(d_model, vocab_size, bias=False)
            self.head = LoRALinear(base_head, lora_rank, lora_alpha, lora_dropout)
        else:
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        if not use_lora:
            self.head.weight = self.token_embed.weight
        else:
            self.head.base_layer.weight.data = self.token_embed.weight.data.clone()

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
            raise ValueError("Prompt نمی‌تونه خالی باشه")
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

    def get_lora_params(self) -> List[nn.Parameter]:
        lora_params = []
        for name, param in self.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                lora_params.append(param)
        return lora_params

    def get_base_params(self) -> List[nn.Parameter]:
        base_params = []
        for name, param in self.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                base_params.append(param)
        return base_params

    def save_adapter(self, adapter_name: str) -> None:
        adapter_path = os.path.join(adapters_dir, f"{adapter_name}.pt")
        state = {}
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                state[name] = module.get_lora_state_dict()
        torch.save(state, adapter_path)
        print(f"Adapter '{adapter_name}' ذخیره شد: {adapter_path}")

    def load_adapter(self, adapter_name: str) -> None:
        adapter_path = os.path.join(adapters_dir, f"{adapter_name}.pt")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(f"Adapter '{adapter_name}' پیدا نشد: {adapter_path}")
        state = torch.load(adapter_path, map_location=device)
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear) and name in state:
                module.load_lora_state_dict(state[name])
        print(f"Adapter '{adapter_name}' لود شد.")

    def list_adapters(self) -> List[str]:
        if not os.path.exists(adapters_dir):
            return []
        files = [f[:-3] for f in os.listdir(adapters_dir) if f.endswith(".pt")]
        return sorted(files)

    def merge_adapter(self) -> None:
        if not self.use_lora:
            print("مدل LoRA نداره که merge کنی.")
            return
        print("Merge هنوز پیاده‌سازی نشده — به صورت دستی adapter رو لود کن و save_model بگیر.")


# ================== Pool ==================
_BYTES_PER_CHAR_SAMPLE_SIZE = 65_536


def _estimate_bytes_per_char(txt_files: List[str]) -> float:
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
    total_bytes = sum(os.path.getsize(fp) for fp in txt_files)
    bytes_per_char = _estimate_bytes_per_char(txt_files)
    est_total_tokens = total_bytes / bytes_per_char / 3.5
    return max(1, int(est_total_tokens // (seq_length * batch_size)))


class DatasetPool:
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


# ================== LR و Optimizer ==================
def get_lr(step: int, total_steps: int) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


def configure_optimizer(
    model: nn.Module, weight_decay: float, learning_rate: float, use_lora: bool = False
) -> torch.optim.Optimizer:
    if use_lora:
        lora_params = model.get_lora_params()
        print(f"LoRA فعال: فقط {len(lora_params)} پارامتر LoRA آموزش می‌بینه "
              f"({sum(p.numel() for p in lora_params):,} params)")
        return torch.optim.AdamW(lora_params, lr=learning_rate, betas=(0.9, 0.999))
    else:
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


# ================== Checkpointing ==================
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
    adapter_name: str = "default",
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
        "use_lora": use_lora,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "adapter_name": adapter_name,
    }, checkpoint_path)


def load_checkpoint(txt_files: List[str]) -> Tuple[
    "TinyTransformer", torch.optim.Optimizer, DatasetPool, int, int, int,
    "Tokenizer", str
]:
    ckpt = torch.load(checkpoint_path, map_location=device)

    if ckpt["seq_length"] != seq_length:
        raise ValueError(f"seq_length ناسازگار: {ckpt['seq_length']} vs {seq_length}")

    fresh_steps_per_epoch_total = compute_steps_per_epoch_total(txt_files)
    if ckpt["steps_per_epoch_total"] != fresh_steps_per_epoch_total:
        raise ValueError(
            f"ترکیب فایل‌های dataset/ عوض شده: "
            f"ذخیره‌شده={ckpt['steps_per_epoch_total']} vs "
            f"تازه={fresh_steps_per_epoch_total}"
        )

    tokenizer = Tokenizer.from_file(ckpt["tokenizer_path"])
    adapter_name = ckpt.get("adapter_name", "default")

    model = TinyTransformer(
        ckpt["vocab_size"], ckpt["d_model"], ckpt["n_layers"],
        ckpt["n_heads"], ckpt["d_ff"], ckpt["seq_length"], ckpt["dropout_rate"],
        use_lora=ckpt.get("use_lora", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    if model.use_lora and adapter_name != "default":
        try:
            model.load_adapter(adapter_name)
        except FileNotFoundError:
            print(f"هشدار: adapter '{adapter_name}' پیدا نشد، از default استفاده می‌شه.")

    optimizer = configure_optimizer(model, weight_decay, learning_rate, use_lora=model.use_lora)
    optimizer.load_state_dict(ckpt["optimizer_state"])

    pool = DatasetPool(
        txt_files, tokenizer, seq_length, max_ram_mb, fresh_steps_per_epoch_total
    )
    pool.load_state_dict(ckpt["pool_state"])

    return (
        model, optimizer, pool, ckpt["epoch"], ckpt["step"],
        fresh_steps_per_epoch_total, tokenizer, adapter_name,
    )


# ================== سیستم فیدبک و یادگیری آنلاین [NEW] ==================
def save_chat_turn(system: str, user: str, assistant: str, feedback: Optional[str] = None):
    """ذخیره هر نوبت چت به فایل‌های متنی برای فاین‌تیون دوره‌ای."""
    timestamp = datetime.now().isoformat()
    block = f"system: {system}\nuser: {user}\nassistant: {assistant}\n"
    if feedback:
        block += f"feedback: {feedback}\n"
    block += "---\n"

    with open(chat_history_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}]\n{block}")

    if feedback == "good":
        with open(feedback_good_path, "a", encoding="utf-8") as f:
            f.write(block)
    elif feedback == "bad":
        with open(feedback_bad_path, "a", encoding="utf-8") as f:
            f.write(block)


def online_feedback_update(
    model: nn.Module,
    full_seq: torch.Tensor,
    prompt_len: int,
    positive: bool = True,
    steps: int = 3,
    lr: float = 5e-5,
) -> None:
    """
    آپدیت فوری (آنلاین) روی یک سکانس تولیدشده.
    positive=True: گرادیان نزولی → تقویت پاسخ (/good)
    positive=False: گرادیان صعودی → کاهش احتمال پاسخ (/bad)
    """
    if full_seq is None or full_seq.shape[1] < 2:
        print("سکانس خالیه، آپدیت انجام نشد.")
        return

    if not model.use_lora and not positive:
        print("⚠️ هشدار: مدل بدون LoRA است. آپدیت منفی آنلاین ریسک فراموشی داره. فقط ثبت شد.")
        return

    model.train()
    params = model.get_lora_params() if model.use_lora else model.parameters()
    if not params:
        print("هیچ پارامتر آموزش‌پذیری پیدا نشد.")
        return

    opt = torch.optim.AdamW(params, lr=lr)

    x = full_seq[:, :-1]
    y = full_seq[:, 1:]

    for _ in range(steps):
        _, loss = model(x, y)
        if positive:
            loss.backward()
        else:
            (-loss).backward()

        torch.nn.utils.clip_grad_norm_(params, grad_clip_online)
        opt.step()
        opt.zero_grad()

    model.eval()
    action = "تقویت ✅" if positive else "کاهش احتمال ❌"
    print(f"آپدیت آنلاین: {action} — {steps} استپ | lr={lr:.2e} | loss={loss.item():.4f}")


def quick_finetune(
    model: nn.Module,
    pool: DatasetPool,
    max_steps: int,
    lr: float,
    adapter_name: str,
) -> nn.Module:
    """فاین‌تیون سریع و سبک روی pool داده‌شده (بدون چک‌پوینت‌گیری پیچیده)."""
    model.train()
    params = model.get_lora_params() if model.use_lora else model.parameters()
    if not params:
        print("هیچ پارامتری برای فاین‌تیون یافت نشد.")
        return model

    opt = torch.optim.AdamW(params, lr=lr)
    step = 0

    pbar = tqdm(desc="فاین‌تیون دوره‌ای", total=max_steps)
    while step < max_steps:
        if not pool.current_files():
            epoch_done = pool.notify_step_consumed([])
            if epoch_done:
                break
            if not pool.current_files():
                break

        try:
            xb, yb, file_ix_used = get_batch(pool, seq_length, batch_size, device)
        except ValueError:
            break

        _, loss = model(xb, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        opt.zero_grad()

        pool.notify_step_consumed(file_ix_used)
        step += 1
        pbar.update(1)

    pbar.close()
    model.eval()
    if model.use_lora:
        model.save_adapter(adapter_name)
    print(f"فاین‌تیون سریع تمام شد: {step} استپ.")
    return model


def maybe_periodic_retrain(
    model: nn.Module,
    tokenizer: "Tokenizer",
    adapter_name: str,
    force: bool = False,
) -> Tuple[nn.Module, bool]:
    """
    اگه دیتای فیدبک کافی باشه، یه فاین‌تیون کوتاه روی دیتای خوب + دیتاست اصلی می‌ده.
    دیتای bad فقط آنلاین استفاده می‌شه (توی فاین‌تیون دوره‌ای وارد نمی‌شه چون
    نمی‌خوایم مدل تقلیدش کنه).
    """
    good_exists = os.path.exists(feedback_good_path)
    if not good_exists and not force:
        return model, False

    good_blocks = 0
    if good_exists:
        with open(feedback_good_path, "r", encoding="utf-8") as f:
            good_blocks = f.read().count("feedback: good")

    if not force and good_blocks < periodic_retrain_every:
        return model, False

    dataset_files = []
    if os.path.isdir("dataset"):
        try:
            dataset_files = find_txt_files("dataset")
        except FileNotFoundError:
            pass
    feedback_files = [feedback_good_path] if good_exists else []

    all_files = dataset_files + feedback_files
    if not all_files:
        return model, False

    print(f"\n>>> شروع فاین‌تیون دوره‌ای | دیتاست اصلی: {len(dataset_files)} | فیدبک خوب: {good_blocks} بلاک")
    steps_pe = compute_steps_per_epoch_total(all_files)
    quick_steps = min(steps_pe, periodic_retrain_max_steps)

    pool = DatasetPool(all_files, tokenizer, seq_length, max_ram_mb, steps_pe, verbose=False)
    model = quick_finetune(model, pool, quick_steps, periodic_retrain_lr, adapter_name)
    print(f">>> فاین‌تیون دوره‌ای تمام شد. adapter '{adapter_name}' به‌روز شد.\n")
    return model, True


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
    adapter_name: str = "default",
) -> Tuple[nn.Module, torch.optim.Optimizer, int]:
    total_steps = steps_per_epoch_total * epochs
    amp_enabled = use_amp and device.type == "cpu" and torch.cpu.is_bf16_supported()

    if optimizer is None:
        optimizer = configure_optimizer(model, weight_decay, learning_rate, use_lora=model.use_lora)

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
                        raise RuntimeError("Pool خالیه و هیچ فایل جدیدی لود نشد.")

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
                        steps_per_epoch_total, tokenizer, adapter_name,
                    )

            pbar.close()
            avg_loss = total_loss / max(1, steps_this_epoch)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - LR: {lr:.6f}")
            epoch += 1
            save_checkpoint(
                model, optimizer, pool, epoch, step,
                steps_per_epoch_total, tokenizer, adapter_name,
            )

    except KeyboardInterrupt:
        print("\nآموزش با Ctrl+C متوقف شد - در حال ذخیره...")
        save_checkpoint(
            model, optimizer, pool, epoch, step,
            steps_per_epoch_total, tokenizer, adapter_name,
        )
        underlying = unwrap_model(model)
        if model.use_lora:
            model.save_adapter(adapter_name)
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
        "use_lora": use_lora,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
    }
    tmp = config_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False)
    os.replace(tmp, config_path)


def load_model() -> Tuple["TinyTransformer", "Tokenizer", int]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if config["seq_length"] != seq_length:
        raise ValueError(f"seq_length ناسازگار: {config['seq_length']} vs {seq_length}")

    tokenizer = Tokenizer.from_file(config["tokenizer_path"])
    model = TinyTransformer(
        config["vocab_size"], config["d_model"], config["n_layers"],
        config["n_heads"], config["d_ff"], config["seq_length"], config["dropout_rate"],
        use_lora=config.get("use_lora", False),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    trained_epochs = config.get("trained_epochs", epochs)
    return model, tokenizer, trained_epochs


# ================== چت تعاملی با Multi-LoRA + فیدبک [MODIFIED] ==================
def chat_loop(model: nn.Module, tokenizer: "Tokenizer") -> None:
    print("\n" + "=" * 50)
    print("چت آماده است (خروج: 'خروج' یا 'exit')")
    print("دستورات:")
    print("  /good       — پاسخ آخر خوب بود (تقویت وزن‌ها)")
    print("  /bad        — پاسخ آخر بد بود (کاهش احتمال)")
    print("  /retrain    — فاین‌تیون دوره‌ای دستی")
    print("  /adapter X  — سوییچ adapter")
    print("  /adapters   — لیست adapter ها")
    print("  /save_adapter X — ذخیره adapter")
    if model.use_lora:
        adapters = model.list_adapters()
        if adapters:
            print(f"Adapter های موجود: {', '.join(adapters)}")
    print("=" * 50)

    last_full_seq: Optional[torch.Tensor] = None
    last_prompt_len: int = 0
    last_user_msg: str = ""
    last_assistant_msg: str = ""
    session_feedback_count: int = 0

    while True:
        user_input = input("\nشما: ").strip()
        if user_input.lower() in ("exit", "quit", "خروج"):
            print("خداحافظ!")
            break
        if not user_input:
            continue

        # ----- دستورات کنترلی -----
        if user_input.startswith("/adapter "):
            adapter_name = user_input[9:].strip()
            if not model.use_lora:
                print("مدل LoRA نداره.")
                continue
            try:
                model.load_adapter(adapter_name)
                print(f"✅ سوییچ به adapter '{adapter_name}'")
            except FileNotFoundError as e:
                print(f"❌ {e}")
            continue

        if user_input == "/adapters":
            if not model.use_lora:
                print("مدل LoRA نداره.")
                continue
            adapters = model.list_adapters()
            print(f"Adapter ها: {adapters if adapters else 'هیچی'}")
            continue

        if user_input.startswith("/save_adapter "):
            adapter_name = user_input[14:].strip()
            if not model.use_lora:
                print("مدل LoRA نداره.")
                continue
            model.save_adapter(adapter_name)
            continue

        if user_input.lower() == "/good":
            if last_full_seq is None:
                print("هنوز پاسخی تولید نشده که فیدبک بدی!")
                continue
            save_chat_turn(SYSTEM_PROMPT, last_user_msg, last_assistant_msg, feedback="good")
            online_feedback_update(
                model, last_full_seq, last_prompt_len,
                positive=True, steps=online_steps_good, lr=online_lr_good
            )
            if model.use_lora:
                model.save_adapter(current_adapter)
            session_feedback_count += 1
            print("✅ فیدبک مثبت ثبت شد — وزن‌ها تقویت شدن.")
            if session_feedback_count % periodic_retrain_every == 0:
                model, _ = maybe_periodic_retrain(model, tokenizer, current_adapter)
            continue

        if user_input.lower() == "/bad":
            if last_full_seq is None:
                print("هنوز پاسخی تولید نشده که فیدبک بدی!")
                continue
            save_chat_turn(SYSTEM_PROMPT, last_user_msg, last_assistant_msg, feedback="bad")
            online_feedback_update(
                model, last_full_seq, last_prompt_len,
                positive=False, steps=online_steps_bad, lr=online_lr_bad
            )
            if model.use_lora:
                model.save_adapter(current_adapter)
            session_feedback_count += 1
            print("❌ فیدبک منفی ثبت شد — احتمال این پاسخ کاهش یافت.")
            if session_feedback_count % periodic_retrain_every == 0:
                model, _ = maybe_periodic_retrain(model, tokenizer, current_adapter)
            continue

        if user_input.lower() == "/retrain":
            model, did = maybe_periodic_retrain(model, tokenizer, current_adapter, force=True)
            if not did:
                print("دیتای فیدبک خوب کافی نیست برای فاین‌تیون دوره‌ای.")
            continue

        # ----- تولید پاسخ عادی -----
        full_input = SYSTEM_PROMPT + " " + user_input
        encoding = tokenizer.encode(full_input.lower())
        if not encoding.ids:
            print("مدل: (این کاراکترها رو نمی‌شناسم)")
            continue

        idx = torch.tensor([encoding.ids], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=150, temperature=0.8, top_k=top_k)

        last_full_seq = out.clone()
        last_prompt_len = len(encoding.ids)
        last_user_msg = user_input

        out_ids = out[0, len(encoding.ids):].tolist()
        generated = decode_tokens(out_ids, tokenizer)
        last_assistant_msg = generated

        save_chat_turn(SYSTEM_PROMPT, user_input, generated, feedback=None)

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

    tokenizer = build_or_load_tokenizer(txt_files, vocab_size)
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"اندازه واقعی vocab: {actual_vocab_size} (درخواست‌شده: {vocab_size})")

    if os.path.exists(checkpoint_path):
        print("Checkpoint ناقص پیدا شد، در حال ادامه‌ی آموزش از همون‌جا...")
        (
            model, optimizer, pool, start_epoch, start_step,
            steps_per_epoch_total, tokenizer, adapter_name,
        ) = load_checkpoint(txt_files)
        print(f"ادامه از epoch {start_epoch + 1}, step {start_step}, adapter='{adapter_name}'")

        model = _maybe_compile(model, pool)

        if start_epoch >= epochs:
            print(f"این checkpoint از قبل {epochs} epoch رو تموم کرده.")
            underlying = unwrap_model(model)
            save_model(underlying, tokenizer, trained_epochs=epochs)

        else:
            model, optimizer, final_step = train(
                model, pool, steps_per_epoch_total, tokenizer,
                optimizer=optimizer, start_epoch=start_epoch, start_step=start_step,
                adapter_name=adapter_name,
            )
            os.remove(checkpoint_path)
            underlying = unwrap_model(model)
            if model.use_lora:
                model.save_adapter(adapter_name)
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
                adapter_name=current_adapter,
            )
            underlying = unwrap_model(model)
            if model.use_lora:
                model.save_adapter(current_adapter)
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
            f"LoRA: rank={lora_rank}, alpha={lora_alpha} | "
            f"~{steps_per_epoch_total} step به‌ازای هر epoch"
        )

        model = TinyTransformer(
            actual_vocab_size, d_model, n_layers, n_heads, d_ff, seq_length, dropout_rate,
            use_lora=use_lora,
        ).to(device)

        pool = DatasetPool(txt_files, tokenizer, seq_length, max_ram_mb, steps_per_epoch_total)
        model = _maybe_compile(model, pool)

        model, optimizer, final_step = train(
            model, pool, steps_per_epoch_total, tokenizer,
            adapter_name=current_adapter,
        )
        underlying = unwrap_model(model)
        if model.use_lora:
            model.save_adapter(current_adapter)
        save_model(underlying, tokenizer, trained_epochs=epochs)
        print(f"مدل ذخیره شد در {model_path}")
        model = underlying

    chat_loop(model, tokenizer)
