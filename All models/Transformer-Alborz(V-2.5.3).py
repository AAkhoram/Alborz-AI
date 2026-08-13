import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
import math

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

model_path = "transformer_model.pt"
config_path = "transformer_config.json"
checkpoint_path = "training_checkpoint.pt"  # برای ادامه‌ی آموزش بعد از قطعی برق/سیستم

# نکته: چون بسته‌ی PyTorch که نصب کردی نسخه‌ی CPU-only است (بدون CUDA کامپایل شده)،
# torch.cuda.is_available() همیشه False برمی‌گردونه و این خط همیشه cpu رو انتخاب می‌کنه؛
# (خود torch.compile محدود به CPU نیست و روی GPU هم کار می‌کنه، این محدودیت فقط
# مربوط به نسخه‌ی نصب‌شده‌ی خود PyTorch است، نه torch.compile)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# برخلاف NumPy (که تست کردیم و دیدیم عملا از چند هسته سود نمی‌بره)، PyTorch با MKL/oneDNN
# واقعا می‌تونه از چند هسته استفاده کنه - این خط دقیقا اون سوالی که قبلا پرسیدی رو جواب می‌ده
torch.set_num_threads(os.cpu_count() or 1)
print(f"Device: {device} | CPU threads: {torch.get_num_threads()}")


# ================== لود دیتاست ==================
def load_data(path="dataset"):
    """
    اگه path یک پوشه باشه، تمام فایل‌های .txt داخلش (و زیرپوشه‌ها) رو پیدا می‌کنه،
    به ترتیب الفبایی می‌خونه و با یک خط خالی بین‌شون به‌هم می‌چسبونه.
    اگه path یک فایل تکی باشه (مثلاً data.txt)، دقیقاً مثل قبل فقط همون رو می‌خونه.
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

        parts = []
        for fp in txt_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    parts.append(f.read())
            except UnicodeDecodeError:
                print(f"هشدار: فایل '{fp}' با UTF-8 خونده نشد و نادیده گرفته شد")
        text = "\n\n".join(parts).lower()
        print(f"{len(txt_files)} فایل از پوشه‌ی '{path}' خونده شد")
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().lower()

    chars = sorted(list(set(text)))
    if not chars:
        raise ValueError("متن لود شده خالیه - واژگانی برای ساخت مدل وجود نداره")
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    return text, chars, char_to_ix, ix_to_char


def get_batch(data, seq_length, batch_size, device):
    """
    به‌جای تقسیم دیتاست به تکه‌های ثابت (مثل نسخه‌ی NumPy)، هر بار نقاط شروع تصادفی
    انتخاب می‌کنیم. این باعث می‌شه مدل تمام offsetهای ممکن رو در طول آموزش ببینه،
    نه فقط تکه‌های از پیش تعیین‌شده - استفاده‌ی بهتر از داده، بدون هزینه‌ی اضافه.
    """
    max_start = len(data) - seq_length  # اصلاح off-by-one: قبلاً -1 اضافه داشت و آخرین کاراکتر رو هیچ‌وقت به‌عنوان target استفاده نمی‌کرد
    if max_start < 1:
        raise ValueError(
            f"دیتاست ({len(data)} کاراکتر) برای seq_length={seq_length} خیلی کوچیکه. "
            f"باید حداقل {seq_length + 1} کاراکتر داشته باشه."
        )
    ix = torch.randint(max_start, (batch_size,))
    x = torch.stack([data[i:i + seq_length] for i in ix])
    y = torch.stack([data[i + 1:i + seq_length + 1] for i in ix])
    return x.to(device), y.to(device)


# ================== Self-Attention (با Flash Attention روی سخت‌افزار پشتیبانی‌شده) ==================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # F.scaled_dot_product_attention پیاده‌سازی بهینه‌ی خود PyTorch است (شامل Flash
        # Attention روی سخت‌افزار پشتیبانی‌شده) - دقیقا همون عملیاتی که در نسخه‌ی NumPy
        # دستی پیاده‌سازی شده بود، ولی اینجا با کد C++ بهینه‌شده و معمولا چند برابر سریع‌تر
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        # Pre-LN: نرمال‌سازی قبل از هر زیرلایه (نه بعدش، مثل نسخه‌ی NumPy قبلی).
        # این دقیقا همون چیزیه که مشکل vanishing gradient با n_layers=12 در نسخه‌ی
        # NumPy رو حل می‌کنه - چون گرادیان از مسیر residual مستقیم و بدون مانع عبور
        # می‌کنه، مجبور نیست هر بار از یک LayerNorm که بعد از بلوک بود رد بشه.
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, seq_length, dropout):
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
        self.head = nn.Linear(d_model, vocab_size)  # bias=True پیش‌فرض، مثل bout در نسخه‌ی NumPy

        # Weight tying: دقیقا مثل نسخه‌ی NumPy، فقط وزن embedding و خروجی مشترکه؛
        # بایاس مستقل و جدا می‌مونه (چون در نسخه‌ی قبلی هم بایاس جدا بود)
        self.head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
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
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=20):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0:
                # temperature=0 یعنی greedy decoding (همیشه محتمل‌ترین کاراکتر)،
                # تقسیم بر صفر معنا نداره پس این حالت رو جدا مدیریت می‌کنیم
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


# ================== نرخ یادگیری (warmup + cosine decay) ==================
def get_lr(step, total_steps):
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


# ================== AdamW با تفکیک weight decay ==================
def configure_optimizer(model, weight_decay, learning_rate):
    """
    بایاس‌ها و پارامترهای LayerNorm (یک‌بعدی هستن) رو از weight decay مستثنی می‌کنیم.
    این یک تکنیک استاندارد در آموزش Transformerهای مدرنه (مثلا در GPT-2 هم استفاده
    می‌شه) چون decay کردن این پارامترهای کوچیک معمولا فایده‌ای نداره و گاهی به
    پایداری آموزش آسیب می‌زنه. ریسک این تغییر پایینه چون فقط پیکربندی optimizer رو
    عوض می‌کنه، نه ریاضیات اصلی مدل رو.
    """
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


# ================== ذخیره/بارگذاری checkpoint آموزش (برای ادامه بعد از قطع برق) ==================
def save_checkpoint(model, optimizer, epoch, step, char_to_ix, ix_to_char):
    """
    برخلاف save_model (که فقط برای مدل نهایی/چت است)، این‌جا وضعیت optimizer
    (momentum و variance در AdamW) و شماره‌ی epoch/step هم ذخیره می‌شه. بدون این‌ها،
    ادامه‌ی آموزش عملاً از نو شروع می‌شد چون optimizer حافظه‌ی خودش رو از دست می‌داد
    و LR schedule هم درست محاسبه نمی‌شد.
    """
    underlying = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({
        "model_state": underlying.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "vocab_size": underlying.vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "seq_length": seq_length,
        "dropout_rate": dropout_rate,
        "char_to_ix": char_to_ix,
        "ix_to_char": {str(k): v for k, v in ix_to_char.items()},
    }, checkpoint_path)


def load_checkpoint():
    ckpt = torch.load(checkpoint_path, map_location=device)
    ix_to_char = {int(k): v for k, v in ckpt["ix_to_char"].items()}
    char_to_ix = ckpt["char_to_ix"]

    model = TinyTransformer(
        ckpt["vocab_size"], ckpt["d_model"], ckpt["n_layers"],
        ckpt["n_heads"], ckpt["d_ff"], ckpt["seq_length"], ckpt["dropout_rate"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    optimizer = configure_optimizer(model, weight_decay, learning_rate)
    optimizer.load_state_dict(ckpt["optimizer_state"])

    return model, optimizer, ckpt["epoch"], ckpt["step"], char_to_ix, ix_to_char


# ================== آموزش ==================
def train(model, data, char_to_ix, ix_to_char, optimizer=None, start_epoch=0, start_step=0,
          checkpoint_every_steps=200):
    """
    checkpoint_every_steps: هر چند step یه‌بار checkpoint ذخیره می‌شه. این مهم‌تر از
    ذخیره‌ی فقط در Ctrl+C است، چون قطعی برق واقعی هیچ فرصتی برای اجرای کد نمی‌ده -
    تنها محافظت واقعی، ذخیره‌ی مرتب و مکرره. عدد ۲۰۰ رو می‌تونی کم/زیاد کنی؛ کمتر
    یعنی کمتر آموزش از دست می‌ره ولی سربار I/O بیشتر (که برای این مدل ناچیزه).
    """
    steps_per_epoch = max(1, len(data) // seq_length)  # جلوگیری از تقسیم بر صفر با دیتاست خیلی کوچیک
    total_steps = steps_per_epoch * epochs

    if optimizer is None:
        optimizer = configure_optimizer(model, weight_decay, learning_rate)

    step = start_step
    try:
        for epoch in range(start_epoch, epochs):
            total_loss = 0.0
            for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}"):
                xb, yb = get_batch(data, seq_length, batch_size, device)

                lr = get_lr(step, total_steps)
                for g in optimizer.param_groups:
                    g["lr"] = lr

                optimizer.zero_grad(set_to_none=True)
                _, loss = model(xb, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                total_loss += loss.item()
                step += 1

                if step % checkpoint_every_steps == 0:
                    save_checkpoint(model, optimizer, epoch, step, char_to_ix, ix_to_char)

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / steps_per_epoch:.4f} - LR: {lr:.6f}")
            # در پایان هر epoch هم یه checkpoint کامل ذخیره می‌کنیم (epoch رو با +1 ثبت
            # می‌کنیم تا در اجرای بعدی از epoch درست بعدی شروع بشه، نه تکرار همون epoch)
            save_checkpoint(model, optimizer, epoch + 1, step, char_to_ix, ix_to_char)

    except KeyboardInterrupt:
        # با Ctrl+C هم یه checkpoint نهایی و آخرین مدل قابل‌چت رو ذخیره می‌کنیم
        print("\nآموزش با Ctrl+C متوقف شد - در حال ذخیره...")
        save_checkpoint(model, optimizer, epoch, step, char_to_ix, ix_to_char)
        underlying = model._orig_mod if hasattr(model, "_orig_mod") else model
        save_model(underlying, char_to_ix, ix_to_char)
        print(f"ذخیره شد. برای ادامه‌ی آموزش، دوباره همین اسکریپت رو اجرا کن - خودش checkpoint رو پیدا و ازش ادامه می‌ده.")
        raise SystemExit(0)


# ================== ذخیره / بارگذاری ==================
def save_model(model, char_to_ix, ix_to_char):
    torch.save(model.state_dict(), model_path)
    config = {
        "vocab_size": model.vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "seq_length": seq_length,
        "dropout_rate": dropout_rate,
        "char_to_ix": char_to_ix,
        "ix_to_char": {str(k): v for k, v in ix_to_char.items()},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False)


def load_model():
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    ix_to_char = {int(k): v for k, v in config["ix_to_char"].items()}
    char_to_ix = config["char_to_ix"]
    model = TinyTransformer(
        config["vocab_size"], config["d_model"], config["n_layers"],
        config["n_heads"], config["d_ff"], config["seq_length"], config["dropout_rate"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, char_to_ix, ix_to_char


# ================== چت تعاملی ==================
def chat_loop(model, char_to_ix, ix_to_char):
    print("\n" + "=" * 50)
    print("چت آماده است (خروج: 'خروج' یا 'exit')")
    print("=" * 50)
    while True:
        user_input = input("\nشما: ").strip().lower()
        if user_input in ("exit", "quit", "خروج"):
            print("خداحافظ!")
            break
        clean = "".join(ch for ch in user_input if ch in char_to_ix)
        if not clean:
            print("مدل: (این کاراکترها رو نمی‌شناسم)")
            continue
        idx = torch.tensor([[char_to_ix[ch] for ch in clean]], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=150, temperature=0.8, top_k=top_k)
        out_ids = out[0, len(clean):].tolist()
        generated = "".join(ix_to_char[i] for i in out_ids)
        print(f"مدل: {clean}{generated}")


# ================== اجرا ==================
if __name__ == "__main__":
    text, chars, char_to_ix, ix_to_char = load_data("dataset")
    vocab_size = len(chars)
    print(f"دیتاست لود شد. تعداد کاراکترها: {len(text)} | واژگان: {vocab_size}")

    if os.path.exists(model_path) and os.path.exists(config_path) and not os.path.exists(checkpoint_path):
        # مدل نهایی از قبل کامل آموزش دیده و checkpoint ناقصی هم باقی نمونده - مستقیم چت
        print("مدل ذخیره‌شده پیدا شد، در حال بارگذاری...")
        model, char_to_ix, ix_to_char = load_model()

    elif os.path.exists(checkpoint_path):
        # یه checkpoint ناقص هست (مثلا به‌خاطر قطعی برق یا Ctrl+C وسط آموزش متوقف شده) -
        # دقیقا از همون epoch/step ادامه می‌دیم، نه از اول
        print(f"Checkpoint ناقص پیدا شد، در حال ادامه‌ی آموزش از همون‌جا...")
        model, optimizer, start_epoch, start_step, char_to_ix, ix_to_char = load_checkpoint()
        print(f"ادامه از epoch {start_epoch + 1}, step {start_step}")

        data = torch.tensor([char_to_ix[ch] for ch in text], dtype=torch.long)

        if start_epoch >= epochs:
            print(f"این checkpoint از قبل {epochs} epoch (کل epochsهای تنظیم‌شده) رو تموم کرده.")
            print("اگه می‌خوای بیشتر آموزش بدی، عدد epochs رو در بالای فایل زیاد کن و دوباره اجرا کن.")
        else:
            train(model, data, char_to_ix, ix_to_char, optimizer=optimizer,
                  start_epoch=start_epoch, start_step=start_step)
            os.remove(checkpoint_path)  # آموزش کامل شد، دیگه به checkpoint نیاز نیست
            save_model(model, char_to_ix, ix_to_char)
            print(f"مدل نهایی ذخیره شد در {model_path}")

    else:
        print("آموزش مدل از ابتدا...")
        data = torch.tensor([char_to_ix[ch] for ch in text], dtype=torch.long)
        model = TinyTransformer(vocab_size, d_model, n_layers, n_heads, d_ff, seq_length, dropout_rate).to(device)

        if use_compile:
            try:
                compiled_model = torch.compile(model)
                # torch.compile تنبله - فقط در اولین اجرا واقعا کامپایل می‌کنه، پس یک
                # forward/backward آزمایشی کوچیک اجرا می‌کنیم تا مطمئن بشیم واقعا کار می‌کنه
                dummy_x, dummy_y = get_batch(data, seq_length, min(batch_size, 2), device)
                _, dummy_loss = compiled_model(dummy_x, dummy_y)
                dummy_loss.backward()
                model.zero_grad(set_to_none=True)
                model = compiled_model
                print("torch.compile فعال شد (با موفقیت تست شد)")
            except Exception as e:
                print(f"torch.compile کار نکرد ({type(e).__name__})، ادامه بدون آن")

        train(model, data, char_to_ix, ix_to_char)
        # اگه مدل کامپایل شده، برای ذخیره باید به مدل اصلی زیرینش دسترسی پیدا کنیم
        underlying_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)  # آموزش کامل شد، دیگه به checkpoint نیاز نیست
        save_model(underlying_model, char_to_ix, ix_to_char)
        print(f"مدل ذخیره شد در {model_path}")
        model = underlying_model

    chat_loop(model, char_to_ix, ix_to_char)
