import numpy as np
from tqdm import tqdm
import os
import pickle

# ================== تنظیمات ==================
d_model = 256       # عرض بردار هر کاراکتر (به‌جای hidden_size در RNN)
n_layers = 12          # تعداد لایه‌های Transformer
n_heads = 8           # تعداد سرهای attention (باید d_model را عاد بزنه)
d_ff = 512            # عرض لایه feed-forward داخل هر بلوک
seq_length = 128        # طول context (چند کاراکتر همزمان می‌بینه)
learning_rate = 0.001
epochs = 8
batch_size = 8
model_path = "transformer_model.pkl"

dropout_rate = 0.3        # احتمال حذف موقت نورون‌ها در آموزش (جلوگیری از overfitting)
weight_decay = 0.01        # ضریب AdamW برای جریمه کردن وزن‌های بزرگ
warmup_steps = 100        # تعداد گام‌هایی که learning rate به‌تدریج زیاد می‌شه
top_k = 1                # هنگام تولید متن، فقط از بین k کاراکتر محتمل‌تر انتخاب می‌شه

assert d_model % n_heads == 0, "d_model باید بر n_heads بخش‌پذیر باشه"
d_head = d_model // n_heads

# ================== لود دیتاست ==================
def load_data(file_path="data.txt"):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    chars = sorted(list(set(text)))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    return text, chars, char_to_ix, ix_to_char

# ================== توابع کمکی ==================
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta, (x_norm, mean, var)

def gelu(x):
    # تقریب استاندارد GELU
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def gelu_grad(x):
    tanh_out = np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))
    left = 0.5 * (1 + tanh_out)
    right = 0.5 * x * (1 - tanh_out ** 2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x ** 2)
    return left + right

def dropout_forward(x, rate, training):
    """اگر training=False (یعنی در حالت chat/generate)، دراپ‌اوت غیرفعاله و x بدون تغییر برمی‌گرده."""
    if not training or rate == 0:
        return x, None
    mask = (np.random.rand(*x.shape) > rate) / (1 - rate)  # inverted dropout: مقیاس در همون لحظه اعمال می‌شه
    return x * mask, mask

def dropout_backward(dout, mask):
    if mask is None:
        return dout
    return dout * mask

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    angles = pos * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe

# ================== یک بلوک Transformer (decoder-only, causal) ==================
class TransformerBlock:
    def __init__(self, d_model, n_heads, d_ff):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_ff = d_ff

        scale = 0.02
        self.Wq = np.random.randn(d_model, d_model) * scale
        self.Wk = np.random.randn(d_model, d_model) * scale
        self.Wv = np.random.randn(d_model, d_model) * scale
        self.Wo = np.random.randn(d_model, d_model) * scale

        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)

        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)

    def params(self):
        return [self.Wq, self.Wk, self.Wv, self.Wo, self.W1, self.b1, self.W2, self.b2,
                self.gamma1, self.beta1, self.gamma2, self.beta2]

    def param_names(self):
        return ["Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2", "gamma1", "beta1", "gamma2", "beta2"]

    def forward(self, x, mask, dropout_rate=0.0, training=True):
        """
        x: (batch, seq, d_model)
        mask: (seq, seq) causal mask (0 برای مجاز، -inf برای ممنوع)
        dropout_rate/training: در chat/generate باید training=False باشه تا dropout غیرفعال بمونه
        """
        cache = {}
        B, T, D = x.shape
        cache["x"] = x

        # --- Self-Attention ---
        Q = x @ self.Wq  # (B,T,D)
        K = x @ self.Wk
        V = x @ self.Wv
        cache["Q"], cache["K"], cache["V"] = Q, K, V

        Qh = Q.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)  # (B,H,T,dh)
        Kh = K.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        Vh = V.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(self.d_head)  # (B,H,T,T)
        scores = scores + mask  # causal mask
        attn = softmax(scores, axis=-1)  # (B,H,T,T)
        cache["attn"], cache["Qh"], cache["Kh"], cache["Vh"] = attn, Qh, Kh, Vh

        out_heads = attn @ Vh  # (B,H,T,dh)
        out = out_heads.transpose(0, 2, 1, 3).reshape(B, T, D)  # (B,T,D)
        cache["out_heads"] = out_heads
        attn_out = out @ self.Wo
        cache["attn_concat"] = out

        # dropout روی خروجی attention (قبل از residual)، مطابق معماری استاندارد Transformer
        attn_out, attn_drop_mask = dropout_forward(attn_out, dropout_rate, training)
        cache["attn_drop_mask"] = attn_drop_mask

        # residual + norm 1
        res1 = x + attn_out
        norm1, ln1_cache = layer_norm(res1, self.gamma1, self.beta1)
        cache["res1"], cache["ln1_cache"] = res1, ln1_cache

        # --- Feed Forward ---
        ff_hidden_pre = norm1 @ self.W1 + self.b1
        ff_hidden = gelu(ff_hidden_pre)
        ff_out = ff_hidden @ self.W2 + self.b2
        cache["ff_hidden_pre"], cache["ff_hidden"], cache["norm1"] = ff_hidden_pre, ff_hidden, norm1

        # dropout روی خروجی feed-forward (قبل از residual)
        ff_out, ff_drop_mask = dropout_forward(ff_out, dropout_rate, training)
        cache["ff_drop_mask"] = ff_drop_mask

        # residual + norm 2
        res2 = norm1 + ff_out
        norm2, ln2_cache = layer_norm(res2, self.gamma2, self.beta2)
        cache["res2"], cache["ln2_cache"] = res2, ln2_cache

        return norm2, cache

    def backward(self, dout, cache):
        B, T, D = cache["x"].shape

        # --- backward norm2 ---
        dres2, dgamma2, dbeta2 = self._layer_norm_backward(dout, cache["res2"], self.gamma2, cache["ln2_cache"])
        dnorm1_a = dres2  # از residual
        dff_out = dres2

        # dropout روی خروجی feed-forward (باید معکوس forward اعمال بشه، قبل از استفاده در محاسبات بعدی)
        dff_out = dropout_backward(dff_out, cache["ff_drop_mask"])

        # --- backward feed-forward ---
        dW2 = cache["ff_hidden"].reshape(-1, self.d_ff).T @ dff_out.reshape(-1, D)
        db2 = dff_out.reshape(-1, D).sum(axis=0)
        dff_hidden = dff_out @ self.W2.T
        dff_hidden_pre = dff_hidden * gelu_grad(cache["ff_hidden_pre"])
        dW1 = cache["norm1"].reshape(-1, D).T @ dff_hidden_pre.reshape(-1, self.d_ff)
        db1 = dff_hidden_pre.reshape(-1, self.d_ff).sum(axis=0)
        dnorm1_b = dff_hidden_pre @ self.W1.T

        dnorm1 = dnorm1_a + dnorm1_b

        # --- backward norm1 ---
        dres1, dgamma1, dbeta1 = self._layer_norm_backward(dnorm1, cache["res1"], self.gamma1, cache["ln1_cache"])
        dx_a = dres1
        dattn_out = dres1

        # dropout روی خروجی attention
        dattn_out = dropout_backward(dattn_out, cache["attn_drop_mask"])

        # --- backward attention output projection ---
        dWo = cache["attn_concat"].reshape(-1, D).T @ dattn_out.reshape(-1, D)
        dout_concat = dattn_out @ self.Wo.T
        dout_heads = dout_concat.reshape(B, T, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        # --- backward attn @ V ---
        dattn = dout_heads @ cache["Vh"].transpose(0, 1, 3, 2)
        dVh = cache["attn"].transpose(0, 1, 3, 2) @ dout_heads

        # --- backward softmax ---
        attn = cache["attn"]
        dscores = attn * (dattn - np.sum(dattn * attn, axis=-1, keepdims=True))
        dscores = dscores / np.sqrt(self.d_head)

        dQh = dscores @ cache["Kh"]
        dKh = dscores.transpose(0, 1, 3, 2) @ cache["Qh"]

        dQ = dQh.transpose(0, 2, 1, 3).reshape(B, T, D)
        dK = dKh.transpose(0, 2, 1, 3).reshape(B, T, D)
        dV = dVh.transpose(0, 2, 1, 3).reshape(B, T, D)

        dWq = cache["x"].reshape(-1, D).T @ dQ.reshape(-1, D)
        dWk = cache["x"].reshape(-1, D).T @ dK.reshape(-1, D)
        dWv = cache["x"].reshape(-1, D).T @ dV.reshape(-1, D)

        dx_b = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T

        dx = dx_a + dx_b

        grads = {
            "Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo,
            "W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
            "gamma1": dgamma1, "beta1": dbeta1,
            "gamma2": dgamma2, "beta2": dbeta2,
        }
        return dx, grads

    @staticmethod
    def _layer_norm_backward(dout, x, gamma, ln_cache):
        x_norm, mean, var = ln_cache
        N = x.shape[-1]
        std_inv = 1.0 / np.sqrt(var + 1e-5)

        dgamma = np.sum(dout * x_norm, axis=tuple(range(dout.ndim - 1)))
        dbeta = np.sum(dout, axis=tuple(range(dout.ndim - 1)))

        dx_norm = dout * gamma
        dx = (1.0 / N) * std_inv * (
            N * dx_norm
            - np.sum(dx_norm, axis=-1, keepdims=True)
            - x_norm * np.sum(dx_norm * x_norm, axis=-1, keepdims=True)
        )
        return dx, dgamma, dbeta

# ================== مدل کامل Transformer ==================
class TinyTransformer:
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, seq_length, ix_to_char):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.seq_length = seq_length
        self.ix_to_char = ix_to_char

        scale = 0.02
        self.embed = np.random.randn(vocab_size, d_model) * scale
        self.pos_enc = positional_encoding(seq_length, d_model)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        # Weight Tying: به‌جای ماتریس خروجی جدا، از transpose خود embedding استفاده می‌کنیم
        # (مثل GPT-2)؛ این هم پارامترها رو کم می‌کنه هم چون یک ماتریس مشترک هست، بهتر آموزش می‌بینه.
        self.bout = np.zeros(vocab_size)

        # حافظه Adam برای هر پارامتر
        self.m = {}
        self.v = {}
        self.t = 0
        self._init_adam()

    def _all_named_params(self):
        # توجه: Wout دیگه پارامتر جدا نیست (weight tying با embed)، پس اینجا لیست نمی‌شه
        params = {"embed": self.embed, "bout": self.bout}
        for li, block in enumerate(self.blocks):
            for name, p in zip(block.param_names(), block.params()):
                params[f"block{li}_{name}"] = p
        return params

    def _init_adam(self):
        for name, p in self._all_named_params().items():
            self.m[name] = np.zeros_like(p)
            self.v[name] = np.zeros_like(p)

    def causal_mask(self, T):
        mask = np.triu(np.ones((T, T)), k=1) * -1e9
        return mask[None, None, :, :]  # (1,1,T,T) قابل broadcast با (B,H,T,T)

    def forward(self, idx, training=True):
        """idx: (batch, seq) از اندیس کاراکترها
        training=False باید هنگام chat/generate استفاده بشه تا dropout غیرفعال باشه.
        """
        B, T = idx.shape
        x = self.embed[idx] + self.pos_enc[:T][None, :, :]
        mask = self.causal_mask(T)

        caches = []
        for block in self.blocks:
            x, cache = block.forward(x, mask, dropout_rate=dropout_rate, training=training)
            caches.append(cache)

        # Weight tying: از transpose خود embedding به‌عنوان ماتریس خروجی استفاده می‌کنیم
        logits = x @ self.embed.T + self.bout  # (B,T,vocab)
        return logits, (idx, x, caches)

    def loss_and_grads(self, idx, targets):
        B, T = idx.shape
        logits, (idx_, last_x, caches) = self.forward(idx, training=True)
        probs = softmax(logits, axis=-1)

        # cross-entropy loss
        target_probs = np.take_along_axis(probs, targets[:, :, None], axis=2).squeeze(-1)
        loss = -np.mean(np.log(target_probs + 1e-9))

        # گرادیان softmax + cross-entropy
        dlogits = probs.copy()
        one_hot_idx = targets
        for b in range(B):
            for t in range(T):
                dlogits[b, t, one_hot_idx[b, t]] -= 1
        dlogits /= (B * T)

        # چون Wout حالا با embed مشترکه (weight tying)، دو مسیر گرادیان به embed می‌رسه:
        # ۱) از همین‌جا: dlogits نسبت به embed.T که در محاسبه logits استفاده شده
        # ۲) پایین‌تر: از مسیر embedding lookup در ورودی شبکه
        # هر دو باید جمع بشن، وگرنه گرادیان embed ناقصه.
        dembed_from_output = dlogits.reshape(-1, self.vocab_size).T @ last_x.reshape(-1, self.d_model)
        dbout = dlogits.reshape(-1, self.vocab_size).sum(axis=0)
        dx = dlogits @ self.embed  # چون logits = x @ embed.T، مشتق نسبت به x برابر dlogits @ embed است

        grads = {"bout": dbout}

        for li in reversed(range(self.n_layers)):
            dx, block_grads = self.blocks[li].backward(dx, caches[li])
            for name, g in block_grads.items():
                grads[f"block{li}_{name}"] = g

        # گرادیان embedding از مسیر lookup (ورودی شبکه)
        dembed = np.zeros_like(self.embed)
        np.add.at(dembed, idx, dx)

        # جمع دو مسیر گرادیان embed (خروجی + ورودی)
        dembed += dembed_from_output
        grads["embed"] = dembed

        return loss, grads

    def get_lr(self, base_lr, warmup_steps, total_steps):
        """Warmup خطی سپس cosine decay - الگوی استاندارد آموزش Transformer."""
        if self.t < warmup_steps:
            return base_lr * (self.t + 1) / warmup_steps
        progress = (self.t - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(progress, 1.0)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

    def adam_step(self, grads, lr=None, beta1=0.9, beta2=0.999, eps=1e-8,
                  weight_decay=weight_decay, warmup_steps=warmup_steps, total_steps=None):
        """
        نسخه AdamW: بر خلاف Adam معمولی، weight decay مستقیماً روی وزن اعمال می‌شه
        (نه این‌که به گرادیان اضافه بشه)، که باعث می‌شه جریمه شدن وزن‌های بزرگ مستقل
        از مقدار gradient باشه - این تفاوت اصلی AdamW با Adam است.
        """
        self.t += 1
        if lr is None:
            lr = self.get_lr(learning_rate, warmup_steps, total_steps or (warmup_steps * 10))

        params = self._all_named_params()
        for name, p in params.items():
            g = grads[name]
            np.clip(g, -5, 5, out=g)

            self.m[name] = beta1 * self.m[name] + (1 - beta1) * g
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * (g * g)
            m_hat = self.m[name] / (1 - beta1 ** self.t)
            v_hat = self.v[name] / (1 - beta2 ** self.t)

            # AdamW: weight decay مستقیم روی پارامتر، جدا از گرادیان تطبیقی Adam
            p -= lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * p)

    def generate(self, seed_ixs, n, temperature=0.8, top_k=top_k):
        """
        training=False حیاتیه: بدون این، dropout در حین تولید متن هم فعال می‌مونه
        و باعث خروجی تصادفی/بی‌کیفیت‌تر می‌شه چون بخشی از نورون‌ها به‌طور تصادفی خاموش می‌شن.

        top_k: فقط از بین k کاراکتر با بیشترین احتمال یکی انتخاب می‌شه (نه از کل واژگان)،
        که جلوی انتخاب کاراکترهای بسیار بعیدِ دم توزیع رو می‌گیره و متن رو منسجم‌تر می‌کنه.
        """
        generated = list(seed_ixs)
        for _ in range(n):
            context = generated[-self.seq_length:]
            idx = np.array([context])
            logits, _ = self.forward(idx, training=False)
            last_logits = logits[0, -1] / temperature

            if top_k is not None and top_k < self.vocab_size:
                # پیدا کردن k مقدار برتر و صفر کردن (نه انتخاب) بقیه با -inf
                top_k_idx = np.argpartition(last_logits, -top_k)[-top_k:]
                filtered_logits = np.full_like(last_logits, -1e9)
                filtered_logits[top_k_idx] = last_logits[top_k_idx]
                last_logits = filtered_logits

            p = softmax(last_logits)
            ix = np.random.choice(range(self.vocab_size), p=p)
            generated.append(ix)
        return [self.ix_to_char[i] for i in generated[len(seed_ixs):]]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

# ================== batch سازی ==================
def get_batches(text, char_to_ix, seq_length, batch_size):
    data = np.array([char_to_ix[ch] for ch in text])
    n_chunks = (len(data) - 1) // seq_length
    usable = n_chunks * seq_length
    x_all = data[:usable].reshape(n_chunks, seq_length)
    y_all = data[1:usable + 1].reshape(n_chunks, seq_length)

    n_batches = n_chunks // batch_size
    for b in range(n_batches):
        xb = x_all[b * batch_size:(b + 1) * batch_size]
        yb = y_all[b * batch_size:(b + 1) * batch_size]
        yield xb, yb

# ================== آموزش ==================
def train(model, text, char_to_ix, epochs):
    batches_per_epoch = len(list(get_batches(text, char_to_ix, seq_length, batch_size)))
    total_steps = batches_per_epoch * epochs  # برای محاسبه cosine decay در طول کل آموزش لازمه

    for epoch in range(epochs):
        batches = list(get_batches(text, char_to_ix, seq_length, batch_size))
        total_loss = 0
        for xb, yb in tqdm(batches, desc=f"Epoch {epoch + 1}/{epochs}"):
            loss, grads = model.loss_and_grads(xb, yb)
            model.adam_step(grads, total_steps=total_steps)
            total_loss += loss
        current_lr = model.get_lr(learning_rate, warmup_steps, total_steps)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss / max(len(batches),1):.4f} - LR: {current_lr:.6f}")

# ================== چت تعاملی ==================
def chat_loop(model, char_to_ix):
    print("\n" + "=" * 50)
    print("چت آماده است (خروج: 'خروج' یا 'exit')")
    print("=" * 50)
    while True:
        user_input = input("\nشما: ").strip().lower()
        if user_input in ("exit", "quit", "خروج"):
            print("خداحافظ!")
            break
        clean = ''.join(ch for ch in user_input if ch in char_to_ix)
        if not clean:
            print("مدل: (این کاراکترها رو نمی‌شناسم)")
            continue
        seed_ixs = [char_to_ix[ch] for ch in clean][-model.seq_length:]
        out_chars = model.generate(seed_ixs, 150)
        print(f"مدل: {clean}{''.join(out_chars)}")

# ================== اجرا ==================
if __name__ == "__main__":
    text, chars, char_to_ix, ix_to_char = load_data("data.txt")
    vocab_size = len(chars)
    print(f"دیتاست لود شد. تعداد کاراکترها: {len(text)} | واژگان: {vocab_size}")

    if os.path.exists(model_path):
        print("مدل ذخیره‌شده پیدا شد، در حال بارگذاری...")
        model = TinyTransformer.load(model_path)
    else:
        print("آموزش مدل از ابتدا...")
        model = TinyTransformer(vocab_size, d_model, n_layers, n_heads, d_ff, seq_length, ix_to_char)
        train(model, text, char_to_ix, epochs)
        model.save(model_path)
        print(f"مدل ذخیره شد در {model_path}")

    chat_loop(model, char_to_ix)
