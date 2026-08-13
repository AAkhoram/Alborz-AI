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

# حداکثر حجمی (به مگابایت) که متن خام دیتاست اجازه داره همزمان توی RAM باشه.
# دیتاست هیچ‌وقت یکجا لود نمی‌شه: فایل‌های dataset/ یکی‌یکی لود می‌شن (تا سقف
# همین مقدار پر بشه)، آموزش می‌بینن، بعد offload می‌شن و نوبت فایل بعدی می‌رسه.
# اگه با کمبود رم مواجه شدی، این عدد رو کم کن.
max_ram_mb = 6000

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


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    اگه مدل با torch.compile کامپایل شده باشه، nn.Module واقعی زیرش (._orig_mod)
    رو برمی‌گردونه؛ در غیر این صورت خود مدل رو بدون تغییر برمی‌گردونه.
    این تابع جایگزین تکرار الگوی 'model._orig_mod if hasattr(...) else model' شده.
    """
    return model._orig_mod if hasattr(model, "_orig_mod") else model


# ================== پیدا کردن فایل‌های دیتاست ==================
def find_txt_files(path: str = "dataset") -> List[str]:
    """
    اگه path یک پوشه باشه، مسیر تمام فایل‌های .txt داخلش (و زیرپوشه‌ها) رو
    برمی‌گردونه؛ خروجی همیشه sorted هست تا ترتیب فایل‌ها بین اجراهای مختلف
    (و بین ذخیره/بارگذاری checkpoint) ثابت بمونه - چون pool پایین‌تر بر اساس
    همین ترتیب یاد می‌گیره کدوم فایل‌ها رو قبلاً دیده و کدوم رو نه.
    اگه path یک فایل تکی باشه (مثلاً data.txt)، یک لیست تک‌عضوی برمی‌گردونه.
    این تابع فقط مسیرها رو برمی‌گردونه، هیچ محتوایی نمی‌خونه.
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


# ================== پاس اول سبک: فقط ساخت vocab ==================
# اندازه‌ی هر تکه (کاراکتر) هنگام خوندن streaming یک فایل برای اسکن vocab.
# این عدد فقط سرعت/دقت پیشرفت اسکن رو تعیین می‌کنه، تأثیری روی حجم حافظه‌ی
# نهایی نداره چون هر تکه بعد از پردازش دور ریخته می‌شه.
VOCAB_SCAN_CHUNK_CHARS = 1_000_000


def scan_vocab(txt_files: List[str]) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    قبل از شروع pooling، باید char_to_ix/ix_to_char مشخص و ثابت باشن (چون
    embedding مدل با اندازه‌ی vocab_size ساخته می‌شه و این اندازه بعداً
    نمی‌تونه عوض بشه). اما نمی‌شه صبر کرد کل دیتاست لود بشه تا vocab بسازیم -
    دقیقاً همون چیزیه که pooling می‌خواد ازش جلوگیری کنه.

    راه‌حل: یک پاس سبک و streaming روی همه‌ی فایل‌ها می‌زنیم که هر فایل رو
    تکه‌تکه (به‌جای یکجا با f.read()) می‌خونه و فقط کاراکترهای یکتا رو توی یک
    set جمع می‌کنه؛ خودِ متن هیچ‌وقت نگه داشته نمی‌شه. حافظه‌ی مصرفی این پاس
    فقط به تعداد کاراکترهای یکتای دیتاست بستگی داره (چند صد کاراکتر معمولاً)،
    نه به حجم کل دیتاست.
    """
    all_chars = set()
    for fp in txt_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                while True:
                    chunk = f.read(VOCAB_SCAN_CHUNK_CHARS)
                    if not chunk:
                        break
                    all_chars.update(chunk.lower())
        except UnicodeDecodeError:
            print(f"هشدار: فایل '{fp}' با UTF-8 خونده نشد و در اسکن vocab نادیده گرفته شد")

    chars = sorted(all_chars)
    if not chars:
        raise ValueError("هیچ کاراکتری در دیتاست پیدا نشد - واژگانی برای ساخت مدل وجود نداره")
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    return chars, char_to_ix, ix_to_char


def encode_text(text: str, char_to_ix: Dict[str, int]) -> torch.Tensor:
    """یک متن رو به تنسور کاراکترهای اندیس‌شده تبدیل می‌کنه."""
    return torch.tensor([char_to_ix[ch] for ch in text], dtype=torch.long)


# ================== Pool: لود/آفلود تدریجی فایل‌های دیتاست ==================
# تعداد کاراکتری که برای تخمین نسبت بایت-به-کاراکتر از ابتدای هر فایل نمونه
# گرفته می‌شه (نه کل فایل، چون این فقط یک تخمینه و لازم نیست کل دیتاست خونده
# بشه). این تابع عمداً deterministic است (همیشه همون بایت‌های ابتدای همون
# فایل‌ها رو می‌خونه، نه نمونه‌ی رندوم) چون steps_per_epoch_total باید بین
# اجرای اول و resume از checkpoint دقیقاً یکسان بمونه (وگرنه LR schedule
# به‌هم می‌ریزه - همون گاردی که توی load_checkpoint چک می‌شه).
_BYTES_PER_CHAR_SAMPLE_SIZE = 65_536  # ۶۴ کیلوبایت نمونه از ابتدای هر فایل


def _estimate_bytes_per_char(txt_files: List[str]) -> float:
    """
    به‌جای فرض ثابت (که برای فارسی/عربی ~۲ ولی برای انگلیسی/کد ~۱ است و
    دیتاست‌های ترکیبی رو غلط تخمین می‌زد - همون مشکلی که با دیتاست ویکی‌پدیای
    ترکیبی فارسی/انگلیسی/اعداد پیش میاد)، این تابع از ابتدای چند فایل واقعیِ
    دیتاست، یک نمونه‌ی کوچیک (حداکثر _BYTES_PER_CHAR_SAMPLE_SIZE بایت از هر
    فایل) می‌خونه و نسبت واقعیِ (بایت / کاراکتر) رو حساب می‌کنه.

    این تابع باید deterministic بمونه (همیشه دقیقاً همون فایل‌ها و همون
    بخش ابتدایی‌شون رو می‌خونه)، چون steps_per_epoch_total که از این تابع
    میاد باید بین اجرای اول و resume از checkpoint فرق نکنه.
    """
    total_sample_bytes = 0
    total_sample_chars = 0
    # از حداکثر ۵ فایل اول نمونه می‌گیریم - کافیه برای یک تخمین معقول از
    # نسبت بایت/کاراکتر، بدون این‌که خوندن نمونه‌ها خودش کند بشه.
    for fp in txt_files[:5]:
        try:
            with open(fp, "rb") as f:
                raw = f.read(_BYTES_PER_CHAR_SAMPLE_SIZE)
        except OSError:
            continue
        if not raw:
            continue
        try:
            # ممکنه آخرین کاراکتر چندبایتی وسط بریده شده باشه (چون raw رو
            # با تعداد بایت ثابت خوندیم، نه تعداد کاراکتر)؛ errors="ignore"
            # این بایت‌های ناقصِ انتهایی رو نادیده می‌گیره - فقط روی دقتِ
            # همین تخمین (نه روی خودِ encode_text واقعی) اثر کوچیکی داره.
            text = raw.decode("utf-8", errors="ignore").lower()
        except Exception:
            continue
        if text:
            total_sample_bytes += len(raw)
            total_sample_chars += len(text)

    if total_sample_chars == 0:
        # اگه به هر دلیلی هیچ نمونه‌ای قابل خوندن نبود (مثلاً همه‌ی فایل‌ها
        # واقعاً خالی/غیرUTF8 بودن)، به همون فرض قبلی محافظه‌کارانه برمی‌گردیم
        # تا حداقل یک عدد معقول (نه صفر یا خطا) برگردونده بشه.
        return 2.0
    return total_sample_bytes / total_sample_chars


def compute_steps_per_epoch_total(txt_files: List[str]) -> int:
    """
    تخمین تعداد کل step های لازم برای یک "epoch" (یعنی یک چرخه‌ی کامل روی
    همه‌ی فایل‌های dataset/)، بدون اینکه کل هیچ فایلی واقعاً خونده بشه - فقط
    از روی حجم بایتی فایل‌ها (os.path.getsize) به‌همراه یک نمونه‌ی کوچیک برای
    تخمین نسبت بایت/کاراکتر واقعیِ همین دیتاست (_estimate_bytes_per_char).
    این عدد هم برای ساخت LR schedule لازمه (total_steps = steps_per_epoch_total
    * epochs) و هم برای تقسیم سهم step بین فایل‌های داخل DatasetPool.

    این تابع باید قطعی و تکرارپذیر باشه (همون ورودی txt_files همیشه همون
    خروجی رو بده) چون هم موقع شروع از صفر و هم موقع resume از checkpoint صدا
    زده می‌شه و اگه بین این دو فرق کنه، LR schedule به‌هم می‌ریزه (به همین
    خاطر توی load_checkpoint با مقدار ذخیره‌شده مقایسه و چک می‌شه).
    """
    total_bytes = sum(os.path.getsize(fp) for fp in txt_files)
    bytes_per_char = _estimate_bytes_per_char(txt_files)
    est_total_chars = total_bytes / bytes_per_char
    return max(1, int(est_total_chars // (seq_length * batch_size)))


class DatasetPool:
    """
    به‌جای اینکه همه‌ی فایل‌های dataset/ یکجا توی RAM باشن، این کلاس فایل‌ها
    رو به ترتیب (sorted، ثابت بین اجراها) یکی‌یکی لود می‌کنه تا مجموع حجم متن
    لودشده به سقف max_ram_mb برسه؛ به هر فایلِ داخل pool، متناسب با حجمش،
    یک سهم از step های آموزش اختصاص می‌ده (فایل ۲ برابر بزرگ‌تر، تقریباً ۲
    برابر step بیشتر می‌گیره - این‌طوری هر بایت از کل دیتاست تقریباً یک‌بار
    با احتمال یکسان دیده می‌شه، نه اینکه فایل‌های کوچیک و بزرگ سهم برابر
    بگیرن).

    وقتی سهم step های یک فایل تموم بشه، اون فایل offload می‌شه (از RAM خارج،
    شیء تنسورش دور ریخته می‌شه) و اگر فایل لود‌نشده‌ای باقی مونده باشه و جا
    توی سقف رم باشه، فایل بعدی لود می‌شه. وقتی تمام فایل‌های دیتاست یک‌بار
    (این چرخه) دیده بشن، یک epoch کامل تموم شده و pool از اول (فایل اول)
    دوباره شروع می‌کنه.

    وضعیت این کلاس (کدوم فایل‌ها دیده شدن، چند step از فایل جاری باقی مونده)
    باید داخل checkpoint ذخیره بشه تا resume درست کار کنه؛ به همین دلیل
    state_dict()/load_state_dict() داره، دقیقاً به سبک optimizer های PyTorch.
    """

    def __init__(
        self,
        txt_files: List[str],
        char_to_ix: Dict[str, int],
        seq_length: int,
        max_ram_mb: int,
        steps_per_epoch_total: int,
    ):
        self.txt_files = txt_files
        self.char_to_ix = char_to_ix
        self.seq_length = seq_length
        self.max_ram_bytes = max_ram_mb * 1_000_000
        # کل دیتاست چند step باید ببینه تا یک epoch کامل بشه؛ این عدد بین
        # فایل‌های pool، متناسب با حجمشون، تقسیم می‌شه (پایین‌تر توی
        # _file_sizes_bytes/_assign_steps_for_file).
        self.steps_per_epoch_total = steps_per_epoch_total

        # حجم هر فایل (بایت، از os.path.getsize - سریع، بدون بازکردن فایل)،
        # برای محاسبه‌ی سهم step و همچنین تصمیم "این فایل توی سقف رم جا می‌شه؟"
        self._file_sizes_bytes = [os.path.getsize(fp) for fp in txt_files]
        total_bytes = sum(self._file_sizes_bytes) or 1

        # سهم step هر فایل، متناسب با حجم بایتیش نسبت به کل دیتاست. حداقل ۱
        # step به هر فایل می‌دیم تا فایل‌های خیلی کوچیک هم حداقل یک‌بار دیده
        # بشن (وگرنه گرد کردن به سمت صفر می‌تونست سهمشون رو کلاً حذف کنه).
        self._steps_for_file = [
            max(1, round(steps_per_epoch_total * b / total_bytes))
            for b in self._file_sizes_bytes
        ]

        # ترتیب پیمایش فایل‌ها در یک epoch: همون ترتیب sorted ثابت (بدون
        # shuffle)، چون shuffle باعث می‌شد resume باید ترتیب رندوم رو هم
        # ذخیره کنه؛ ترتیب ثابت resume رو ساده نگه می‌داره و چون هر فایل به
        # هر حال طی هر epoch دیده می‌شه، ترتیب دیدنش اهمیت کمی داره.
        self._file_order = list(range(len(txt_files)))

        # -------- وضعیت pool (چیزی که resume باید بازیابی کنه) --------
        self.epoch = 0                  # چندمین دور کامل روی کل دیتاست
        self._order_pos = 0             # موقعیت بعدی در self._file_order که هنوز لود نشده
        self._loaded: Dict[int, torch.Tensor] = {}   # {file_index: encoded_tensor}
        self._loaded_bytes = 0          # مجموع حجم بایتی چیزی که الان توی RAM هست
        self._steps_left: Dict[int, int] = {}         # {file_index: چند step از سهمش مونده}

        # -------- وضعیت موقت این epoch (در checkpoint ذخیره نمی‌شه) --------
        # فایل‌هایی که توی همین epoch رد شدن (خالی/کوتاه/غیرUTF8). این set فقط
        # در حافظه‌ست و عمداً توی state_dict() ذخیره نمی‌شه: epoch بعدی همه‌ی
        # فایل‌ها (حتی رد‌شده‌های epoch قبل) دوباره امتحان می‌شن، چون ممکنه
        # کاربر بین این‌بین فایل رو دستی درست کرده باشه.
        self._skipped_this_epoch: set = set()

        self._fill_pool()

    # -------- لود/آفلود --------
    def _fill_pool(self) -> None:
        """
        تا وقتی جا هست (self._loaded_bytes + حجم فایل بعدی <= max_ram_bytes)
        و فایل لود‌نشده‌ای باقی مونده، فایل بعدی رو از دیسک می‌خونه و به pool
        اضافه می‌کنه. اگه حتی یک فایل هم بزرگ‌تر از کل سقف رم باشه، همون یک
        فایل به‌تنهایی لود می‌شه (استثنا، تا برنامه به‌کلی متوقف نشه) و یک
        هشدار چاپ می‌شه - چون در این حالت واقعاً نمی‌شه به سقف تعیین‌شده
        پایبند موند.

        اگه _load_file شکست بخوره (فایل خالی/کوتاه/غیرUTF8)، fi به
        self._skipped_this_epoch اضافه می‌شه (برای همین epoch رد می‌شه، ولی
        _order_pos بازم پیش می‌ره تا فایل بعدی امتحان بشه - رد شدنِ یک فایل
        نباید کل pooling رو متوقف کنه).
        """
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
                # جا نیست و pool هم خالی نیست (یعنی این فایل باید صبر کنه تا
                # فایل‌های دیگه offload بشن)
                break

            if not fits and pool_empty:
                print(
                    f"هشدار: فایل '{self.txt_files[fi]}' به‌تنهایی "
                    f"({fbytes / 1_000_000:.1f} مگابایت) از سقف max_ram_mb="
                    f"{self.max_ram_bytes / 1_000_000:.0f} بزرگ‌تره. برای جلوگیری "
                    f"از توقف کامل برنامه، همین یک فایل استثنائاً لود می‌شه. "
                    f"اگه این باعث پر شدن کامل RAM سیستم بشه، برنامه ممکنه با "
                    f"خطای OOM (Out Of Memory) متوقف بشه - در این صورت این فایل "
                    f"رو از dataset/ خارج کن یا با ابزار جداگانه به چند فایل "
                    f"کوچیک‌تر تقسیمش کن."
                )

            success = self._load_file(fi)
            if not success:
                self._skipped_this_epoch.add(fi)
            self._order_pos += 1


    def _load_file(self, fi: int) -> bool:
        """
        سعی می‌کنه فایل fi رو لود کنه. خروجی True یعنی موفق (self._loaded[fi]
        و self._steps_left[fi] ست شدن)، False یعنی رد شد (خالی/کوتاه/غیرUTF8).

        نکته‌ی مهم: این تابع دیگه خودش تصمیم نمی‌گیره که رد شدن «دائمی» باشه
        یا نه (قبلاً با self._steps_for_file[fi]=0 این کار رو می‌کرد) - فقط
        شکست رو گزارش می‌ده. تصمیم اینکه یه فایل رد‌شده توی epoch بعدی دوباره
        امتحان بشه یا نه، به‌عهده‌ی فراخوان (پایین‌تر: self._skipped_this_epoch)
        است.
        """
        fp = self.txt_files[fi]
        try:
            with open(fp, "r", encoding="utf-8") as f:
                content = f.read().lower()
        except UnicodeDecodeError:
            print(f"هشدار: فایل '{fp}' با UTF-8 خونده نشد و در این epoch نادیده گرفته شد")
            return False
        if not content:
            print(f"هشدار: فایل '{fp}' خالیه و در این epoch نادیده گرفته شد")
            return False
        if len(content) < self.seq_length + 1:
            print(
                f"هشدار: فایل '{fp}' فقط {len(content)} کاراکتر داره، کمتر از "
                f"seq_length+1={self.seq_length + 1}؛ در این epoch نادیده گرفته شد."
            )
            return False

        try:
            tensor = encode_text(content, self.char_to_ix)
        except KeyError as e:
            raise ValueError(
                f"فایل '{fp}' کاراکتری داره که توی واژگان (char_to_ix) فعلی نیست: {e}. "
                f"این معمولاً یعنی فایل‌های dataset/ بعد از ساخت این مدل/checkpoint "
                f"تغییر کرده‌ن یا کاراکتر جدیدی بهشون اضافه شده. برای رفع این مشکل، "
                f"یا مدل/checkpoint رو حذف کن و از صفر با واژگان جدید آموزش بده، یا "
                f"فایل '{fp}' رو موقتاً از dataset/ خارج کن."
            )
        self._loaded[fi] = tensor
        self._loaded_bytes += self._file_sizes_bytes[fi]
        if fi not in self._steps_left:
            self._steps_left[fi] = self._steps_for_file[fi]
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
        print(f"[pool] آفلود شد: '{fp}' (سهم step آموزشیش تموم شد)")

    # -------- استفاده در حلقه‌ی آموزش --------
    def current_files(self) -> List[int]:
        """اندیس فایل‌هایی که الان توی RAM هستن (برای get_batch)."""
        return list(self._loaded.keys())

    def get_tensor(self, fi: int) -> torch.Tensor:
        return self._loaded[fi]

    def notify_step_consumed(self, file_ix_used: List[int]) -> bool:
        """
        بعد از هر batch صدا زده می‌شه؛ file_ix_used لیست اندیس فایل‌هاییه که
        توی همین batch حداقل یک نمونه ازشون گرفته شده. سهم step هر کدوم رو
        یکی کم می‌کنه؛ اگه سهم فایلی صفر شد، offload‌ش می‌کنه و جای خالی رو
        با _fill_pool پر می‌کنه.

        خروجی True یعنی epoch همین الان کامل شد (همه‌ی فایل‌ها offload شدن و
        فایل جدیدی برای لود کردن نمونده)، که یعنی باید از اول (self.epoch+=1
        و بازگشت به فایل اول) شروع بشه.

        اگه pool خالی باشه و همه‌ی فایل‌های دیتاست (نه فقط بعضی) رد شده باشن
        (self._skipped_this_epoch == تمام فایل‌ها)، این یعنی هیچ آموزشی توی
        این epoch اتفاق نیفتاده - این یک خطای صریح می‌ده، نه اینکه بی‌صدا
        epoch رو «کامل» حساب کنه (باگی که قبلاً اینجا وجود داشت).
        """
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
                    f"قابل استفاده نبودن (همه خالی/کوتاه‌تر از seq_length+1="
                    f"{self.seq_length + 1}/غیرUTF8 بودن). آموزش امکان‌پذیر نیست. "
                    f"فایل‌های dataset/ رو بررسی کن."
                )
            self._start_new_epoch()
            return True
        return False

    def _start_new_epoch(self) -> None:
        self.epoch += 1
        self._order_pos = 0
        self._steps_left = {}
        # مهم: _skipped_this_epoch هم باید ریست بشه، وگرنه فایل‌هایی که این
        # epoch رد شدن برای همیشه رد‌شده می‌مونن و هیچ‌وقت دوباره امتحان
        # نمی‌شن - که دقیقاً برخلاف طراحیِ «فقط این epoch رد شه» است.
        self._skipped_this_epoch = set()
        print(f"[pool] === epoch {self.epoch} کامل شد؛ چرخه‌ی جدید از فایل اول شروع می‌شه ===")
        self._fill_pool()

    # -------- ذخیره/بارگذاری برای checkpoint --------
    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "order_pos": self._order_pos,
            "steps_left": dict(self._steps_left),
            # FIX #2: persist skipped files so resume doesn't re-try them mid-epoch
            "skipped_this_epoch": list(self._skipped_this_epoch),
        }

    def load_state_dict(self, state: dict) -> None:
        """
        توجه: این تابع فایل‌های در حال بارگذاری رو دوباره از دیسک می‌خونه
        (offset دقیق کجای فایل بودیم ذخیره نمی‌شه، فقط اینکه چند step از سهم
        فایل باقی مونده) - چون batchها هر بار از موقعیت رندوم داخل فایل
        گرفته می‌شن، نه به‌ترتیب، پس "موقعیت دقیق" اصلاً مفهومی نداره که نیاز
        به ذخیره داشته باشه؛ فقط "چند step دیگه از این فایل مونده" مهمه.

        نکته‌ی مهم: بین ذخیره‌ی checkpoint و resume ممکنه یه فایل از دیسک حذف
        شده باشه یا محتواش خراب/کوتاه شده باشه (مثلاً کاربر دستی چیزی رو عوض
        کرده). اگه _load_file برای یه fi که توی state["steps_left"] بوده
        شکست بخوره، اون fi باید از self._steps_left هم حذف بشه - وگرنه بعداً
        notify_step_consumed سعی می‌کنه offload‌ش کنه (self._offload_file)
        در حالی که هیچ‌وقت واقعاً توی self._loaded نبوده، که باعث KeyError
        می‌شد (باگی که قبلاً اینجا وجود داشت).
        """
        self.epoch = state["epoch"]
        self._order_pos = state["order_pos"]
        self._steps_left = {int(k): v for k, v in state["steps_left"].items()}
        self._loaded = {}
        self._loaded_bytes = 0
        # FIX #2: restore skipped files so we don't re-try them or skip valid ones after them
        self._skipped_this_epoch = set(state.get("skipped_this_epoch", []))
        for fi in list(self._steps_left.keys()):
            success = self._load_file(fi)
            if not success:
                print(
                    f"هشدار: فایل با اندیس {fi} در وضعیت ذخیره‌شده‌ی checkpoint "
                    f"فعال بود ولی الان قابل لود نیست (احتمالاً بین ذخیره‌ی "
                    f"checkpoint و الان تغییر کرده یا حذف شده). سهم step "
                    f"باقی‌مونده‌ش کنار گذاشته می‌شه و توی epoch بعدی دوباره "
                    f"امتحان می‌شه."
                )
                del self._steps_left[fi]
                self._skipped_this_epoch.add(fi)

        # اگه بعضی (یا حتی همه‌ی) فایل‌های فعالِ checkpoint شکست خورده باشن،
        # ممکنه الان جای خالی توی pool باشه (یا حتی کاملاً خالی باشه). این
        # فراخوانی سعی می‌کنه فایل‌های بعدی (که _order_pos هنوز بهشون نرسیده)
        # رو لود کنه، به‌جای اینکه pool برای همیشه خالی بمونه.
        self._fill_pool()


def get_batch(
    pool: DatasetPool, seq_length: int, batch_size: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    نسخه‌ی pool-aware همون get_batch قبلی: به‌جای گرفتن نمونه از بین همه‌ی
    فایل‌های دیتاست، فقط از بین فایل‌هایی که الان توی pool.current_files()
    (یعنی همین الان توی RAM) هستن نمونه می‌گیره. منطق وزن‌دهی بر اساس طول
    فایل و تضمین «حداقل یکی از هر فایل در batch»، دقیقاً همون منطق نسخه‌ی
    قبلیه؛ فقط دامنه‌ش به‌جای «همه‌ی فایل‌های دیتاست» شده «فایل‌های داخل pool».

    خروجی سوم (file_ix_used) لیست اندیس فایل‌هاییه که توی این batch حداقل
    یک‌بار انتخاب شدن - این برای pool.notify_step_consumed لازمه تا بدونه
    سهم کدوم فایل‌ها رو کم کنه.
    """
    usable = [fi for fi in pool.current_files() if len(pool.get_tensor(fi)) >= seq_length + 1]
    if not usable:
        raise ValueError(
            f"هیچ فایلی در pool فعلی به‌اندازه‌ی کافی (حداقل {seq_length + 1} کاراکتر) "
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

    xs, ys, file_ix_used = [], [], []
    for pi in pick_ix:
        fi = usable[pi]
        d = pool.get_tensor(fi)
        max_start = len(d) - seq_length
        i = torch.randint(max_start, (1,)).item()
        xs.append(d[i:i + seq_length])
        ys.append(d[i + 1:i + seq_length + 1])
        file_ix_used.append(fi)

    x = torch.stack(xs)
    y = torch.stack(ys)
    return x.to(device), y.to(device), file_ix_used


# ================== Self-Attention (با Flash Attention روی سخت‌افزار پشتیبانی‌شده) ==================
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


# ================== نرخ یادگیری (warmup + cosine decay) ==================
def get_lr(step: int, total_steps: int) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return learning_rate * 0.5 * (1 + math.cos(math.pi * progress))


# ================== AdamW با تفکیک weight decay ==================
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


# ================== ذخیره/بارگذاری checkpoint آموزش ==================
def _atomic_torch_save(obj, path: str) -> None:
    """ذخیره‌ی اتومیک: اول توی فایل tmp می‌نویسیم، بعد با replace جابه‌جا می‌کنیم."""
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
    char_to_ix: Dict[str, int],
    ix_to_char: Dict[int, str],
) -> None:
    underlying = unwrap_model(model)
    _atomic_torch_save({
        "model_state": underlying.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "pool_state": pool.state_dict(),
        "epoch": epoch,
        "step": step,
        # ذخیره‌ی steps_per_epoch_total: اگه بعداً فایل‌های dataset/ عوض بشن
        # (اضافه/کم شدن فایل)، این عدد با مقدار تازه‌محاسبه‌شده فرق می‌کنه و
        # load_checkpoint می‌تونه زودتر و با پیام روشن جلوی یک LR-schedule
        # خراب رو بگیره - دقیقاً همون گاردی که برای seq_length وجود داره.
        "steps_per_epoch_total": steps_per_epoch_total,
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


def load_checkpoint(txt_files: List[str]) -> Tuple[
    "TinyTransformer", torch.optim.Optimizer, DatasetPool, int, int, int,
    Dict[str, int], Dict[int, str]
]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    ix_to_char = {int(k): v for k, v in ckpt["ix_to_char"].items()}
    char_to_ix = ckpt["char_to_ix"]

    # اگه seq_length توی تنظیمات بالای فایل عوض شده باشه ولی checkpoint با seq_length
    # قدیمی ساخته شده باشه، pos_embed مدل با seq_length جدید (که get_batch ازش استفاده
    # می‌کنه) سازگار نیست و باعث خطای shape توی forward می‌شه. اینجا زودتر و با پیام
    # واضح جلوش رو می‌گیریم، به‌جای اینکه یه stack trace گنگ توی وسط training بیاد.
    if ckpt["seq_length"] != seq_length:
        raise ValueError(
            f"seq_length ناسازگار: این checkpoint با seq_length={ckpt['seq_length']} "
            f"ساخته شده ولی تنظیمات فعلی seq_length={seq_length} است. "
            f"یا seq_length رو به {ckpt['seq_length']} برگردون، یا checkpoint رو حذف "
            f"کن و از صفر شروع کن."
        )

    # همون‌طور برای steps_per_epoch_total: اگه فایل‌های dataset/ از زمان ساخت
    # این checkpoint عوض شده باشن (فایل اضافه/کم شده)، steps_per_epoch_total
    # تازه‌محاسبه‌شده با مقدار ذخیره‌شده فرق می‌کنه؛ ادامه دادن با مقدار
    # نامنطبق یعنی LR schedule (که به total_steps وابسته‌ست) بی‌صدا خراب
    # می‌شه. اینجا هم زودتر و با پیام روشن جلوش رو می‌گیریم.
    fresh_steps_per_epoch_total = compute_steps_per_epoch_total(txt_files)
    if ckpt["steps_per_epoch_total"] != fresh_steps_per_epoch_total:
        raise ValueError(
            f"ترکیب فایل‌های dataset/ از زمان ساخت این checkpoint عوض شده: "
            f"steps_per_epoch_total ذخیره‌شده={ckpt['steps_per_epoch_total']} ولی "
            f"مقدار تازه‌محاسبه‌شده از فایل‌های فعلی={fresh_steps_per_epoch_total}. "
            f"ادامه‌ی آموزش با این ناهماهنگی، زمان‌بندی نرخ یادگیری (LR schedule) رو "
            f"به‌هم می‌ریزه. یا فایل‌های dataset/ رو به حالت قبلی برگردون، یا "
            f"checkpoint رو حذف کن و از صفر با دیتاست جدید شروع کن."
        )

    model = TinyTransformer(
        ckpt["vocab_size"], ckpt["d_model"], ckpt["n_layers"],
        ckpt["n_heads"], ckpt["d_ff"], ckpt["seq_length"], ckpt["dropout_rate"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    optimizer = configure_optimizer(model, weight_decay, learning_rate)
    optimizer.load_state_dict(ckpt["optimizer_state"])

    pool = DatasetPool(
        txt_files, char_to_ix, seq_length, max_ram_mb, fresh_steps_per_epoch_total
    )
    pool.load_state_dict(ckpt["pool_state"])

    return (
        model, optimizer, pool, ckpt["epoch"], ckpt["step"],
        fresh_steps_per_epoch_total, char_to_ix, ix_to_char,
    )


# ================== آموزش ==================
def train(
    model: nn.Module,
    pool: DatasetPool,
    steps_per_epoch_total: int,
    char_to_ix: Dict[str, int],
    ix_to_char: Dict[int, str],
    optimizer: Optional[torch.optim.Optimizer] = None,
    start_epoch: int = 0,
    start_step: int = 0,
    checkpoint_every_steps: int = 200,
) -> Tuple[nn.Module, torch.optim.Optimizer, int]:
    """
    تفاوت اصلی با نسخه‌ی قبلی: قبلاً هر epoch یک بازه‌ی از پیش‌معلوم
    (range(remaining_steps)) بود، چون تمام دیتاست همیشه توی RAM و در دسترس
    بود. الان که فایل‌ها تدریجی لود/آفلود می‌شن، "پایان epoch" یک عدد از پیش
    دانسته نیست، بلکه یک *رویداد* است: pool.notify_step_consumed() وقتی همه‌ی
    فایل‌ها یک‌بار offload بشن True برمی‌گردونه. پس حلقه به‌جای شمردن step از
    قبل، منتظر همین سیگنال از خودِ pool می‌مونه.

    total_steps (برای LR schedule) هنوز از قبل معلومه (steps_per_epoch_total
    * epochs) چون این عدد فقط به حجم بایتی فایل‌ها بستگی داره، نه به اینکه
    چه‌وقت واقعاً لود می‌شن؛ در نتیجه warmup/decay دقیقاً مثل قبل، طبق step
    شماری مطلق پیش می‌ره و تحت تأثیر pooling قرار نمی‌گیره.
    """
    total_steps = steps_per_epoch_total * epochs

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
                # FIX #1: if pool is empty mid-epoch, let notify_step_consumed
                # try to load next files before get_batch crashes.
                if not pool.current_files():
                    epoch_finished = pool.notify_step_consumed([])
                    if epoch_finished:
                        break
                    if not pool.current_files():
                        raise RuntimeError(
                            "Pool خالیه و هیچ فایل جدیدی لود نشد. "
                            "ممکنه همه فایل‌های باقی‌مانده نامعتبر باشن یا "
                            "بزرگ‌تر از max_ram_mb باشن."
                        )

                xb, yb, file_ix_used = get_batch(pool, seq_length, batch_size, device)

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
                steps_this_epoch += 1
                pbar.update(1)

                # pool خودش تصمیم می‌گیره کدوم فایل باید offload و کدوم لود
                # بشه؛ خروجی True یعنی همه‌ی فایل‌های دیتاست یک‌بار (این
                # epoch) کامل دیده شدن.
                epoch_finished = pool.notify_step_consumed(file_ix_used)

                if step % checkpoint_every_steps == 0:
                    save_checkpoint(
                        model, optimizer, pool, epoch, step,
                        steps_per_epoch_total, char_to_ix, ix_to_char,
                    )

            pbar.close()
            avg_loss = total_loss / max(1, steps_this_epoch)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - LR: {lr:.6f}")
            epoch += 1
            save_checkpoint(
                model, optimizer, pool, epoch, step,
                steps_per_epoch_total, char_to_ix, ix_to_char,
            )

    except KeyboardInterrupt:
        print("\nآموزش با Ctrl+C متوقف شد - در حال ذخیره...")
        save_checkpoint(
            model, optimizer, pool, epoch, step,
            steps_per_epoch_total, char_to_ix, ix_to_char,
        )
        underlying = unwrap_model(model)
        save_model(underlying, char_to_ix, ix_to_char)
        print("ذخیره شد. برای ادامه‌ی آموزش، دوباره همین اسکریپت رو اجرا کن.")
        raise SystemExit(0)

    return model, optimizer, step


# ================== ذخیره / بارگذاری ==================
def save_model(
    model: nn.Module,
    char_to_ix: Dict[str, int],
    ix_to_char: Dict[int, str],
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
        "char_to_ix": char_to_ix,
        "ix_to_char": {str(k): v for k, v in ix_to_char.items()},
    }
    tmp = config_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False)
    os.replace(tmp, config_path)


def load_model() -> Tuple["TinyTransformer", Dict[str, int], Dict[int, str], int]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # همون دلیل load_checkpoint: اگه seq_length توی بالای فایل بعد از ذخیره‌ی این
    # مدل عوض شده باشه، ادامه‌ی آموزش با seq_length جدید روی pos_embed قدیمی خطا
    # می‌ده. اینجا هم زودتر با پیام روشن جلوش رو می‌گیریم.
    if config["seq_length"] != seq_length:
        raise ValueError(
            f"seq_length ناسازگار: مدل ذخیره‌شده با seq_length={config['seq_length']} "
            f"ساخته شده ولی تنظیمات فعلی seq_length={seq_length} است. "
            f"یا seq_length رو به {config['seq_length']} برگردون، یا مدل رو از صفر "
            f"دوباره آموزش بده."
        )

    ix_to_char = {int(k): v for k, v in config["ix_to_char"].items()}
    char_to_ix = config["char_to_ix"]
    model = TinyTransformer(
        config["vocab_size"], config["d_model"], config["n_layers"],
        config["n_heads"], config["d_ff"], config["seq_length"], config["dropout_rate"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    trained_epochs = config.get("trained_epochs", epochs)
    return model, char_to_ix, ix_to_char, trained_epochs


# ================== چت تعاملی ==================
def chat_loop(
    model: nn.Module, char_to_ix: Dict[str, int], ix_to_char: Dict[int, str]
) -> None:
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


def _maybe_compile(model: nn.Module, pool: DatasetPool) -> nn.Module:
    if not use_compile:
        return model
    try:
        compiled_model = torch.compile(model)
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

    if os.path.exists(checkpoint_path):
        print("Checkpoint ناقص پیدا شد، در حال ادامه‌ی آموزش از همون‌جا...")
        (
            model, optimizer, pool, start_epoch, start_step,
            steps_per_epoch_total, char_to_ix, ix_to_char,
        ) = load_checkpoint(txt_files)
        print(f"ادامه از epoch {start_epoch + 1}, step {start_step}")
        vocab_size = len(char_to_ix)

        model = _maybe_compile(model, pool)

        if start_epoch >= epochs:
            print(f"این checkpoint از قبل {epochs} epoch رو تموم کرده.")
            print("اگه می‌خوای بیشتر آموزش بدی، عدد epochs رو زیاد کن و دوباره اجرا کن.")
            underlying = unwrap_model(model)
            save_model(underlying, char_to_ix, ix_to_char, trained_epochs=epochs)
        else:
            model, optimizer, final_step = train(
                model, pool, steps_per_epoch_total, char_to_ix, ix_to_char,
                optimizer=optimizer, start_epoch=start_epoch, start_step=start_step,
            )
            os.remove(checkpoint_path)
            underlying = unwrap_model(model)
            save_model(underlying, char_to_ix, ix_to_char, trained_epochs=epochs)
            print(f"مدل نهایی ذخیره شد در {model_path}")

    elif os.path.exists(model_path) and os.path.exists(config_path):
        print("مدل ذخیره‌شده پیدا شد، در حال بارگذاری...")
        model, char_to_ix, ix_to_char, trained_epochs = load_model()
        vocab_size = len(char_to_ix)

        # اینجا هم مثل load_checkpoint: اگه فایل‌های dataset/ از زمان ذخیره‌ی
        # این مدل عوض شده باشن، steps_per_epoch_total فرق می‌کنه و LR schedule
        # یک آموزش تازه (از epoch=trained_epochs به بعد) با schedule متفاوتی
        # نسبت به قسمت قبلی آموزش پیش می‌ره. برخلاف load_checkpoint این حالت
        # چون آموزش قبلی کامل شده (checkpoint حذف شده)، این ناسازگاری مانع
        # کار نیست - فقط یک هشدار می‌دیم، چون این "ادامه‌ی آموزش" بیشتر شبیه
        # شروع یک دوره‌ی جدیده تا resume دقیق یک epoch نیمه‌کاره.
        steps_per_epoch_total = compute_steps_per_epoch_total(txt_files)

        if trained_epochs < epochs:
            print(f"مدل {trained_epochs} epoch آموزش دیده ولی epochs فعلی = {epochs}. ادامه‌ی آموزش...")
            pool = DatasetPool(txt_files, char_to_ix, seq_length, max_ram_mb, steps_per_epoch_total)
            model = _maybe_compile(model, pool)
            model, optimizer, final_step = train(
                model, pool, steps_per_epoch_total, char_to_ix, ix_to_char,
                start_epoch=trained_epochs,
            )
            underlying = unwrap_model(model)
            save_model(underlying, char_to_ix, ix_to_char, trained_epochs=epochs)
            print(f"مدل نهایی ذخیره شد در {model_path}")
        else:
            print(f"مدل کامل آموزش دیده ({trained_epochs}/{epochs} epoch) — مستقیم چت.")

    else:
        print("آموزش مدل از ابتدا...")
        # پاس اول سبک: فقط ساخت vocab (streaming، بدون نگه‌داشتن متن در RAM)
        chars, char_to_ix, ix_to_char = scan_vocab(txt_files)
        vocab_size = len(chars)
        steps_per_epoch_total = compute_steps_per_epoch_total(txt_files)
        total_bytes = sum(os.path.getsize(fp) for fp in txt_files)
        print(
            f"واژگان ساخته شد: {vocab_size} کاراکتر یکتا | "
            f"حجم کل دیتاست: {total_bytes / 1_000_000:.1f} مگابایت | "
            f"سقف pool: {max_ram_mb} مگابایت | "
            f"~{steps_per_epoch_total} step به‌ازای هر epoch"
        )

        model = TinyTransformer(vocab_size, d_model, n_layers, n_heads, d_ff, seq_length, dropout_rate).to(device)

        pool = DatasetPool(txt_files, char_to_ix, seq_length, max_ram_mb, steps_per_epoch_total)
        model = _maybe_compile(model, pool)

        model, optimizer, final_step = train(model, pool, steps_per_epoch_total, char_to_ix, ix_to_char)
        underlying = unwrap_model(model)
        save_model(underlying, char_to_ix, ix_to_char, trained_epochs=epochs)
        print(f"مدل ذخیره شد در {model_path}")
        model = underlying

    chat_loop(model, char_to_ix, ix_to_char)
