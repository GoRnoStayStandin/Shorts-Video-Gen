print("RUNNING FILE:", __file__)

import gc
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

# =========================
# OPTIONAL IMPORTS
# =========================

try:
    import psutil
except ImportError:
    psutil = None

try:
    from faster_whisper import WhisperModel, BatchedInferencePipeline
except ImportError:
    WhisperModel = None
    BatchedInferencePipeline = None

try:
    from gpt4all import GPT4All
except ImportError:
    GPT4All = None

try:
    from yt_dlp import YoutubeDL
except ImportError:
    YoutubeDL = None


# =========================
# CONFIG
# =========================

def get_env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on", "да"}


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


VIDEO_PATH = os.getenv("VIDEO_PATH", "data/sample.mp4")

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

FASTER_WHISPER_MODEL = os.getenv("FASTER_WHISPER_MODEL", "medium")
GPT4ALL_MODEL_PATH = os.getenv(
    "GPT4ALL_MODEL_PATH",
    "C:/vs/Practice/models/mistral-7b-instruct-v0.1.Q4_0.gguf"
)
MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))

def resolve_media_bin(env_name: str, executable: str) -> str:
    override = os.getenv(env_name)
    if override:
        return override

    found = shutil.which(executable)
    if found:
        return found

    local_appdata = os.getenv("LOCALAPPDATA")
    if local_appdata:
        winget_packages_dir = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        if winget_packages_dir.exists():
            matches = sorted(
                winget_packages_dir.glob(f"**/{executable}.exe"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if matches:
                return str(matches[0])

    return executable


FFMPEG_BIN = resolve_media_bin("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = resolve_media_bin("FFPROBE_BIN", "ffprobe")

LANGUAGE = os.getenv("LANGUAGE", "ru")

# ASR_DEVICE:
# - auto: попробовать CUDA, если видна NVIDIA GPU; при ошибке откатиться на CPU;
# - cuda: принудительно попробовать CUDA;
# - cpu: всегда CPU, удобно для ноутбука без NVIDIA/CUDA.
ASR_DEVICE = os.getenv("ASR_DEVICE", "auto").strip().lower()

# Обратная совместимость со старой переменной.
if "USE_CUDA_FOR_ASR" in os.environ:
    ASR_DEVICE = "cuda" if get_env_bool("USE_CUDA_FOR_ASR", False) else "cpu"

FW_CUDA_COMPUTE_TYPE = os.getenv("FW_CUDA_COMPUTE_TYPE", "float16")
FW_CPU_COMPUTE_TYPE = os.getenv("FW_CPU_COMPUTE_TYPE", "int8")
FW_GPU_BATCH_SIZE = get_env_int("FW_GPU_BATCH_SIZE", 8)
FW_CPU_BATCH_SIZE = get_env_int("FW_CPU_BATCH_SIZE", 1)
FW_BEAM_SIZE = get_env_int("FW_BEAM_SIZE", 5)
FW_VAD_FILTER = get_env_bool("FW_VAD_FILTER", True)
ENABLE_ASR_CACHE = get_env_bool("ENABLE_ASR_CACHE", True)
ASR_AUTO_TUNE = get_env_bool("ASR_AUTO_TUNE", True)
ASR_ALLOW_MODEL_DOWNGRADE = get_env_bool("ASR_ALLOW_MODEL_DOWNGRADE", True)
ASR_GPU_LOW_VRAM_MB = get_env_int("ASR_GPU_LOW_VRAM_MB", 6144)
ASR_GPU_MIN_FREE_VRAM_MB = get_env_int("ASR_GPU_MIN_FREE_VRAM_MB", 1024)

# GPT4All на Windows часто стабильнее на CPU. При желании можно поставить cuda/nvidia/kompute.
GPT4ALL_DEVICE = os.getenv("GPT4ALL_DEVICE", "cpu").strip().lower()
GPT4ALL_THREADS = get_env_int("GPT4ALL_THREADS", 0)
GPT4ALL_TOPIC_MAX_TOKENS = get_env_int("GPT4ALL_TOPIC_MAX_TOKENS", 220)
GPT4ALL_FAST_TOPIC_MAX_TOKENS = get_env_int("GPT4ALL_FAST_TOPIC_MAX_TOKENS", 90)
TOPIC_MAX_SEGMENTS_PER_WINDOW = get_env_int("TOPIC_MAX_SEGMENTS_PER_WINDOW", 12)
TOPIC_OVERLAP_SEGMENTS = get_env_int("TOPIC_OVERLAP_SEGMENTS", 2)
TOPIC_FAST_MAX_SEGMENTS_PER_WINDOW = get_env_int("TOPIC_FAST_MAX_SEGMENTS_PER_WINDOW", 24)
TOPIC_FAST_OVERLAP_SEGMENTS = get_env_int("TOPIC_FAST_OVERLAP_SEGMENTS", 0)
CHAPTER_BLOCK_TARGET_SEC = get_env_int("CHAPTER_BLOCK_TARGET_SEC", 30)
CHAPTER_BLOCK_MAX_CHARS = get_env_int("CHAPTER_BLOCK_MAX_CHARS", 90)
CHAPTER_MAX_BLOCKS_PER_WINDOW = get_env_int("CHAPTER_MAX_BLOCKS_PER_WINDOW", 64)
CHAPTER_OVERLAP_BLOCKS = get_env_int("CHAPTER_OVERLAP_BLOCKS", 2)
CHAPTER_MAX_TOKENS = get_env_int("CHAPTER_MAX_TOKENS", 220)

# Экспорт клипов. Если NVENC недоступен, код автоматически откатится на libx264.
USE_NVENC_FOR_EXPORT = get_env_bool("USE_NVENC_FOR_EXPORT", True)

# EXPORT_MODE=reencode — точное перекодирование; EXPORT_MODE=copy — максимально быстрый экспорт без перекодирования,
# но рез может попасть не точно в кадр, если start не на keyframe.
EXPORT_MODE = os.getenv("EXPORT_MODE", "reencode").strip().lower()

LAST_ASR_RUNTIME: Dict = {}

SUPPORTED_VIDEO_HOSTS = {
    "youtube.com",
    "youtu.be",
    "vk.com",
    "vkvideo.ru",
}


# =========================
# BASIC HELPERS
# =========================

def run_cmd(cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str, float]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )
    except FileNotFoundError as e:
        executable = cmd[0] if cmd else "unknown"
        raise RuntimeError(
            f"Не найден исполняемый файл `{executable}`. "
            "Установите FFmpeg/ffprobe или задайте переменные FFMPEG_BIN и FFPROBE_BIN. "
            "Если PATH был изменен недавно, перезапустите терминал и Streamlit."
        ) from e

    elapsed = time.perf_counter() - start
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def save_json(path: Path, obj):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json_if_exists(path: str):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None


def read_text_if_exists(path: str) -> Optional[str]:
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8", errors="ignore").strip()
    return None


def get_memory_mb() -> Optional[float]:
    if psutil is None:
        return None
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def get_video_duration_sec(video_path: str) -> float:
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    code, out, err, _ = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(f"ffprobe failed: {err}")
    return float(out.strip())


def extract_audio_wav(video_path: str, wav_path: str, sr: int = 16000):
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", str(video_path),
        "-ac", "1",
        "-ar", str(sr),
        "-vn",
        str(wav_path)
    ]
    code, _, err, _ = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {err}")


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}"


def format_srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_millis = int(round(seconds * 1000))
    hours = total_millis // 3600000
    minutes = (total_millis % 3600000) // 60000
    secs = (total_millis % 60000) // 1000
    millis = total_millis % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\sа-яё]", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_text_for_topic_analysis(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def make_topic_label(text: str, max_words: int = 8) -> str:
    text = normalize_text(text)
    words = text.split()
    if not words:
        return "topic"
    return "_".join(words[:max_words])


def safe_filename(name: str, max_len: int = 90) -> str:
    name = name.strip()
    name = re.sub(r"[^\w\dа-яА-ЯёЁ\-. ]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    return (name[:max_len] or "file").strip("_")


def is_supported_video_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())

    if parsed.scheme not in {"http", "https"}:
        return False

    host = (parsed.hostname or "").lower()

    if host.startswith("www."):
        host = host[4:]

    if host in SUPPORTED_VIDEO_HOSTS:
        return True

    return any(
        host.endswith(f".{base_host}")
        for base_host in SUPPORTED_VIDEO_HOSTS
        if base_host != "youtu.be"
    )


def download_video_from_url(
    url: str,
    out_dir: Path,
    progress_callback=None
) -> Path:
    """Скачивает публичное видео YouTube/VK через yt-dlp и возвращает путь к локальному файлу."""
    url = str(url or "").strip()

    if not is_supported_video_url(url):
        raise ValueError("Поддерживаются только публичные ссылки YouTube, YouTube Shorts, VK и VK Video.")

    if YoutubeDL is None:
        raise RuntimeError("yt-dlp не установлен. Обновите зависимости: pip install -r requirements.txt")

    out_dir.mkdir(exist_ok=True, parents=True)

    before_files = {p.resolve() for p in out_dir.iterdir() if p.is_file()}
    download_token = int(time.time())
    outtmpl = str(out_dir / f"{download_token}_%(extractor_key)s_%(id)s.%(ext)s")

    def ytdlp_progress_hook(status: Dict):
        if progress_callback is None:
            return

        status_name = status.get("status")

        if status_name == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            downloaded = status.get("downloaded_bytes") or 0

            if total > 0:
                progress = min(downloaded / total, 0.95)
                progress_callback(progress, f"Скачиваю видео: {progress * 100:.1f}%")
            else:
                progress_callback(0.10, "Скачиваю видео...")

        elif status_name == "finished":
            progress_callback(0.96, "Скачивание завершено, подготавливаю MP4...")

    ydl_opts = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [ytdlp_progress_hook],
        "restrictfilenames": True,
    }

    if progress_callback:
        progress_callback(0.02, "Получаю информацию о видео...")

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        candidate_paths: List[Path] = []

        for item in info.get("requested_downloads") or []:
            item_path = item.get("filepath") or item.get("_filename")
            if item_path:
                candidate_paths.append(Path(item_path))

        prepared_path = Path(ydl.prepare_filename(info))
        candidate_paths.append(prepared_path)
        candidate_paths.append(prepared_path.with_suffix(".mp4"))

    video_extensions = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
    new_video_files = [
        p
        for p in out_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in video_extensions
        and p.resolve() not in before_files
    ]

    candidate_paths.extend(new_video_files)

    existing_candidates = [
        p
        for p in candidate_paths
        if p.exists() and p.is_file() and p.suffix.lower() in video_extensions and p.stat().st_size > 0
    ]

    if not existing_candidates:
        raise RuntimeError("yt-dlp завершился, но итоговый видеофайл не найден.")

    video_path = max(existing_candidates, key=lambda p: p.stat().st_mtime)

    save_json(
        video_path.with_suffix(".source.json"),
        {
            "source_url": url,
            "title": info.get("title"),
            "extractor": info.get("extractor"),
            "extractor_key": info.get("extractor_key"),
            "webpage_url": info.get("webpage_url"),
            "duration": info.get("duration"),
            "downloaded_path": str(video_path),
        }
    )

    if progress_callback:
        progress_callback(1.0, "Видео скачано.")

    return video_path


def get_system_usage_snapshot() -> Dict:
    snapshot = {}

    if psutil is not None:
        vm = psutil.virtual_memory()
        snapshot.update({
            "cpu_percent": psutil.cpu_percent(interval=1.0),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_mb": vm.total / (1024 * 1024),
            "ram_used_mb": vm.used / (1024 * 1024),
            "ram_available_mb": vm.available / (1024 * 1024),
            "ram_percent": vm.percent,
        })

    return snapshot


def get_nvidia_gpu_snapshot() -> Dict:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name",
        "--format=csv,noheader,nounits"
    ]

    try:
        code, out, err, _ = run_cmd(cmd)
        if code != 0:
            return {"gpu_error": err.strip() or "nvidia-smi failed"}

        line = out.strip().splitlines()[0]
        parts = [x.strip() for x in line.split(",")]

        if len(parts) < 5:
            return {"gpu_error": f"unexpected nvidia-smi output: {line}"}

        return {
            "gpu_name": parts[4],
            "gpu_util_percent": float(parts[0]),
            "gpu_memory_used_mb": float(parts[1]),
            "gpu_memory_total_mb": float(parts[2]),
            "gpu_temperature_c": float(parts[3]),
        }
    except Exception as e:
        return {"gpu_error": str(e)}


# =========================
# ASR
# =========================

KNOWN_WHISPER_MODEL_ORDER = ["large-v3", "large-v2", "large", "medium", "small", "base"]


def parse_optional_float(value) -> Optional[float]:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(str(value).strip())
    except Exception:
        return None


def detect_hardware_profile() -> Dict:
    """Lightweight hardware snapshot for ASR auto-tuning."""
    profile: Dict = {
        "asr_device_request": ASR_DEVICE,
        "asr_auto_tune": ASR_AUTO_TUNE,
        "ram_total_mb": None,
        "ram_available_mb": None,
        "gpu_available": False,
        "gpu_name": None,
        "gpu_memory_total_mb": None,
        "gpu_memory_free_mb": None,
        "gpu_compute_capability": None,
        "gpu_error": None,
    }

    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            profile["ram_total_mb"] = round(vm.total / (1024 * 1024), 1)
            profile["ram_available_mb"] = round(vm.available / (1024 * 1024), 1)
        except Exception as e:
            profile["ram_error"] = str(e)

    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ]

    try:
        code, out, err, _ = run_cmd(cmd)
        if code != 0:
            profile["gpu_error"] = err.strip() or "nvidia-smi failed"
            return profile

        line = out.strip().splitlines()[0]
        parts = [part.strip() for part in line.split(",")]

        if len(parts) < 3:
            profile["gpu_error"] = f"unexpected nvidia-smi output: {line}"
            return profile

        profile.update({
            "gpu_available": True,
            "gpu_name": parts[0],
            "gpu_memory_total_mb": parse_optional_float(parts[1]),
            "gpu_memory_free_mb": parse_optional_float(parts[2]),
            "gpu_compute_capability": parts[3] if len(parts) >= 4 else None,
        })
    except Exception as e:
        profile["gpu_error"] = str(e)

    return profile


def build_whisper_model_ladder(selected_model: str, allow_downgrade: bool = ASR_ALLOW_MODEL_DOWNGRADE) -> List[str]:
    selected_model = str(selected_model or "medium").strip() or "medium"

    if not allow_downgrade:
        return [selected_model]

    selected_key = selected_model.lower()
    lower_order = [name.lower() for name in KNOWN_WHISPER_MODEL_ORDER]

    if selected_key in lower_order:
        start_index = lower_order.index(selected_key)
        return KNOWN_WHISPER_MODEL_ORDER[start_index:]

    # Custom/HF model IDs are tried first, then safe built-in fallbacks.
    ladder = [selected_model]
    for fallback in ("small", "base"):
        if fallback.lower() != selected_key:
            ladder.append(fallback)
    return ladder


def add_asr_profile_once(profiles: List[Dict], profile: Dict):
    key = (
        profile.get("model"),
        profile.get("device"),
        profile.get("compute_type"),
        int(profile.get("batch_size", 1)),
        int(profile.get("beam_size", FW_BEAM_SIZE)),
    )

    for existing in profiles:
        existing_key = (
            existing.get("model"),
            existing.get("device"),
            existing.get("compute_type"),
            int(existing.get("batch_size", 1)),
            int(existing.get("beam_size", FW_BEAM_SIZE)),
        )
        if existing_key == key:
            return

    profile["index"] = len(profiles) + 1
    profiles.append(profile)


def choose_gpu_batch_size(hardware: Dict) -> int:
    if "FW_GPU_BATCH_SIZE" in os.environ:
        return max(1, FW_GPU_BATCH_SIZE)

    total_vram = float(hardware.get("gpu_memory_total_mb") or 0)
    free_vram = float(hardware.get("gpu_memory_free_mb") or 0)
    usable_vram = free_vram if free_vram > 0 else total_vram

    if total_vram >= 20000 and usable_vram >= 12000:
        return 8
    if total_vram >= 12000 and usable_vram >= 7000:
        return 4
    if total_vram >= 8000 and usable_vram >= 4500:
        return 2
    return 1


def choose_gpu_compute_types(hardware: Dict) -> List[str]:
    if "FW_CUDA_COMPUTE_TYPE" in os.environ:
        preferred = FW_CUDA_COMPUTE_TYPE
        compute_types = [preferred, "int8_float16", "int8", "float16"]
    else:
        total_vram = float(hardware.get("gpu_memory_total_mb") or 0)
        free_vram = float(hardware.get("gpu_memory_free_mb") or 0)

        if total_vram >= 12000 and free_vram >= 7000:
            preferred = "float16"
            compute_types = [preferred, "int8_float16", "int8"]
        else:
            preferred = "int8_float16"
            compute_types = [preferred, "int8"]

    result = []

    for compute_type in compute_types:
        if compute_type and compute_type not in result:
            result.append(compute_type)

    return result


def choose_asr_beam_size(hardware: Dict, device: str) -> int:
    if "FW_BEAM_SIZE" in os.environ:
        return max(1, FW_BEAM_SIZE)

    if device == "cuda":
        total_vram = float(hardware.get("gpu_memory_total_mb") or 0)
        if total_vram and total_vram < ASR_GPU_LOW_VRAM_MB:
            return 1
        if total_vram and total_vram < 12000:
            return 3

    return FW_BEAM_SIZE


def build_asr_profile_ladder() -> Tuple[Dict, List[Dict]]:
    hardware = detect_hardware_profile()
    requested_device = (ASR_DEVICE or "auto").strip().lower()
    selected_model = str(FASTER_WHISPER_MODEL or "medium").strip() or "medium"
    model_ladder = build_whisper_model_ladder(selected_model)
    profiles: List[Dict] = []

    if requested_device == "cpu":
        cpu_beam_size = choose_asr_beam_size(hardware, "cpu")
        for model_name in model_ladder:
            add_asr_profile_once(profiles, {
                "model": model_name,
                "device": "cpu",
                "compute_type": FW_CPU_COMPUTE_TYPE,
                "batch_size": max(1, FW_CPU_BATCH_SIZE),
                "beam_size": cpu_beam_size,
                "reason": "ASR_DEVICE=cpu",
            })
        return hardware, profiles

    can_try_gpu = requested_device in {"auto", "cuda"} and bool(hardware.get("gpu_available"))

    if can_try_gpu and ASR_AUTO_TUNE:
        gpu_batch_size = choose_gpu_batch_size(hardware)
        gpu_beam_size = choose_asr_beam_size(hardware, "cuda")
        compute_types = choose_gpu_compute_types(hardware)
        total_vram = float(hardware.get("gpu_memory_total_mb") or 0)
        free_vram = float(hardware.get("gpu_memory_free_mb") or 0)
        reason = "auto_gpu"

        if total_vram and total_vram < ASR_GPU_LOW_VRAM_MB:
            reason = "auto_gpu_low_vram"
        elif free_vram and free_vram < ASR_GPU_MIN_FREE_VRAM_MB:
            reason = "auto_gpu_low_free_vram"

        # Try the requested model first, then lower models, keeping batch=1 fallbacks to stay on GPU.
        for model_index, model_name in enumerate(model_ladder):
            for compute_type in compute_types:
                batch_size = gpu_batch_size if model_index == 0 and compute_type == compute_types[0] else 1
                add_asr_profile_once(profiles, {
                    "model": model_name,
                    "device": "cuda",
                    "compute_type": compute_type,
                    "batch_size": batch_size,
                    "beam_size": gpu_beam_size,
                    "reason": reason if model_index == 0 else "auto_gpu_model_downgrade",
                })
    elif can_try_gpu:
        add_asr_profile_once(profiles, {
            "model": selected_model,
            "device": "cuda",
            "compute_type": FW_CUDA_COMPUTE_TYPE,
            "batch_size": max(1, FW_GPU_BATCH_SIZE),
            "beam_size": choose_asr_beam_size(hardware, "cuda"),
            "reason": "manual_gpu_profile",
        })

    cpu_beam_size = choose_asr_beam_size(hardware, "cpu")
    for model_name in model_ladder:
        add_asr_profile_once(profiles, {
            "model": model_name,
            "device": "cpu",
            "compute_type": FW_CPU_COMPUTE_TYPE,
            "batch_size": max(1, FW_CPU_BATCH_SIZE),
            "beam_size": cpu_beam_size,
            "reason": "cpu_fallback",
        })

    return hardware, profiles


def has_nvidia_gpu() -> bool:
    """Быстрая проверка: видит ли система NVIDIA GPU через nvidia-smi."""
    try:
        code, out, _, _ = run_cmd(["nvidia-smi", "-L"])
        return code == 0 and "GPU" in out
    except Exception:
        return False


def resolve_asr_device() -> str:
    """Возвращает целевое устройство для faster-whisper: cuda или cpu."""
    if ASR_DEVICE == "cpu":
        return "cpu"

    if ASR_DEVICE == "cuda":
        return "cuda"

    if ASR_DEVICE == "auto" and has_nvidia_gpu():
        return "cuda"

    return "cpu"


def load_faster_whisper_model(preferred_device: Optional[str] = None, profile: Optional[Dict] = None):
    """
    Загружает faster-whisper с автоматическим выбором CUDA/CPU.

    На мощном ПК с NVIDIA сначала пробуем CUDA. Если не хватает CUDA/cuBLAS/cuDNN
    или возникает ошибка загрузки, автоматически откатываемся на CPU int8, чтобы
    приложение не падало на ноутбуке или неподготовленной системе.
    """
    if WhisperModel is None:
        raise RuntimeError("faster-whisper не установлен")

    if profile is None:
        target_device = (preferred_device or resolve_asr_device()).strip().lower()
        profile = {
            "model": FASTER_WHISPER_MODEL,
            "device": target_device,
            "compute_type": FW_CUDA_COMPUTE_TYPE if target_device == "cuda" else FW_CPU_COMPUTE_TYPE,
            "batch_size": FW_GPU_BATCH_SIZE if target_device == "cuda" else FW_CPU_BATCH_SIZE,
            "beam_size": FW_BEAM_SIZE,
            "reason": "direct_load",
            "index": 1,
        }

    model_name = str(profile.get("model") or FASTER_WHISPER_MODEL)
    device = str(profile.get("device") or "cpu")
    compute_type = str(profile.get("compute_type") or (FW_CUDA_COMPUTE_TYPE if device == "cuda" else FW_CPU_COMPUTE_TYPE))

    print(
        f"[INFO] Loading faster-whisper model={model_name} "
        f"device={device} compute_type={compute_type} "
        f"batch={profile.get('batch_size', 1)} reason={profile.get('reason', 'unknown')}"
    )
    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type
    )
    LAST_ASR_RUNTIME.clear()
    LAST_ASR_RUNTIME.update({
        "device": device,
        "compute_type": compute_type,
        "model": model_name,
        "batch_size": int(profile.get("batch_size", 1) or 1),
        "beam_size": int(profile.get("beam_size", FW_BEAM_SIZE) or FW_BEAM_SIZE),
        "profile_index": profile.get("index"),
        "profile_reason": profile.get("reason"),
    })
    return model


def run_faster_whisper_transcribe(model, wav_path: Path, profile: Optional[Dict] = None):
    """
    Запускает обычную или batched-транскрибацию в зависимости от устройства.

    Важно: в новых версиях faster-whisper BatchedInferencePipeline требует
    либо vad_filter=True, либо clip_timestamps. Без этого появляется ошибка:
    "No clip timestamps found. Set 'vad_filter' to True or provide 'clip_timestamps'."

    Поэтому для batched-режима VAD включается принудительно. Если batched-режим
    всё равно падает не из-за CUDA, код откатывается на обычный transcribe()
    на том же устройстве, чтобы обработка не прерывалась.
    """
    device = str((profile or {}).get("device") or LAST_ASR_RUNTIME.get("device", "cpu"))
    batch_size = int((profile or {}).get("batch_size") or LAST_ASR_RUNTIME.get("batch_size") or (FW_GPU_BATCH_SIZE if device == "cuda" else FW_CPU_BATCH_SIZE))
    beam_size = int((profile or {}).get("beam_size") or LAST_ASR_RUNTIME.get("beam_size") or FW_BEAM_SIZE)

    base_kwargs = {
        "language": LANGUAGE,
        "beam_size": beam_size,
        "vad_filter": FW_VAD_FILTER,
    }

    if batch_size > 1 and BatchedInferencePipeline is not None:
        print(f"[INFO] Using faster-whisper batched inference, batch_size={batch_size}")
        batched_model = BatchedInferencePipeline(model=model)

        batched_kwargs = dict(base_kwargs)
        batched_kwargs["vad_filter"] = True
        batched_kwargs.setdefault("vad_parameters", {"min_silence_duration_ms": 500})

        try:
            return batched_model.transcribe(
                str(wav_path),
                batch_size=batch_size,
                **batched_kwargs
            )
        except Exception as e:
            error_text = str(e).lower()

            # Ошибки CUDA пусть обработает внешний fallback на CPU.
            if any(token in error_text for token in ("cuda", "cublas", "cudnn", "out of memory", "cublas64")):
                raise

            print(f"[WARN] Batched faster-whisper failed: {e}")
            print("[INFO] Retrying faster-whisper without batched inference on the same device...")
            return model.transcribe(str(wav_path), **base_kwargs)

    if batch_size > 1 and BatchedInferencePipeline is None:
        print("[WARN] BatchedInferencePipeline недоступен в установленной версии faster-whisper. Использую обычный режим.")

    return model.transcribe(str(wav_path), **base_kwargs)


def is_cuda_runtime_error(error: Exception) -> bool:
    """Проверяет, относится ли ошибка к CUDA/cuBLAS/cuDNN/VRAM."""
    error_text = str(error).lower()
    cuda_markers = (
        "cuda",
        "cublas",
        "cudnn",
        "cublas64",
        "cudnn_ops",
        "cudnn_cnn",
        "out of memory",
        "no kernel image",
        "driver version",
    )
    return any(marker in error_text for marker in cuda_markers)


def transcribe_with_faster_whisper(
    video_path: str,
    out_dir: Path,
    progress_callback=None
) -> List[Dict]:
    """
    Возвращает ASR-сегменты:
    [
      {"start": 0.0, "end": 3.2, "text": "..."},
      ...
    ]

    Важный момент: faster-whisper возвращает ленивый генератор сегментов.
    Поэтому часть CUDA-ошибок, например OOM или отсутствие cublas64_12.dll,
    возникает не в момент вызова model.transcribe(), а позже — при проходе по
    segments_iter. Из-за этого весь проход находится внутри try/except, а ASR
    использует GPU-first лестницу профилей перед CPU fallback.
    """
    out_dir.mkdir(exist_ok=True, parents=True)

    media_duration = get_video_duration_sec(str(video_path))

    wav_path = out_dir / "audio_for_faster_whisper.wav"
    txt_path = out_dir / "faster_whisper.txt"
    timed_txt_path = out_dir / "faster_whisper_timestamps.txt"
    segments_json_path = out_dir / "faster_whisper_segments.json"
    srt_path = out_dir / "faster_whisper.srt"

    if ENABLE_ASR_CACHE and segments_json_path.exists():
        cached_segments = read_json_if_exists(str(segments_json_path))
        if isinstance(cached_segments, list) and cached_segments:
            print(f"[INFO] Reusing cached ASR segments: {segments_json_path}")
            if progress_callback:
                progress_callback(0.65, "Использую кэш распознавания речи...")
            return cached_segments

    if progress_callback:
        progress_callback(0.08, "Извлекаю аудио из видео...")

    extract_audio_wav(str(video_path), str(wav_path), sr=16000)

    hardware_profile, asr_profiles = build_asr_profile_ladder()
    save_json(
        out_dir / "asr_auto_profile_ladder.json",
        {
            "hardware": hardware_profile,
            "profiles": asr_profiles,
        }
    )

    if not asr_profiles:
        raise RuntimeError("Не удалось построить ни одного ASR-профиля.")

    def run_and_collect(profile: Dict) -> Tuple[List[str], List[str], List[Dict], List[str]]:
        profile_index = int(profile.get("index") or 1)
        profile_count = len(asr_profiles)
        model_name = str(profile.get("model") or FASTER_WHISPER_MODEL)
        device_label = str(profile.get("device") or "auto")
        compute_type = str(profile.get("compute_type") or "default")
        batch_size = int(profile.get("batch_size") or 1)

        if progress_callback:
            progress_callback(
                0.15,
                f"Загружаю faster-whisper: профиль {profile_index}/{profile_count} "
                f"({model_name}, {device_label}, {compute_type}, batch {batch_size})..."
            )

        model = load_faster_whisper_model(profile=profile)

        if progress_callback:
            progress_callback(
                0.22,
                f"Распознаю речь ({device_label}, {model_name}, {compute_type}, batch {batch_size})..."
            )

        segments_iter, _info = run_faster_whisper_transcribe(model, wav_path, profile=profile)

        transcript_parts: List[str] = []
        timed_lines: List[str] = []
        segments_json: List[Dict] = []
        srt_blocks: List[str] = []

        # Ошибки CUDA/cuBLAS/cuDNN могут возникнуть именно здесь, потому что
        # segments_iter — ленивый генератор.
        for _idx, seg in enumerate(segments_iter, start=1):
            seg_start = float(seg.start)
            seg_end = float(seg.end)
            seg_text = seg.text.strip()

            if not seg_text:
                continue

            transcript_parts.append(seg_text)

            timed_lines.append(
                f"{format_timestamp(seg_start)} - {format_timestamp(seg_end)} {seg_text}"
            )

            segments_json.append({
                "start": seg_start,
                "end": seg_end,
                "text": seg_text
            })

            srt_blocks.append(
                f"{len(srt_blocks) + 1}\n"
                f"{format_srt_timestamp(seg_start)} --> {format_srt_timestamp(seg_end)}\n"
                f"{seg_text}\n"
            )

            if progress_callback and media_duration > 0:
                local = min(seg_end / media_duration, 1.0)
                runtime_device = LAST_ASR_RUNTIME.get("device", "unknown")
                runtime_model = LAST_ASR_RUNTIME.get("model", "unknown")
                progress_callback(
                    0.22 + local * 0.43,
                    f"Распознаю речь ({runtime_device}, {runtime_model}): "
                    f"{format_timestamp(seg_end)} / {format_timestamp(media_duration)}"
                )

        return transcript_parts, timed_lines, segments_json, srt_blocks

    errors = []
    transcript_parts: List[str] = []
    timed_lines: List[str] = []
    segments_json: List[Dict] = []
    srt_blocks: List[str] = []

    for profile in asr_profiles:
        try:
            transcript_parts, timed_lines, segments_json, srt_blocks = run_and_collect(profile)
            break
        except Exception as e:
            profile_label = (
                f"{profile.get('model')} / {profile.get('device')} / "
                f"{profile.get('compute_type')} / batch {profile.get('batch_size')}"
            )
            errors.append(f"{profile_label}: {e}")
            print(f"[WARN] faster-whisper failed on ASR profile {profile_label}: {e}")

            gc.collect()

            is_last_profile = int(profile.get("index") or 0) >= len(asr_profiles)
            if is_last_profile:
                raise RuntimeError("Не удалось выполнить ASR ни на одном профиле:\n" + "\n".join(errors)) from e

            next_profile = asr_profiles[int(profile.get("index") or 1)]
            if progress_callback:
                progress_callback(
                    0.20,
                    "Текущий ASR-профиль упал. Пробую следующий: "
                    f"{next_profile.get('model')} / {next_profile.get('device')} / "
                    f"{next_profile.get('compute_type')} / batch {next_profile.get('batch_size')}..."
                )

            continue

    if not segments_json:
        print("[WARN] ASR completed but returned no speech segments.")

    txt_path.write_text(" ".join(transcript_parts).strip(), encoding="utf-8")
    timed_txt_path.write_text("\n".join(timed_lines), encoding="utf-8")
    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")
    save_json(segments_json_path, segments_json)
    save_json(
        out_dir / "asr_runtime_profile.json",
        {
            **LAST_ASR_RUNTIME,
            "hardware": hardware_profile,
            "profile_count": len(asr_profiles),
        }
    )

    return segments_json

# =========================
# GPT4ALL
# =========================

def load_gpt4all_model(n_ctx: int = 4096, model_path: Optional[str] = None):
    """
    Загружает GPT4All.
    Важно: у некоторых версий gpt4all параметры n_ctx/device могут отличаться,
    поэтому есть несколько попыток.
    """
    if GPT4All is None:
        raise RuntimeError("gpt4all не установлен")

    selected_model_path = str(model_path or GPT4ALL_MODEL_PATH)

    if not Path(selected_model_path).exists():
        raise RuntimeError(f"Модель GPT4All не найдена: {selected_model_path}")

    base_kwargs = {
        "model_name": selected_model_path,
        "allow_download": False,
        "n_ctx": n_ctx,
    }

    if GPT4ALL_THREADS > 0:
        base_kwargs["n_threads"] = GPT4ALL_THREADS

    attempts = []

    if GPT4ALL_DEVICE and GPT4ALL_DEVICE not in {"auto", "none"}:
        attempts.append({**base_kwargs, "device": GPT4ALL_DEVICE})

    # CPU fallback — самый стабильный вариант на Windows.
    attempts.append({**base_kwargs, "device": "cpu"})
    attempts.append({"model_name": selected_model_path, "allow_download": False, "device": "cpu"})
    attempts.append({"model_name": selected_model_path, "allow_download": False})

    last_error = None

    for kwargs in attempts:
        try:
            print(
                f"[INFO] Loading GPT4All model={selected_model_path} "
                f"device={kwargs.get('device', 'default')} n_ctx={kwargs.get('n_ctx', 'default')}"
            )
            return GPT4All(**kwargs)
        except TypeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            print(f"[WARN] GPT4All load failed with {kwargs.get('device', 'default')}: {e}")
            continue

    raise RuntimeError(f"Не удалось загрузить GPT4All: {last_error}")


def extract_json_from_model_answer(answer: str):
    if not answer:
        return None

    text = answer.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1

            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None

    return None


def analyze_topic_with_gpt4all_model(model, text: str) -> Dict:
    """
    Используется для перегенерации title/summary выбранного клипа.
    """
    text = clean_text_for_topic_analysis(text)[:2500]

    prompt = f"""
Проанализируй фрагмент транскрипта видео.

Правила:
- отвечай только на русском языке;
- не добавляй фактов, которых нет в тексте;
- title короткий;
- summary 1-2 предложения;
- ответ строго в формате:
TITLE: ...
SUMMARY: ...

Текст:
{text}
""".strip()

    with model.chat_session():
        answer = model.generate(prompt, max_tokens=140, temp=0.2)

    title = None
    summary = None

    for line in answer.splitlines():
        line = line.strip()

        if line.upper().startswith("TITLE:"):
            title = line.split(":", 1)[1].strip()
        elif line.upper().startswith("SUMMARY:"):
            summary = line.split(":", 1)[1].strip()

    if not title:
        title = make_topic_label(text, max_words=6).replace("_", " ")

    if not summary:
        summary = answer.strip()

    return {
        "title": title,
        "summary": summary,
        "error": None
    }


# =========================
# LLM TOPIC SEGMENTATION
# =========================

def prepare_whisper_segments_with_ids(segments: List[Dict]) -> List[Dict]:
    prepared = []

    for i, seg in enumerate(segments):
        text = clean_text_for_topic_analysis(seg.get("text", ""))

        if not text:
            continue

        prepared.append({
            "id": i,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": text
        })

    return prepared


def format_segment_for_llm(seg: Dict) -> str:
    """
    Специально НЕ добавляем таймкоды в prompt, чтобы экономить context window.
    Модель возвращает границы через id, а код сам сопоставляет id с таймкодами.
    """
    return f"[{seg['id']}] {seg['text']}"


def split_segments_for_llm_by_count(
    segments: List[Dict],
    max_segments_per_window: int = 8,
    overlap_segments: int = 2
) -> List[List[Dict]]:
    if not segments:
        return []

    max_segments_per_window = max(3, int(max_segments_per_window))
    overlap_segments = max(0, min(int(overlap_segments), max_segments_per_window - 1))

    windows = []
    start = 0

    while start < len(segments):
        end = min(start + max_segments_per_window, len(segments))
        window = segments[start:end]

        if window:
            windows.append(window)

        if end >= len(segments):
            break

        start = max(end - overlap_segments, start + 1)

    return windows


def truncate_text_for_llm(text: str, max_chars: int) -> str:
    text = clean_text_for_topic_analysis(text)
    max_chars = max(20, int(max_chars))

    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    return truncated or text[:max_chars].strip()


def build_chapter_blocks_from_segments(
    whisper_segments: List[Dict],
    target_duration_sec: int = CHAPTER_BLOCK_TARGET_SEC,
    max_text_chars: int = CHAPTER_BLOCK_MAX_CHARS
) -> List[Dict]:
    prepared_segments = prepare_whisper_segments_with_ids(whisper_segments)

    if not prepared_segments:
        return []

    target_duration_sec = max(10, int(target_duration_sec))
    blocks: List[Dict] = []
    current: List[Dict] = []

    def flush_current():
        if not current:
            return

        text = " ".join(seg.get("text", "").strip() for seg in current).strip()
        blocks.append({
            "id": len(blocks),
            "start_id": int(current[0]["id"]),
            "end_id": int(current[-1]["id"]),
            "start": float(current[0]["start"]),
            "end": float(current[-1]["end"]),
            "text": truncate_text_for_llm(text, max_text_chars),
        })

    for seg in prepared_segments:
        current.append(seg)

        if float(current[-1]["end"]) - float(current[0]["start"]) >= target_duration_sec:
            flush_current()
            current = []

    flush_current()
    return blocks


def format_chapter_block_for_llm(block: Dict) -> str:
    return (
        f"[{block['id']}] "
        f"{format_timestamp(float(block['start']))}-{format_timestamp(float(block['end']))}: "
        f"{block.get('text', '')}"
    )


def ask_gpt4all_for_chapters_in_block_window(
    model,
    block_window: List[Dict],
    include_metadata: bool = False,
    max_tokens: Optional[int] = None
) -> List[Dict]:
    if not block_window:
        return []

    blocks_text = "\n".join(format_chapter_block_for_llm(block) for block in block_window)
    first_id = int(block_window[0]["id"])
    last_id = int(block_window[-1]["id"])

    if include_metadata:
        prompt = f"""
Раздели блоки транскрипта на крупные главы видео.

Правила:
- глава = крупный раздел видео, похожий на авторское оглавление;
- не ищи Shorts, хайлайты или микро-подтемы;
- не отделяй отдельный пример или мысль, если они относятся к той же главе;
- главы должны идти по порядку, не пересекаться и вместе покрывать диапазон блоков;
- если весь диапазон относится к одной главе, верни один диапазон;
- start_block_id и end_block_id бери только из списка;
- end_block_id включается;
- ответ только JSON.

Формат:
{{"chapters":[{{"title":"...","summary":"...","start_block_id":{first_id},"end_block_id":{last_id}}}]}}

Блоки:
{blocks_text}
""".strip()
    else:
        prompt = f"""
Раздели блоки транскрипта на крупные главы видео.

Правила:
- глава = крупный раздел видео, похожий на авторское оглавление;
- не ищи Shorts, хайлайты или микро-подтемы;
- не отделяй отдельный пример или мысль, если они относятся к той же главе;
- главы должны идти по порядку, не пересекаться и вместе покрывать диапазон блоков;
- если весь диапазон относится к одной главе, верни один диапазон;
- start_block_id и end_block_id бери только из списка;
- end_block_id включается;
- не пиши title и summary;
- ответ только JSON.

Формат:
{{"chapters":[{{"start_block_id":{first_id},"end_block_id":{last_id}}}]}}

Блоки:
{blocks_text}
""".strip()

    token_limit = int(max_tokens or CHAPTER_MAX_TOKENS)

    try:
        with model.chat_session():
            answer = model.generate(prompt, max_tokens=token_limit, temp=0.1)
    except Exception as e:
        print(f"[WARN] GPT4All chapter window failed: {e}")
        return []

    if not answer:
        return []

    if "context window" in answer.lower() or "prompt is" in answer.lower():
        print(f"[WARN] GPT4All context error: {answer[:300]}")
        return []

    parsed = extract_json_from_model_answer(answer)

    if not parsed:
        print("[WARN] Could not parse JSON from GPT4All chapter answer")
        print(answer[:500])
        return []

    chapters = parsed.get("chapters") or parsed.get("topics") or []

    if not isinstance(chapters, list):
        return []

    allowed_ids = {int(block["id"]) for block in block_window}
    min_id = min(allowed_ids)
    max_id = max(allowed_ids)
    valid_chapters = []

    for chapter in chapters:
        try:
            start_block_id = int(chapter.get("start_block_id", chapter.get("start_id")))
            end_block_id = int(chapter.get("end_block_id", chapter.get("end_id")))
        except Exception:
            continue

        if end_block_id < start_block_id:
            start_block_id, end_block_id = end_block_id, start_block_id

        start_block_id = max(min_id, min(start_block_id, max_id))
        end_block_id = max(min_id, min(end_block_id, max_id))

        if start_block_id not in allowed_ids or end_block_id not in allowed_ids:
            continue

        valid_chapters.append({
            "title": str(chapter.get("title", "")).strip(),
            "summary": str(chapter.get("summary", "")).strip(),
            "start_block_id": start_block_id,
            "end_block_id": end_block_id,
        })

    return valid_chapters


def ask_gpt4all_for_topics_in_window(
    model,
    window_segments: List[Dict],
    include_metadata: bool = True,
    max_tokens: Optional[int] = None
) -> List[Dict]:
    """
    Просит GPT4All выделить подтемы внутри маленького окна ASR-сегментов.
    Prompt короткий, чтобы помещаться даже в context window 2048.
    """
    segments_text = "\n".join(format_segment_for_llm(seg) for seg in window_segments)

    first_id = int(window_segments[0]["id"])
    last_id = int(window_segments[-1]["id"])

    if include_metadata:
        prompt = f"""
Раздели ASR-сегменты видео на смысловые подтемы.

Правила:
- новая подтема = новая мысль, пример, вопрос, проблема, решение или вывод;
- не объединяй разные мысли в одну тему;
- start_id и end_id бери только из списка;
- end_id включается;
- ответ только JSON.

Формат:
{{"topics":[{{"title":"...","summary":"...","start_id":{first_id},"end_id":{last_id}}}]}}

Сегменты:
{segments_text}
""".strip()
    else:
        prompt = f"""
Раздели ASR-сегменты видео на смысловые подтемы.

Правила:
- новая подтема = новая мысль, пример, вопрос, проблема, решение или вывод;
- start_id и end_id бери только из списка;
- end_id включается;
- не пиши title и summary;
- ответ только JSON.

Формат:
{{"topics":[{{"start_id":{first_id},"end_id":{last_id}}}]}}

Сегменты:
{segments_text}
""".strip()

    token_limit = int(max_tokens or (GPT4ALL_TOPIC_MAX_TOKENS if include_metadata else GPT4ALL_FAST_TOPIC_MAX_TOKENS))

    try:
        with model.chat_session():
            answer = model.generate(prompt, max_tokens=token_limit, temp=0.1)
    except Exception as e:
        print(f"[WARN] GPT4All topic window failed: {e}")
        return []

    if not answer:
        return []

    if "context window" in answer.lower() or "prompt is" in answer.lower():
        print(f"[WARN] GPT4All context error: {answer[:300]}")
        return []

    parsed = extract_json_from_model_answer(answer)

    if not parsed:
        print("[WARN] Could not parse JSON from GPT4All answer")
        print(answer[:500])
        return []

    topics = parsed.get("topics", [])

    if not isinstance(topics, list):
        return []

    valid_topics = []

    allowed_ids = {int(seg["id"]) for seg in window_segments}
    min_id = min(allowed_ids)
    max_id = max(allowed_ids)

    for topic in topics:
        try:
            start_id = int(topic.get("start_id"))
            end_id = int(topic.get("end_id"))
        except Exception:
            continue

        if end_id < start_id:
            start_id, end_id = end_id, start_id

        start_id = max(min_id, min(start_id, max_id))
        end_id = max(min_id, min(end_id, max_id))

        if start_id not in allowed_ids or end_id not in allowed_ids:
            continue

        title = str(topic.get("title", "")).strip()
        summary = str(topic.get("summary", "")).strip()

        valid_topics.append({
            "title": title,
            "summary": summary,
            "start_id": start_id,
            "end_id": end_id
        })

    return valid_topics


def build_topic_from_segment_ids(
    segments_by_id: Dict[int, Dict],
    start_id: int,
    end_id: int,
    title: str = "",
    summary: str = ""
) -> Optional[Dict]:
    if end_id < start_id:
        start_id, end_id = end_id, start_id

    ids = [i for i in range(start_id, end_id + 1) if i in segments_by_id]

    if not ids:
        return None

    selected = [segments_by_id[i] for i in ids]
    text = " ".join(seg.get("text", "").strip() for seg in selected).strip()

    if not text:
        return None

    start = float(selected[0]["start"])
    end = float(selected[-1]["end"])
    duration = end - start

    return {
        "start_id": ids[0],
        "end_id": ids[-1],
        "start": start,
        "end": end,
        "duration": duration,
        "title": title.strip() if title else make_topic_label(text, max_words=6).replace("_", " "),
        "summary": summary.strip() if summary else "",
        "text": text
    }


def topic_range_iou(a: Dict, b: Dict) -> float:
    a1, a2 = int(a["start_id"]), int(a["end_id"])
    b1, b2 = int(b["start_id"]), int(b["end_id"])

    inter = max(0, min(a2, b2) - max(a1, b1) + 1)
    union = max(a2, b2) - min(a1, b1) + 1

    return inter / union if union > 0 else 0.0


def choose_better_topic(a: Dict, b: Dict) -> Dict:
    """
    Для почти одинаковых тем оставляем более информативную.
    """
    len_a = int(a["end_id"]) - int(a["start_id"])
    len_b = int(b["end_id"]) - int(b["start_id"])

    score_a = len_a + (2 if a.get("summary") else 0) + (1 if a.get("title") else 0)
    score_b = len_b + (2 if b.get("summary") else 0) + (1 if b.get("title") else 0)

    return a if score_a >= score_b else b


def deduplicate_topic_candidates(
    candidates: List[Dict],
    segments_by_id: Dict[int, Dict],
    min_topic_duration_sec: float = 8.0
) -> List[Dict]:
    """
    ВАЖНО:
    Не объединяем любое пересечение, иначе из окон с overlap может снова получиться один длинный клип.
    Делаем так:
    - почти одинаковые темы удаляем/заменяем;
    - небольшие пересечения между соседними темами обрезаем по id.
    """
    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda x: (int(x["start_id"]), int(x["end_id"]))
    )

    selected: List[Dict] = []

    for cand in candidates:
        if int(cand["end_id"]) < int(cand["start_id"]):
            continue

        handled = False

        for i, existing in enumerate(selected):
            iou = topic_range_iou(cand, existing)
            same_start = abs(int(cand["start_id"]) - int(existing["start_id"])) <= 1
            same_end = abs(int(cand["end_id"]) - int(existing["end_id"])) <= 1

            if iou >= 0.60 or (same_start and same_end):
                selected[i] = choose_better_topic(existing, cand)
                handled = True
                break

        if not handled:
            selected.append(cand)

    resolved: List[Dict] = []

    for cand in sorted(selected, key=lambda x: int(x["start_id"])):
        cand = dict(cand)

        if not resolved:
            resolved.append(cand)
            continue

        last = resolved[-1]

        if int(cand["start_id"]) <= int(last["end_id"]):
            # Небольшой overlap из-за соседних окон. Не склеиваем темы, а обрезаем начало новой.
            new_start_id = int(last["end_id"]) + 1
            new_end_id = int(cand["end_id"])

            if new_start_id <= new_end_id:
                trimmed = build_topic_from_segment_ids(
                    segments_by_id=segments_by_id,
                    start_id=new_start_id,
                    end_id=new_end_id,
                    title=cand.get("title", ""),
                    summary=cand.get("summary", "")
                )

                if trimmed and float(trimmed["duration"]) >= min_topic_duration_sec:
                    resolved.append(trimmed)

            continue

        resolved.append(cand)

    return resolved


def split_long_topic_by_internal_llm(
    model,
    topic: Dict,
    segments_by_id: Dict[int, Dict],
    max_segments_per_window: int = 8,
    min_topic_duration_sec: float = 8.0,
    include_metadata: bool = True,
    max_tokens: Optional[int] = None
) -> List[Dict]:
    """
    Если LLM всё-таки вернула слишком длинную тему, пробуем повторно попросить
    модель разделить только этот фрагмент на более мелкие подтемы.
    Это семантический fallback, не fallback по паузам.
    """
    ids = [i for i in range(int(topic["start_id"]), int(topic["end_id"]) + 1) if i in segments_by_id]

    if len(ids) <= max_segments_per_window:
        return [topic]

    inner_segments = [segments_by_id[i] for i in ids]
    windows = split_segments_for_llm_by_count(
        inner_segments,
        max_segments_per_window=max_segments_per_window,
        overlap_segments=1
    )

    candidates = []

    for window in windows:
        raw_topics = ask_gpt4all_for_topics_in_window(
            model,
            window,
            include_metadata=include_metadata,
            max_tokens=max_tokens
        )

        for raw in raw_topics:
            subtopic = build_topic_from_segment_ids(
                segments_by_id=segments_by_id,
                start_id=int(raw["start_id"]),
                end_id=int(raw["end_id"]),
                title=raw.get("title", ""),
                summary=raw.get("summary", "")
            )

            if subtopic and float(subtopic["duration"]) >= min_topic_duration_sec:
                candidates.append(subtopic)

    if not candidates:
        return [topic]

    result = deduplicate_topic_candidates(
        candidates,
        segments_by_id=segments_by_id,
        min_topic_duration_sec=min_topic_duration_sec
    )

    return result if len(result) > 1 else [topic]


def score_topic_segment_v2(seg: Dict) -> float:
    """
    Оценка пригодности клипа.
    Она НЕ режет видео, а только сортирует уже найденные смысловые фрагменты.
    """
    text = clean_text_for_topic_analysis(seg.get("text", ""))
    duration = float(seg.get("duration", 0.0))

    words = text.split()
    word_count = len(words)
    words_per_sec = word_count / duration if duration > 0 else 0.0

    norm_text = normalize_text(text)

    strong_markers = [
        "важно", "главное", "интересно", "почему", "итог",
        "проблема", "решение", "результат", "суть", "вывод",
        "пример", "покажу", "объясню", "получается", "значит"
    ]

    weak_noise_markers = [
        "подписывайтесь", "ставьте лайк", "колокольчик",
        "всем привет", "ну короче", "как бы"
    ]

    if word_count >= 80:
        content_score = 0.30
    elif word_count >= 40:
        content_score = 0.22
    elif word_count >= 20:
        content_score = 0.14
    else:
        content_score = 0.06

    if 20 <= duration <= 180:
        duration_score = 0.25
    elif 10 <= duration < 20 or 180 < duration <= 360:
        duration_score = 0.16
    elif 5 <= duration < 10:
        duration_score = 0.08
    else:
        duration_score = 0.04

    density_score = min(words_per_sec / 2.8, 1.0) * 0.20

    marker_score = 0.0
    for marker in strong_markers:
        if marker in norm_text:
            marker_score += 0.035
    marker_score = min(marker_score, 0.18)

    noise_penalty = 0.0
    for marker in weak_noise_markers:
        if marker in norm_text:
            noise_penalty += 0.04
    noise_penalty = min(noise_penalty, 0.12)

    total = content_score + duration_score + density_score + marker_score - noise_penalty
    return round(max(0.0, min(total, 1.0)), 4)


def llm_topic_segmentation(
    whisper_segments: List[Dict],
    model=None,
    max_segments_per_window: int = TOPIC_MAX_SEGMENTS_PER_WINDOW,
    overlap_segments: int = TOPIC_OVERLAP_SEGMENTS,
    min_topic_duration_sec: float = 8.0,
    max_topic_duration_sec: float = 240.0,
    fast_mode: bool = False,
    generate_metadata: bool = True,
    topic_max_tokens: Optional[int] = None,
    progress_callback=None
) -> List[Dict]:
    """
    Основная логика:
    1. Берём ASR-сегменты faster-whisper.
    2. Даём каждому сегменту id.
    3. Отправляем маленькие окна в GPT4All.
    4. Модель возвращает start_id/end_id смысловых подтем.
    5. Код сопоставляет id с таймкодами.
    6. Убираем дубликаты/overlap.
    7. Сортируем темы по score для UI.

    Здесь НЕТ fallback по ASR-паузам как результата.
    Если LLM не справилась, возвращается [].
    """
    prepared_segments = prepare_whisper_segments_with_ids(whisper_segments)

    if not prepared_segments:
        return []

    if model is None:
        model = load_gpt4all_model(n_ctx=4096)

    segments_by_id = {int(seg["id"]): seg for seg in prepared_segments}

    windows = split_segments_for_llm_by_count(
        prepared_segments,
        max_segments_per_window=max_segments_per_window,
        overlap_segments=overlap_segments
    )

    candidates = []
    include_metadata = bool(generate_metadata)
    effective_max_tokens = int(
        topic_max_tokens or (GPT4ALL_TOPIC_MAX_TOKENS if include_metadata else GPT4ALL_FAST_TOPIC_MAX_TOKENS)
    )

    for window_index, window in enumerate(windows, start=1):
        if progress_callback:
            progress_callback(
                window_index,
                len(windows),
                f"ИИ выделяет подтемы: окно {window_index} из {len(windows)}"
            )

        raw_topics = ask_gpt4all_for_topics_in_window(
            model,
            window,
            include_metadata=include_metadata,
            max_tokens=effective_max_tokens
        )

        for raw in raw_topics:
            topic = build_topic_from_segment_ids(
                segments_by_id=segments_by_id,
                start_id=int(raw["start_id"]),
                end_id=int(raw["end_id"]),
                title=raw.get("title", ""),
                summary=raw.get("summary", "")
            )

            if not topic:
                continue

            if float(topic["duration"]) < min_topic_duration_sec:
                continue

            candidates.append(topic)

    topics = deduplicate_topic_candidates(
        candidates=candidates,
        segments_by_id=segments_by_id,
        min_topic_duration_sec=min_topic_duration_sec
    )

    print(f"[DEBUG] LLM windows: {len(windows)}")
    print(f"[DEBUG] LLM fast_mode: {fast_mode}, generate_metadata={include_metadata}, max_tokens={effective_max_tokens}")
    print(f"[DEBUG] Raw topic candidates: {len(candidates)}")
    print(f"[DEBUG] Topics after deduplication: {len(topics)}")

    if not topics:
        print("[WARN] LLM returned 0 topics. Semantic segmentation failed.")
        return []

    # Семантическая повторная попытка разделить слишком длинные темы.
    refined_topics = []

    for topic in topics:
        if float(topic.get("duration", 0.0)) > max_topic_duration_sec:
            refined_topics.extend(
                split_long_topic_by_internal_llm(
                    model=model,
                    topic=topic,
                    segments_by_id=segments_by_id,
                    max_segments_per_window=max_segments_per_window,
                    min_topic_duration_sec=min_topic_duration_sec,
                    include_metadata=include_metadata,
                    max_tokens=effective_max_tokens
                )
            )
        else:
            refined_topics.append(topic)

    topics = deduplicate_topic_candidates(
        candidates=refined_topics,
        segments_by_id=segments_by_id,
        min_topic_duration_sec=min_topic_duration_sec
    )

    topics = sorted(topics, key=lambda x: float(x["start"]))

    for i, topic in enumerate(topics, start=1):
        topic["id"] = i
        topic["score"] = score_topic_segment_v2(topic)

        print(
            f"[DEBUG] Topic {i}: "
            f"{format_timestamp(topic['start'])} - {format_timestamp(topic['end'])}, "
            f"duration={topic['duration']:.1f}s, "
            f"score={topic['score']:.2f}, "
            f"title={topic.get('title')}"
        )

    # Для UI показываем лучшие сверху.
    topics.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    return topics


def llm_chapter_segmentation(
    whisper_segments: List[Dict],
    model=None,
    target_block_duration_sec: int = CHAPTER_BLOCK_TARGET_SEC,
    max_block_text_chars: int = CHAPTER_BLOCK_MAX_CHARS,
    max_blocks_per_window: int = CHAPTER_MAX_BLOCKS_PER_WINDOW,
    overlap_blocks: int = CHAPTER_OVERLAP_BLOCKS,
    min_chapter_duration_sec: float = 20.0,
    generate_metadata: bool = False,
    max_tokens: Optional[int] = None,
    progress_callback=None
) -> List[Dict]:
    """
    Делит видео на крупные главы. В отличие от llm_topic_segmentation,
    сначала сжимает ASR в блоки, чтобы LLM видела структуру всего видео.
    """
    prepared_segments = prepare_whisper_segments_with_ids(whisper_segments)

    if not prepared_segments:
        return []

    if model is None:
        model = load_gpt4all_model(n_ctx=4096)

    blocks = build_chapter_blocks_from_segments(
        whisper_segments=whisper_segments,
        target_duration_sec=target_block_duration_sec,
        max_text_chars=max_block_text_chars
    )

    if not blocks:
        return []

    block_windows = split_segments_for_llm_by_count(
        blocks,
        max_segments_per_window=max_blocks_per_window,
        overlap_segments=overlap_blocks
    )

    segments_by_id = {int(seg["id"]): seg for seg in prepared_segments}
    blocks_by_id = {int(block["id"]): block for block in blocks}
    candidates: List[Dict] = []
    effective_max_tokens = int(max_tokens or CHAPTER_MAX_TOKENS)

    for window_index, block_window in enumerate(block_windows, start=1):
        if progress_callback:
            progress_callback(
                window_index,
                len(block_windows),
                f"ИИ выделяет главы: окно {window_index} из {len(block_windows)}"
            )

        raw_chapters = ask_gpt4all_for_chapters_in_block_window(
            model,
            block_window,
            include_metadata=generate_metadata,
            max_tokens=effective_max_tokens
        )

        for raw in raw_chapters:
            start_block = blocks_by_id.get(int(raw["start_block_id"]))
            end_block = blocks_by_id.get(int(raw["end_block_id"]))

            if not start_block or not end_block:
                continue

            chapter = build_topic_from_segment_ids(
                segments_by_id=segments_by_id,
                start_id=int(start_block["start_id"]),
                end_id=int(end_block["end_id"]),
                title=raw.get("title", ""),
                summary=raw.get("summary", "")
            )

            if not chapter:
                continue

            if float(chapter["duration"]) < min_chapter_duration_sec:
                continue

            chapter["segment_kind"] = "chapter"
            chapter["start_block_id"] = int(raw["start_block_id"])
            chapter["end_block_id"] = int(raw["end_block_id"])
            candidates.append(chapter)

    chapters = deduplicate_topic_candidates(
        candidates=candidates,
        segments_by_id=segments_by_id,
        min_topic_duration_sec=min_chapter_duration_sec
    )

    chapters = sorted(chapters, key=lambda x: float(x["start"]))

    print(f"[DEBUG] Chapter blocks: {len(blocks)}")
    print(f"[DEBUG] Chapter windows: {len(block_windows)}")
    print(f"[DEBUG] Raw chapter candidates: {len(candidates)}")
    print(f"[DEBUG] Chapters after deduplication: {len(chapters)}")

    for i, chapter in enumerate(chapters, start=1):
        chapter["id"] = i
        chapter["segment_kind"] = "chapter"
        chapter["score"] = score_topic_segment_v2(chapter)

        print(
            f"[DEBUG] Chapter {i}: "
            f"{format_timestamp(chapter['start'])} - {format_timestamp(chapter['end'])}, "
            f"duration={chapter['duration']:.1f}s, "
            f"title={chapter.get('title')}"
        )

    return chapters


# =========================
# SUBTITLES AND EXPORT
# =========================

def build_srt_for_clip(
    whisper_segments: List[Dict],
    clip_start: float,
    clip_end: float,
    srt_path: Path
):
    srt_blocks = []
    idx = 1

    for seg in whisper_segments:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        seg_text = seg.get("text", "").strip()

        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        local_start = max(seg_start, clip_start) - clip_start
        local_end = min(seg_end, clip_end) - clip_start

        if local_end <= local_start or not seg_text:
            continue

        srt_blocks.append(
            f"{idx}\n"
            f"{format_srt_timestamp(local_start)} --> {format_srt_timestamp(local_end)}\n"
            f"{seg_text}\n"
        )
        idx += 1

    srt_path.parent.mkdir(exist_ok=True, parents=True)
    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")


def build_subtitle_filter(srt_path: Path, position: str = "bottom", font_size: int = 28) -> str:
    srt_filter_path = str(srt_path).replace("\\", "/").replace(":", "\\:")

    if position == "center":
        alignment = 5
        margin_v = 0
    elif position == "upper_bottom":
        alignment = 2
        margin_v = 260
    else:
        alignment = 2
        margin_v = 80

    font_size = max(12, min(int(font_size), 72))
    force_style = (
        f"FontName=Arial,FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        f"Outline=2,Shadow=1,Alignment={alignment},MarginV={margin_v}"
    )

    return f"subtitles='{srt_filter_path}':force_style='{force_style}'"


def export_clip_with_optional_srt(
    video_path: Path,
    seg: Dict,
    out_dir: Path,
    srt_path: Optional[Path] = None,
    burn_subtitles: bool = False,
    aspect_mode: str = "original",
    subtitle_position: str = "bottom",
    subtitle_font_size: int = 28
) -> Path:
    """
    Экспортирует выбранный клип.
    По умолчанию SRT просто копируется рядом, не вшивается в видео.
    """
    out_dir.mkdir(exist_ok=True, parents=True)

    clip_id = int(seg.get("id", 0))
    title = safe_filename(seg.get("title") or f"clip_{clip_id}")

    start = max(0.0, float(seg["start"]))
    end = max(start, float(seg["end"]))
    duration = end - start

    if duration <= 0:
        raise RuntimeError("Некорректная длительность клипа.")

    variant_parts = []
    aspect_mode = (aspect_mode or "original").strip().lower()

    filters = []

    if aspect_mode == "vertical_9_16":
        variant_parts.append("9x16")
        filters.append("scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920")

    if burn_subtitles and srt_path is not None and srt_path.exists():
        variant_parts.append("subs")
        filters.append(build_subtitle_filter(srt_path, position=subtitle_position, font_size=subtitle_font_size))

    variant_suffix = f"_{'_'.join(variant_parts)}" if variant_parts else ""
    clip_path = out_dir / f"clip_{clip_id:03d}_{title}_{start:.2f}_{end:.2f}{variant_suffix}.mp4"

    vf_args = ["-vf", ",".join(filters)] if filters else []

    def cmd_with_encoder(encoder: str) -> List[str]:
        if encoder == "copy":
            return [
                FFMPEG_BIN,
                "-y",
                "-ss", str(start),
                "-i", str(video_path),
                "-t", str(duration),
                "-c", "copy",
                "-movflags", "+faststart",
                str(clip_path)
            ]

        if encoder == "h264_nvenc":
            return [
                FFMPEG_BIN,
                "-y",
                "-ss", str(start),
                "-i", str(video_path),
                "-t", str(duration),
                *vf_args,
                "-c:v", "h264_nvenc",
                "-preset", "p5",
                "-cq", "23",
                "-c:a", "aac",
                "-movflags", "+faststart",
                str(clip_path)
            ]

        return [
            FFMPEG_BIN,
            "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            *vf_args,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(clip_path)
        ]

    errors = []

    encoders = []
    if EXPORT_MODE == "copy" and not vf_args and not burn_subtitles:
        encoders.append("copy")

    if USE_NVENC_FOR_EXPORT:
        encoders.append("h264_nvenc")

    encoders.append("libx264")

    for encoder in encoders:
        cmd = cmd_with_encoder(encoder)
        code, _, err, _ = run_cmd(cmd)

        if code == 0:
            if srt_path is not None and srt_path.exists() and not burn_subtitles:
                copied_srt_path = clip_path.with_suffix(".srt")
                copied_srt_path.write_text(srt_path.read_text(encoding="utf-8"), encoding="utf-8")

            save_json(
                clip_path.with_suffix(".json"),
                {
                    "clip_path": str(clip_path),
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "title": seg.get("title"),
                    "summary": seg.get("summary"),
                    "text": seg.get("text"),
                    "score": seg.get("score"),
                    "encoder": encoder,
                    "srt_path": str(srt_path) if srt_path else None,
                    "burn_subtitles": burn_subtitles,
                    "aspect_mode": aspect_mode,
                    "subtitle_position": subtitle_position,
                    "subtitle_font_size": subtitle_font_size,
                }
            )

            return clip_path

        errors.append(f"{encoder}: {err}")

    raise RuntimeError("\n".join(errors) or "Ошибка FFmpeg при экспорте клипа.")


# =========================
# OPTIONAL CONSOLE TEST
# =========================

def main():
    video_path = Path(VIDEO_PATH)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    project_dir = OUTPUT_DIR / "console_run"
    asr_dir = project_dir / "asr" / "faster_whisper"
    topic_dir = project_dir / "topic_segmentation"

    media_duration = get_video_duration_sec(str(video_path))
    print(f"[INFO] Video duration: {media_duration:.2f} sec")

    def progress(p, message):
        print(f"[{p:.2f}] {message}")

    fw_segments = transcribe_with_faster_whisper(
        video_path=str(video_path),
        out_dir=asr_dir,
        progress_callback=progress
    )

    model = load_gpt4all_model(n_ctx=4096)

    topics = llm_topic_segmentation(
        whisper_segments=fw_segments,
        model=model,
        max_segments_per_window=TOPIC_MAX_SEGMENTS_PER_WINDOW,
        overlap_segments=TOPIC_OVERLAP_SEGMENTS,
        min_topic_duration_sec=8.0,
        max_topic_duration_sec=240.0
    )

    save_json(topic_dir / "topic_segments.json", topics)

    system_report = {**get_system_usage_snapshot(), **get_nvidia_gpu_snapshot()}
    save_json(project_dir / "system_usage_summary.json", system_report)

    print(f"[INFO] Saved topics: {topic_dir / 'topic_segments.json'}")


if __name__ == "__main__":
    main()
