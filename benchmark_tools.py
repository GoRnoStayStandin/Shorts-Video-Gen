print("RUNNING FILE:", __file__)

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =========================
# OPTIONAL IMPORTS
# =========================

try:
    import psutil
except ImportError:
    psutil = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    from gpt4all import GPT4All
except ImportError:
    GPT4All = None


# =========================
# CONFIG
# =========================

VIDEO_PATH = "data/sample.mp4"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

FASTER_WHISPER_MODEL = "medium"
USE_CUDA_FOR_WHISPER = False
GPT4ALL_MODEL_PATH = "C:/vs/Practice/models/mistral-7b-instruct-v0.1.Q4_0.gguf"

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"

LANGUAGE = "ru"

# CUDA для faster-whisper отключена принудительно, чтобы не требовался cublas64_12.dll.
USE_CUDA_FOR_ASR = False

# NVENC временно отключен для стабильности. Экспорт идёт через libx264.
USE_NVENC_FOR_EXPORT = False


# =========================
# BASIC HELPERS
# =========================

def run_cmd(cmd: List[str], capture_output: bool = True) -> Tuple[int, str, str, float]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )
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

def load_faster_whisper_model():
    """
    Загружает faster-whisper строго на CPU.

    В этой версии CUDA специально отключена, чтобы приложение не падало
    на Windows с ошибкой cublas64_12.dll. GPU можно будет включить позже,
    когда будут установлены CUDA/cuBLAS/cuDNN.
    """
    if WhisperModel is None:
        raise RuntimeError("faster-whisper не установлен")

    print("[INFO] Loading faster-whisper on CPU int8... CUDA is disabled in code.")
    return WhisperModel(
        FASTER_WHISPER_MODEL,
        device="cpu",
        compute_type="int8"
    )


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
    """
    out_dir.mkdir(exist_ok=True, parents=True)

    media_duration = get_video_duration_sec(str(video_path))

    wav_path = out_dir / "audio_for_faster_whisper.wav"
    txt_path = out_dir / "faster_whisper.txt"
    timed_txt_path = out_dir / "faster_whisper_timestamps.txt"
    segments_json_path = out_dir / "faster_whisper_segments.json"
    srt_path = out_dir / "faster_whisper.srt"

    if progress_callback:
        progress_callback(0.08, "Извлекаю аудио из видео...")

    extract_audio_wav(str(video_path), str(wav_path), sr=16000)

    if progress_callback:
        progress_callback(0.15, "Загружаю faster-whisper...")

    model = load_faster_whisper_model()

    if progress_callback:
        progress_callback(0.22, "Распознаю речь...")

    try:
        segments_iter, _info = model.transcribe(
            str(wav_path),
            language=LANGUAGE
        )
    except Exception as e:
        error_text = str(e).lower()

        if "cuda" in error_text or "cublas" in error_text or "cudnn" in error_text:
            print(f"[WARN] faster-whisper CUDA failed during transcription: {e}")
            print("[INFO] Retrying faster-whisper on CPU int8...")

            model = WhisperModel(
                FASTER_WHISPER_MODEL,
                device="cpu",
                compute_type="int8"
            )

            segments_iter, _info = model.transcribe(
                str(wav_path),
                language=LANGUAGE
            )
        else:
            raise

    transcript_parts = []
    timed_lines = []
    segments_json = []
    srt_blocks = []

    for idx, seg in enumerate(segments_iter, start=1):
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
            f"{idx}\n"
            f"{format_srt_timestamp(seg_start)} --> {format_srt_timestamp(seg_end)}\n"
            f"{seg_text}\n"
        )

        if progress_callback and media_duration > 0:
            local = min(seg_end / media_duration, 1.0)
            progress_callback(
                0.22 + local * 0.43,
                f"Распознаю речь: {format_timestamp(seg_end)} / {format_timestamp(media_duration)}"
            )

    txt_path.write_text(" ".join(transcript_parts).strip(), encoding="utf-8")
    timed_txt_path.write_text("\n".join(timed_lines), encoding="utf-8")
    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")
    save_json(segments_json_path, segments_json)

    return segments_json


# =========================
# GPT4ALL
# =========================

def load_gpt4all_model(n_ctx: int = 4096):
    """
    Загружает GPT4All.
    Важно: у некоторых версий gpt4all параметры n_ctx/device могут отличаться,
    поэтому есть несколько попыток.
    """
    if GPT4All is None:
        raise RuntimeError("gpt4all не установлен")

    if not Path(GPT4ALL_MODEL_PATH).exists():
        raise RuntimeError(f"Модель GPT4All не найдена: {GPT4ALL_MODEL_PATH}")

    attempts = [
        {
            "model_name": GPT4ALL_MODEL_PATH,
            "allow_download": False,
            "device": "cpu",
            "n_ctx": n_ctx
        },
        {
            "model_name": GPT4ALL_MODEL_PATH,
            "allow_download": False,
            "device": "cpu"
        },
        {
            "model_name": GPT4ALL_MODEL_PATH,
            "allow_download": False
        }
    ]

    last_error = None

    for kwargs in attempts:
        try:
            return GPT4All(**kwargs)
        except TypeError as e:
            last_error = e
            continue

    raise last_error


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


def ask_gpt4all_for_topics_in_window(model, window_segments: List[Dict]) -> List[Dict]:
    """
    Просит GPT4All выделить подтемы внутри маленького окна ASR-сегментов.
    Prompt короткий, чтобы помещаться даже в context window 2048.
    """
    segments_text = "\n".join(format_segment_for_llm(seg) for seg in window_segments)

    first_id = int(window_segments[0]["id"])
    last_id = int(window_segments[-1]["id"])

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

    try:
        with model.chat_session():
            answer = model.generate(prompt, max_tokens=300, temp=0.1)
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
    min_topic_duration_sec: float = 8.0
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
        raw_topics = ask_gpt4all_for_topics_in_window(model, window)

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
    max_segments_per_window: int = 8,
    overlap_segments: int = 2,
    min_topic_duration_sec: float = 8.0,
    max_topic_duration_sec: float = 240.0,
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

    for window_index, window in enumerate(windows, start=1):
        if progress_callback:
            progress_callback(
                window_index,
                len(windows),
                f"ИИ выделяет подтемы: окно {window_index} из {len(windows)}"
            )

        raw_topics = ask_gpt4all_for_topics_in_window(model, window)

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
                    min_topic_duration_sec=min_topic_duration_sec
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


def export_clip_with_optional_srt(
    video_path: Path,
    seg: Dict,
    out_dir: Path,
    srt_path: Optional[Path] = None,
    burn_subtitles: bool = False
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

    clip_path = out_dir / f"clip_{clip_id:03d}_{title}_{start:.2f}_{end:.2f}.mp4"

    vf_args = []
    if burn_subtitles and srt_path is not None and srt_path.exists():
        # Windows-friendly path for ffmpeg subtitles filter
        srt_filter_path = str(srt_path).replace("\\", "/").replace(":", "\\:")
        vf_args = ["-vf", f"subtitles='{srt_filter_path}'"]

    def cmd_with_encoder(encoder: str) -> List[str]:
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

    encoders = ["h264_nvenc", "libx264"] if USE_NVENC_FOR_EXPORT else ["libx264"]

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
                    "burn_subtitles": burn_subtitles
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
        max_segments_per_window=8,
        overlap_segments=2,
        min_topic_duration_sec=8.0,
        max_topic_duration_sec=240.0
    )

    save_json(topic_dir / "topic_segments.json", topics)

    system_report = {**get_system_usage_snapshot(), **get_nvidia_gpu_snapshot()}
    save_json(project_dir / "system_usage_summary.json", system_report)

    print(f"[INFO] Saved topics: {topic_dir / 'topic_segments.json'}")


if __name__ == "__main__":
    main()
