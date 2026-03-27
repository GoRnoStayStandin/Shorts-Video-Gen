print("RUNNING FILE:", __file__)
import os
import re
import json
import time
import math
import shutil
import subprocess
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd

# optional imports
try:
    import psutil
except ImportError:
    psutil = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    import wave
except ImportError:
    VoskModel = None
    KaldiRecognizer = None
    wave = None

try:
    from gpt4all import GPT4All
except ImportError:
    GPT4All = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None

try:
    import jiwer
except ImportError:
    jiwer = None

try:
    import torch
except ImportError:
    torch = None


# =========================
# CONFIG
# =========================

VIDEO_PATH = "data/sample.mp4"
REFERENCE_TRANSCRIPT = "data/reference_transcript.txt"      # optional
REFERENCE_SUMMARY = "data/reference_summary.txt"            # optional
REFERENCE_SEGMENTS = "data/reference_segments.json"         # optional

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

WHISPER_CPP_BIN = "whisper-cli"     # or full path, e.g. ./whisper.cpp/build/bin/whisper-cli
WHISPER_CPP_MODEL = "models/ggml-medium.bin"

FASTER_WHISPER_MODEL = "medium"

VOSK_MODEL_PATH = "models/vosk-model-small-ru-0.22"

GPT4ALL_MODEL_PATH = "C:/vs/Practice/models/mistral-7b-instruct-v0.1.Q4_0.gguf"

AUTO_EDITOR_BIN = "auto-editor"

FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"

SILERO_VAD_THRESHOLD = 0.5
SILERO_MIN_SPEECH_MS = 250

LANGUAGE = "ru"

# =========================
# HELPERS
# =========================

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
        code, out, err, elapsed = run_cmd(cmd)
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

def run_cmd(cmd: List[str], capture_output=True) -> Tuple[int, str, str, float]:
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


def file_exists(path: str) -> bool:
    return path is not None and Path(path).exists()


def read_text_if_exists(path: str) -> Optional[str]:
    if file_exists(path):
        return Path(path).read_text(encoding="utf-8").strip()
    return None


def read_json_if_exists(path: str):
    if file_exists(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return None


def save_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def get_video_duration_sec(video_path: str) -> float:
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    code, out, err, elapsed = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(f"ffprobe failed: {err}")
    return float(out.strip())


def extract_audio_wav(video_path: str, wav_path: str, sr: int = 16000):
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", video_path,
        "-ac", "1",
        "-ar", str(sr),
        "-vn",
        wav_path
    ]
    code, out, err, elapsed = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {err}")


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\sа-яё]", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def compute_asr_metrics(hyp: str, ref: Optional[str]) -> Dict:
    metrics = {}
    hyp_norm = normalize_text(hyp)

    metrics["hyp_chars"] = len(hyp_norm)
    metrics["hyp_words"] = len(hyp_norm.split())

    if ref is not None and jiwer is not None:
        ref_norm = normalize_text(ref)
        metrics["wer"] = jiwer.wer(ref_norm, hyp_norm)
        metrics["cer"] = jiwer.cer(ref_norm, hyp_norm)
    else:
        metrics["wer"] = None
        metrics["cer"] = None
    return metrics


def compute_rouge(summary: str, ref_summary: Optional[str]) -> Dict:
    if ref_summary is None or rouge_scorer is None:
        return {"rouge1_f": None, "rougeL_f": None}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    scores = scorer.score(ref_summary, summary)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure
    }


def intervals_total_length(intervals: List[Dict]) -> float:
    return sum(max(0.0, x["end"] - x["start"]) for x in intervals)


def interval_intersection(a: Dict, b: Dict) -> float:
    left = max(a["start"], b["start"])
    right = min(a["end"], b["end"])
    return max(0.0, right - left)


def segmentation_metrics(pred: List[Dict], ref: Optional[List[Dict]]) -> Dict:
    total_pred = intervals_total_length(pred)
    out = {
        "pred_segments_count": len(pred),
        "pred_total_sec": total_pred,
        "pred_avg_sec": (total_pred / len(pred)) if pred else 0.0
    }

    if ref is None:
        out.update({
            "seg_precision": None,
            "seg_recall": None,
            "seg_f1": None,
            "ref_total_sec": None,
            "intersection_sec": None
        })
        return out

    total_ref = intervals_total_length(ref)
    intersection = 0.0
    for p in pred:
        for r in ref:
            intersection += interval_intersection(p, r)

    precision = intersection / total_pred if total_pred > 0 else 0.0
    recall = intersection / total_ref if total_ref > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    out.update({
        "seg_precision": precision,
        "seg_recall": recall,
        "seg_f1": f1,
        "ref_total_sec": total_ref,
        "intersection_sec": intersection
    })
    return out


def rtf(elapsed_sec: float, media_duration_sec: float) -> float:
    return elapsed_sec / media_duration_sec if media_duration_sec > 0 else math.nan


def get_memory_mb() -> Optional[float]:
    if psutil is None:
        return None
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def export_video_clips(video_path: str, segments: List[Dict], out_dir: Path, prefix: str = "clip"):
    out_dir.mkdir(exist_ok=True, parents=True)

    exported = []
    for i, seg in enumerate(segments, start=1):
        start = max(0.0, float(seg["start"]))
        end = max(start, float(seg["end"]))
        duration = end - start

        if duration <= 0:
            continue

        clip_path = out_dir / f"{prefix}_{i:03d}_{start:.2f}_{end:.2f}.mp4"

        cmd = [
            FFMPEG_BIN,
            "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(clip_path)
        ]

        code, out, err, elapsed = run_cmd(cmd)
        if code == 0:
            exported.append({
                "clip_path": str(clip_path),
                "start": start,
                "end": end,
                "duration": duration
            })

    return exported

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

def group_whisper_segments_into_topics(
    segments: List[Dict],
    max_topic_duration_sec: float = 50.0,
    min_topic_duration_sec: float = 12.0,
    max_pause_between_segments_sec: float = 2.5
) -> List[Dict]:
    """
    Build topic-like blocks from faster-whisper segments.

    Split conditions:
    - long pause between neighboring ASR segments
    - current block becomes too long
    - next segment looks like a new topic by text markers
    """
    if not segments:
        return []

    topic_markers = [
        "теперь",
        "теперь о",
        "дальше",
        "далее",
        "следующая",
        "следующий",
        "следующее",
        "еще одна",
        "ещё одна",
        "еще один",
        "ещё один",
        "еще одно",
        "ещё одно",
        "кстати",
        "итак",
        "ну а теперь",
        "что касается",
        "перейдем к",
        "перейдём к",
        "отдельно",
        "с другой стороны",
        "а теперь",
        "следующая тема",
        "другая тема",
        "еще один момент",
        "ещё один момент",
    ]

    def looks_like_new_topic(text: str) -> bool:
        t = normalize_text(text)
        return any(t.startswith(marker) for marker in topic_markers)

    def flush_current(topics: List[Dict], current: Dict):
        text = " ".join(x for x in current["text_parts"] if x).strip()
        duration = current["end"] - current["start"]

        if duration >= min_topic_duration_sec and text:
            topics.append({
                "start": current["start"],
                "end": current["end"],
                "duration": duration,
                "text": text
            })

    segments = sorted(segments, key=lambda x: float(x["start"]))

    topics = []
    current = {
        "start": float(segments[0]["start"]),
        "end": float(segments[0]["end"]),
        "text_parts": [segments[0].get("text", "").strip()]
    }

    for seg in segments[1:]:
        seg_start = float(seg["start"])
        seg_end = float(seg["end"])
        seg_text = seg.get("text", "").strip()

        pause = seg_start - current["end"]
        new_duration = seg_end - current["start"]

        split_by_pause = pause > max_pause_between_segments_sec
        split_by_length = new_duration > max_topic_duration_sec
        split_by_marker = looks_like_new_topic(seg_text)

        should_split = split_by_pause or split_by_length or split_by_marker

        if should_split:
            flush_current(topics, current)
            current = {
                "start": seg_start,
                "end": seg_end,
                "text_parts": [seg_text]
            }
        else:
            current["end"] = seg_end
            if seg_text:
                current["text_parts"].append(seg_text)

    flush_current(topics, current)

    return topics

def make_topic_label(text: str, max_words: int = 8) -> str:
    text = normalize_text(text)
    words = text.split()
    if not words:
        return "topic"
    return "_".join(words[:max_words])

def export_topic_video_clips(video_path: str, topic_segments: List[Dict], out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)

    exported = []

    for i, seg in enumerate(topic_segments, start=1):
        start = max(0.0, float(seg["start"]))
        end = max(start, float(seg["end"]))
        duration = end - start

        if duration <= 0:
            continue

        label = make_topic_label(seg.get("text", "topic"))
        clip_path = out_dir / f"topic_{i:03d}_{start:.2f}_{end:.2f}_{label}.mp4"

        cmd = [
            FFMPEG_BIN,
            "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(clip_path)
        ]

        code, out, err, elapsed = run_cmd(cmd)
        if code == 0:
            exported.append({
                "clip_path": str(clip_path),
                "start": start,
                "end": end,
                "duration": duration,
                "text": seg.get("text", "")
            })

    return exported

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

        # пропускаем сегменты вне клипа
        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        # обрезаем сегмент по границам клипа
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

    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")

def export_topic_video_clips_with_subtitles(
    video_path: str,
    topic_segments: List[Dict],
    whisper_segments: List[Dict],
    out_dir: Path
):
    out_dir.mkdir(exist_ok=True, parents=True)
    exported = []

    for i, seg in enumerate(topic_segments, start=1):
        start = max(0.0, float(seg["start"]))
        end = max(start, float(seg["end"]))
        duration = end - start

        if duration <= 0:
            continue

        label = make_topic_label(seg.get("text", "topic"))
        base_name = f"topic_{i:03d}_{start:.2f}_{end:.2f}_{label}"

        clip_path = out_dir / f"{base_name}.mp4"
        srt_path = out_dir / f"{base_name}.srt"

        cmd = [
            FFMPEG_BIN,
            "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-movflags", "+faststart",
            str(clip_path)
        ]

        code, out, err, elapsed = run_cmd(cmd)
        if code == 0:
            build_srt_for_clip(
                whisper_segments=whisper_segments,
                clip_start=start,
                clip_end=end,
                srt_path=srt_path
            )

            exported.append({
                "clip_path": str(clip_path),
                "srt_path": str(srt_path),
                "start": start,
                "end": end,
                "duration": duration,
                "text": seg.get("text", "")
            })

    return exported

def clean_text_for_topic_analysis(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text

def score_topic_segment(seg: Dict) -> float:
    text = clean_text_for_topic_analysis(seg.get("text", ""))
    duration = float(seg.get("duration", 0.0))

    words = text.split()
    word_count = len(words)

    # плотность речи
    words_per_sec = word_count / duration if duration > 0 else 0.0

    # бонус за слова, часто встречающиеся в содержательных фрагментах
    strong_markers = [
        "важно", "главное", "интересно", "новость", "новости",
        "почему", "итог", "проблема", "решение", "результат",
        "суть", "кстати", "теперь", "план", "ошибка", "вывод"
    ]

    marker_bonus = 0.0
    norm_text = normalize_text(text)
    for marker in strong_markers:
        if marker in norm_text:
            marker_bonus += 0.05

    # предпочтительная длина для shorts-кандидата
    duration_score = 0.0
    if 15 <= duration <= 60:
        duration_score = 0.4
    elif 10 <= duration < 15 or 60 < duration <= 75:
        duration_score = 0.25
    elif 7 <= duration < 10 or 75 < duration <= 90:
        duration_score = 0.1

    density_score = min(words_per_sec / 2.5, 1.0) * 0.4
    marker_score = min(marker_bonus, 0.2)

    total = duration_score + density_score + marker_score
    return round(min(total, 1.0), 4)

# =========================
# ASR TOOLS
# =========================

def run_whisper_cpp(video_path: str, out_dir: Path, ref_transcript: Optional[str], media_duration: float) -> Dict:
    wav_path = out_dir / "audio_for_whispercpp.wav"
    txt_path = out_dir / "whispercpp.txt"

    extract_audio_wav(video_path, str(wav_path))

    cmd = [
        WHISPER_CPP_BIN,
        "-m", WHISPER_CPP_MODEL,
        "-f", str(wav_path),
        "-l", LANGUAGE,
        "-otxt",
        "-of", str(out_dir / "whispercpp")
    ]

    mem_before = get_memory_mb()
    code, out, err, elapsed = run_cmd(cmd)
    mem_after = get_memory_mb()

    if code != 0:
        return {
            "tool": "whisper.cpp",
            "status": "failed",
            "error": err
        }

    transcript = txt_path.read_text(encoding="utf-8", errors="ignore") if txt_path.exists() else ""
    metrics = compute_asr_metrics(transcript, ref_transcript)

    return {
        "tool": "whisper.cpp",
        "status": "ok",
        "time_sec": elapsed,
        "rtf": rtf(elapsed, media_duration),
        "memory_mb_before": mem_before,
        "memory_mb_after": mem_after,
        "transcript_path": str(txt_path),
        **metrics
    }


def run_faster_whisper(video_path: str, out_dir: Path, ref_transcript: Optional[str], media_duration: float) -> Dict:
    if WhisperModel is None:
        return {
            "tool": "faster-whisper",
            "status": "skipped",
            "error": "faster_whisper not installed"
        }

    wav_path = out_dir / "audio_for_faster_whisper.wav"
    txt_path = out_dir / "faster_whisper.txt"
    timed_txt_path = out_dir / "faster_whisper_timestamps.txt"
    segments_json_path = out_dir / "faster_whisper_segments.json"

    extract_audio_wav(video_path, str(wav_path))

    mem_before = get_memory_mb()
    start = time.perf_counter()

    model = WhisperModel(FASTER_WHISPER_MODEL, device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(wav_path), language=LANGUAGE)
    segments = list(segments)

    elapsed = time.perf_counter() - start
    mem_after = get_memory_mb()

    transcript = " ".join(seg.text.strip() for seg in segments).strip()
    txt_path.write_text(transcript, encoding="utf-8")

    timed_lines = []
    segments_json = []

    for seg in segments:
        seg_start = float(seg.start)
        seg_end = float(seg.end)
        seg_text = seg.text.strip()

        timed_lines.append(f"{format_timestamp(seg_start)} - {format_timestamp(seg_end)} {seg_text}")
        segments_json.append({
            "start": seg_start,
            "end": seg_end,
            "text": seg_text
        })

    timed_txt_path.write_text("\n".join(timed_lines), encoding="utf-8")
    save_json(segments_json_path, segments_json)

    metrics = compute_asr_metrics(transcript, ref_transcript)

    srt_path = out_dir / "faster_whisper.srt"
    srt_blocks = []

    for idx, seg in enumerate(segments, start=1):
        seg_start = float(seg.start)
        seg_end = float(seg.end)
        seg_text = seg.text.strip()

        srt_blocks.append(
            f"{idx}\n"
            f"{format_srt_timestamp(seg_start)} --> {format_srt_timestamp(seg_end)}\n"
            f"{seg_text}\n"
        )

    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")

    return {
        "tool": "faster-whisper",
        "status": "ok",
        "time_sec": elapsed,
        "rtf": rtf(elapsed, media_duration),
        "memory_mb_before": mem_before,
        "memory_mb_after": mem_after,
        "segments_count": len(segments),
        "transcript_path": str(txt_path),
        "timed_transcript_path": str(timed_txt_path),
        "segments_json_path": str(segments_json_path),
        "srt_path": str(srt_path),
        **metrics
    }


def run_vosk(video_path: str, out_dir: Path, ref_transcript: Optional[str], media_duration: float) -> Dict:
    if VoskModel is None or KaldiRecognizer is None or wave is None:
        return {
            "tool": "vosk",
            "status": "skipped",
            "error": "vosk not installed"
        }

    if not Path(VOSK_MODEL_PATH).exists():
        return {
            "tool": "vosk",
            "status": "skipped",
            "error": f"model not found: {VOSK_MODEL_PATH}"
        }

    wav_path = out_dir / "audio_for_vosk.wav"
    txt_path = out_dir / "vosk.txt"

    extract_audio_wav(video_path, str(wav_path), sr=16000)

    with wave.open(str(wav_path), "rb") as wf:
        model = VoskModel(VOSK_MODEL_PATH)
        rec = KaldiRecognizer(model, wf.getframerate())

        mem_before = get_memory_mb()
        start = time.perf_counter()

        texts = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                texts.append(result.get("text", ""))

        final_result = json.loads(rec.FinalResult())
        texts.append(final_result.get("text", ""))

    elapsed = time.perf_counter() - start
    mem_after = get_memory_mb()

    transcript = " ".join(t for t in texts if t).strip()
    txt_path.write_text(transcript, encoding="utf-8")

    metrics = compute_asr_metrics(transcript, ref_transcript)

    return {
        "tool": "vosk",
        "status": "ok",
        "time_sec": elapsed,
        "rtf": rtf(elapsed, media_duration),
        "memory_mb_before": mem_before,
        "memory_mb_after": mem_after,
        "transcript_path": str(txt_path),
        **metrics
    }

def burn_subtitles_into_video(input_video: Path, input_srt: Path, output_video: Path):
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-i", str(input_video),
        "-vf", f"subtitles={str(input_srt).replace('\\', '/')}",
        "-c:v", "libx264",
        "-c:a", "aac",
        str(output_video)
    ]
    return run_cmd(cmd)

def enrich_topic_segments(topic_segments: List[Dict]) -> List[Dict]:
    enriched = []

    for seg in topic_segments:
        analysis = analyze_topic_with_gpt4all(seg.get("text", ""))
        score = score_topic_segment(seg)

        new_seg = dict(seg)
        new_seg["title"] = analysis.get("title")
        new_seg["summary"] = analysis.get("summary")
        new_seg["score"] = score
        if analysis.get("error"):
            new_seg["analysis_error"] = analysis.get("error")

        enriched.append(new_seg)

    return enriched

def select_best_topics_for_shorts(topic_segments: List[Dict], top_k: int = 3) -> List[Dict]:
    scored = [seg for seg in topic_segments if seg.get("score") is not None]
    scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return scored[:top_k]

# =========================
# SEGMENTATION / HIGHLIGHT TOOLS
# =========================

def parse_ffmpeg_silencedetect(stderr_text: str, media_duration: float) -> List[Dict]:
    silence_starts = []
    silence_ends = []

    for line in stderr_text.splitlines():
        m1 = re.search(r"silence_start:\s*([0-9.]+)", line)
        m2 = re.search(r"silence_end:\s*([0-9.]+)", line)
        if m1:
            silence_starts.append(float(m1.group(1)))
        if m2:
            silence_ends.append(float(m2.group(1)))

    silence_intervals = []
    for s, e in zip(silence_starts, silence_ends):
        if e > s:
            silence_intervals.append({"start": s, "end": e})

    # Convert silence intervals to non-silence intervals
    nonsilent = []
    cur = 0.0
    for sil in silence_intervals:
        if sil["start"] > cur:
            nonsilent.append({"start": cur, "end": sil["start"]})
        cur = sil["end"]
    if cur < media_duration:
        nonsilent.append({"start": cur, "end": media_duration})

    # remove tiny segments
    nonsilent = [x for x in nonsilent if x["end"] - x["start"] >= 1.0]
    return nonsilent


def run_ffmpeg_silencedetect(video_path: str, out_dir: Path, ref_segments, media_duration: float) -> Dict:
    cmd = [
        FFMPEG_BIN,
        "-i", video_path,
        "-af", "silencedetect=n=-35dB:d=0.5",
        "-f", "null",
        "-"
    ]

    mem_before = get_memory_mb()
    code, out, err, elapsed = run_cmd(cmd)
    mem_after = get_memory_mb()

    if code != 0 and "silence_" not in err:
        return {
            "tool": "ffmpeg-silencedetect",
            "status": "failed",
            "error": err
        }

    pred_segments = parse_ffmpeg_silencedetect(err, media_duration)
    save_json(out_dir / "ffmpeg_silencedetect_segments.json", pred_segments)

    seg_metrics = segmentation_metrics(pred_segments, ref_segments)

    return {
        "tool": "ffmpeg-silencedetect",
        "status": "ok",
        "time_sec": elapsed,
        "rtf": rtf(elapsed, media_duration),
        "memory_mb_before": mem_before,
        "memory_mb_after": mem_after,
        "segments_path": str(out_dir / "ffmpeg_silencedetect_segments.json"),
        **seg_metrics
    }

def parse_auto_editor_v1_timeline(timeline_json: Dict) -> List[Dict]:
    """
    Parse auto-editor timeline v1 JSON and return kept segments in seconds.
    Keeps only chunks with speed > 0.
    """
    chunks = timeline_json.get("chunks", [])
    if not isinstance(chunks, list):
        return []

    # v1 uses implicit timebase = 1 / average_framerate of the source.
    # We can get fps from ffprobe.
    source = timeline_json.get("source", VIDEO_PATH)

    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        source
    ]
    code, out, err, _ = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(f"ffprobe avg_frame_rate failed: {err}")

    fps_text = out.strip()
    if "/" in fps_text:
        num, den = fps_text.split("/")
        fps = float(num) / float(den)
    else:
        fps = float(fps_text)

    if fps <= 0:
        raise RuntimeError(f"Invalid fps from ffprobe: {fps_text}")

    pred_segments = []
    for chunk in chunks:
        if not isinstance(chunk, list) or len(chunk) != 3:
            continue

        start_units, end_units, speed = chunk

        try:
            start_units = float(start_units)
            end_units = float(end_units)
            speed = float(speed)
        except Exception:
            continue

        # Keep only non-cut chunks
        if speed > 0 and end_units > start_units:
            pred_segments.append({
                "start": start_units / fps,
                "end": end_units / fps
            })

    return pred_segments

def run_auto_editor(video_path: str, out_dir: Path, ref_segments, media_duration: float) -> Dict:
    preview_txt = out_dir / "auto_editor_preview.txt"
    segments_json_path = out_dir / "auto_editor_segments.json"

    cmd = [
        AUTO_EDITOR_BIN,
        video_path,
        "--no-open",
        "--preview"
    ]

    mem_before = get_memory_mb()
    code, out, err, elapsed = run_cmd(cmd)
    mem_after = get_memory_mb()

    preview_txt.write_text(out + "\n" + err, encoding="utf-8")

    if code != 0:
        return {
            "tool": "auto-editor",
            "status": "failed",
            "error": err
        }

    pred_segments = []

    for line in (out + "\n" + err).splitlines():
        m = re.findall(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", line)
        for a, b in m:
            try:
                a = float(a)
                b = float(b)
                if b > a:
                    pred_segments.append({"start": a, "end": b})
            except:
                continue

    # удаляем дубликаты
    unique = []
    seen = set()
    for seg in pred_segments:
        key = (round(seg["start"], 2), round(seg["end"], 2))
        if key not in seen:
            seen.add(key)
            unique.append(seg)

    pred_segments = unique

    if not pred_segments:
        error_msg = "No segments parsed from auto-editor preview output"
    else:
        error_msg = None

    save_json(segments_json_path, pred_segments)
    seg_metrics = segmentation_metrics(pred_segments, ref_segments)

    return {
        "tool": "auto-editor",
        "status": "ok" if pred_segments else "ok",  
        "time_sec": elapsed,
        "rtf": rtf(elapsed, media_duration),
        "memory_mb_before": mem_before,
        "memory_mb_after": mem_after,
        "segments_path": str(segments_json_path),
        "parsed_segments": len(pred_segments),
        "error": error_msg,
        **seg_metrics
    }

def run_silero_vad(video_path: str, out_dir: Path, ref_segments, media_duration: float) -> Dict:
    if torch is None:
        return {
            "tool": "silero-vad",
            "status": "skipped",
            "error": "torch not installed"
        }

    wav_path = out_dir / "audio_for_silero.wav"
    extract_audio_wav(video_path, str(wav_path), sr=16000)

    try:
        mem_before = get_memory_mb()
        start = time.perf_counter()

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True
        )

        get_speech_timestamps = utils[0]

        audio, sr = sf.read(str(wav_path))
        if sr != 16000:
            return {
                "tool": "silero-vad",
                "status": "failed",
                "error": f"unexpected sample rate: {sr}"
            }

        # mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        wav = torch.tensor(audio, dtype=torch.float32)

        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            threshold=SILERO_VAD_THRESHOLD,
            sampling_rate=16000,
            min_speech_duration_ms=SILERO_MIN_SPEECH_MS
        )

        elapsed = time.perf_counter() - start
        mem_after = get_memory_mb()

        pred_segments = [
            {
                "start": x["start"] / 16000.0,
                "end": x["end"] / 16000.0
            }
            for x in speech_timestamps
        ]

        save_json(out_dir / "silero_vad_segments.json", pred_segments)
        seg_metrics = segmentation_metrics(pred_segments, ref_segments)

        return {
            "tool": "silero-vad",
            "status": "ok",
            "time_sec": elapsed,
            "rtf": rtf(elapsed, media_duration),
            "memory_mb_before": mem_before,
            "memory_mb_after": mem_after,
            "segments_path": str(out_dir / "silero_vad_segments.json"),
            **seg_metrics
        }

    except Exception as e:
        return {
            "tool": "silero-vad",
            "status": "failed",
            "error": str(e)
        }


# =========================
# SUMMARIZATION / TEXT ANALYSIS
# =========================

def simple_textrank_baseline(transcript: str, max_sentences: int = 5) -> str:
    # very simple sentence scoring baseline
    sentences = re.split(r"(?<=[.!?])\s+", transcript.strip())
    if len(sentences) <= max_sentences:
        return transcript.strip()

    word_freq = {}
    for sent in sentences:
        for w in normalize_text(sent).split():
            word_freq[w] = word_freq.get(w, 0) + 1

    scored = []
    for sent in sentences:
        words = normalize_text(sent).split()
        score = sum(word_freq.get(w, 0) for w in words) / max(1, len(words))
        scored.append((sent, score))

    top = sorted(scored, key=lambda x: x[1], reverse=True)[:max_sentences]
    selected = [x[0] for x in top]

    # preserve original order
    ordered = [s for s in sentences if s in selected]
    return " ".join(ordered).strip()

def clean_transcript_for_summary(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def run_gpt4all_summary(transcript: str, out_dir: Path, ref_summary: Optional[str]) -> Dict:
    if GPT4All is None:
        return {
            "tool": "gpt4all",
            "status": "skipped",
            "error": "gpt4all not installed"
        }

    if not Path(GPT4ALL_MODEL_PATH).exists():
        return {
            "tool": "gpt4all",
            "status": "skipped",
            "error": f"model not found: {GPT4ALL_MODEL_PATH}"
        }

    transcript = clean_transcript_for_summary(transcript)
    transcript = transcript[:6000]

    prompt = f"""
    Ты делаешь краткое резюме транскрипта видео на русском языке.

    Правила:
    - Не добавляй факты, которых нет в транскрипте.
    - Не выдумывай имена, события или выводы.
    - Убирай повторы, слова-паразиты и ошибки распознавания.
    - Не используй списки, заголовки и маркированные пункты.
    - Напиши связный краткий текст на 5-7 предложений.
    - Если какой-то фрагмент транскрипта неясен, пропусти его, а не додумывай.

    Транскрипт:
    {transcript}
    """.strip()

    mem_before = get_memory_mb()
    start = time.perf_counter()

    try:
        model = GPT4All(model_name=GPT4ALL_MODEL_PATH, allow_download=False)
        with model.chat_session():
            summary = model.generate(prompt, max_tokens=300, temp=0.2)

        elapsed = time.perf_counter() - start
        mem_after = get_memory_mb()

        summary_path = out_dir / "gpt4all_summary.txt"
        summary_path.write_text(summary, encoding="utf-8")

        rouge = compute_rouge(summary, ref_summary)

        return {
            "tool": "gpt4all",
            "status": "ok",
            "time_sec": elapsed,
            "memory_mb_before": mem_before,
            "memory_mb_after": mem_after,
            "summary_chars": len(summary),
            "summary_words": len(summary.split()),
            "summary_path": str(summary_path),
            **rouge
        }

    except Exception as e:
        return {
            "tool": "gpt4all",
            "status": "failed",
            "error": str(e)
        }


def run_textrank_summary(transcript: str, out_dir: Path, ref_summary: Optional[str]) -> Dict:
    start = time.perf_counter()
    summary = simple_textrank_baseline(transcript, max_sentences=5)
    elapsed = time.perf_counter() - start

    summary_path = out_dir / "textrank_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    rouge = compute_rouge(summary, ref_summary)

    return {
        "tool": "textrank-baseline",
        "status": "ok",
        "time_sec": elapsed,
        "summary_chars": len(summary),
        "summary_words": len(summary.split()),
        "summary_path": str(summary_path),
        **rouge
    }

def analyze_topic_with_gpt4all(text: str) -> Dict:
    if GPT4All is None:
        return {"title": None, "summary": None, "error": "gpt4all not installed"}

    if not Path(GPT4ALL_MODEL_PATH).exists():
        return {"title": None, "summary": None, "error": f"model not found: {GPT4ALL_MODEL_PATH}"}

    text = clean_text_for_topic_analysis(text)
    text = text[:3000]

    prompt = f"""
Проанализируй фрагмент транскрипта видео.

ВАЖНО:
- Отвечай ТОЛЬКО на русском языке.
- НЕ используй английский язык.

Пример ответа:
TITLE: Планы на будущее
SUMMARY: Автор рассказывает о своих планах и будущих проектах.

Теперь сделай для текста ниже.

Формат:
TITLE: ...
SUMMARY: ...

Текст:
{text}
""".strip()

    try:
        model = GPT4All(model_name=GPT4ALL_MODEL_PATH, allow_download=False)
        with model.chat_session():
            answer = model.generate(prompt, max_tokens=120, temp=0.2)

        title = None
        summary = None

        for line in answer.splitlines():
            line = line.strip()
            if line.upper().startswith("TITLE:"):
                title = line.split(":", 1)[1].strip()
            elif line.upper().startswith("SUMMARY:"):
                summary = line.split(":", 1)[1].strip()

        # fallback
        if not title:
            title = make_topic_label(text, max_words=6).replace("_", " ")
        if not summary:
            summary = answer.strip()

        return {
            "title": title,
            "summary": summary,
            "error": None
        }

    except Exception as e:
        return {
            "title": make_topic_label(text, max_words=6).replace("_", " "),
            "summary": None,
            "error": str(e)
        }

# =========================
# MAIN
# =========================

def main():
    video_path = VIDEO_PATH
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ref_transcript = read_text_if_exists(REFERENCE_TRANSCRIPT)
    ref_summary = read_text_if_exists(REFERENCE_SUMMARY)
    ref_segments = read_json_if_exists(REFERENCE_SEGMENTS)

    media_duration = get_video_duration_sec(video_path)
    print(f"[INFO] Video duration: {media_duration:.2f} sec")

    # ---------- ASR ----------
    asr_dir = OUTPUT_DIR / "asr"
    asr_dir.mkdir(exist_ok=True, parents=True)

    asr_results = []

    # whispercpp_dir = asr_dir / "whisper_cpp"
    # whispercpp_dir.mkdir(exist_ok=True, parents=True)
    # asr_results.append(run_whisper_cpp(video_path, whispercpp_dir, ref_transcript, media_duration))

    fw_dir = asr_dir / "faster_whisper"
    fw_dir.mkdir(exist_ok=True, parents=True)
    asr_results.append(run_faster_whisper(video_path, fw_dir, ref_transcript, media_duration))

    vosk_dir = asr_dir / "vosk"
    vosk_dir.mkdir(exist_ok=True, parents=True)
    asr_results.append(run_vosk(video_path, vosk_dir, ref_transcript, media_duration))

    asr_df = pd.DataFrame(asr_results)
    asr_df.to_csv(OUTPUT_DIR / "asr_benchmark.csv", index=False, encoding="utf-8-sig")

    # pick best transcript for summary stage:
    # priority by WER if available, else by lowest RTF among successful tools
    
    # successful_asr = [x for x in asr_results if x.get("status") == "ok" and x.get("transcript_path")]
    # chosen_transcript = None
    # if successful_asr:
    #     if any(x.get("wer") is not None for x in successful_asr):
    #         successful_asr.sort(key=lambda x: x["wer"] if x.get("wer") is not None else 999.0)
    #     else:
    #         successful_asr.sort(key=lambda x: x.get("rtf", 999.0))
    #     chosen_transcript = Path(successful_asr[0]["transcript_path"]).read_text(encoding="utf-8", errors="ignore")
    # else:
    #     chosen_transcript = ""

    successful_asr = [x for x in asr_results if x.get("status") == "ok" and x.get("transcript_path")]
    chosen_transcript = ""

    if successful_asr:
        preferred_order = ["faster-whisper", "whisper.cpp", "vosk"]
        successful_asr.sort(
            key=lambda x: preferred_order.index(x["tool"]) if x["tool"] in preferred_order else 999
        )
        chosen_transcript = Path(successful_asr[0]["transcript_path"]).read_text(
            encoding="utf-8",
            errors="ignore"
        )

    # ---------- SEGMENTATION ----------
    seg_dir = OUTPUT_DIR / "segmentation"
    seg_dir.mkdir(exist_ok=True, parents=True)

    seg_results = []

    ffmpeg_dir = seg_dir / "ffmpeg_silencedetect"
    ffmpeg_dir.mkdir(exist_ok=True, parents=True)
    seg_results.append(run_ffmpeg_silencedetect(video_path, ffmpeg_dir, ref_segments, media_duration))

    auto_dir = seg_dir / "auto_editor"
    auto_dir.mkdir(exist_ok=True, parents=True)
    seg_results.append(run_auto_editor(video_path, auto_dir, ref_segments, media_duration))

    silero_dir = seg_dir / "silero_vad"
    silero_dir.mkdir(exist_ok=True, parents=True)

    silero_result = run_silero_vad(video_path, silero_dir, ref_segments, media_duration)
    seg_results.append(silero_result)

    if silero_result.get("status") == "ok" and silero_result.get("segments_path"):
        silero_segments = read_json_if_exists(silero_result["segments_path"])
        if silero_segments:
            clips_dir = seg_dir / "silero_vad_clips"
            clips_info = export_video_clips(
                video_path,
                silero_segments,
                clips_dir,
                prefix="silero"
            )
            save_json(clips_dir / "clips_manifest.json", clips_info)

    seg_df = pd.DataFrame(seg_results)
    seg_df.to_csv(OUTPUT_DIR / "segmentation_benchmark.csv", index=False, encoding="utf-8-sig")

        # ---------- TOPIC SEGMENTATION FROM FASTER-WHISPER ----------
    # ---------- TOPIC SEGMENTATION FROM FASTER-WHISPER ----------
    topic_dir = OUTPUT_DIR / "topic_segmentation"
    topic_dir.mkdir(exist_ok=True, parents=True)

    fw_segments_json_path = fw_dir / "faster_whisper_segments.json"

    if fw_segments_json_path.exists():
        fw_segments = read_json_if_exists(str(fw_segments_json_path))

        if fw_segments:
            topic_segments = group_whisper_segments_into_topics(
                fw_segments,
                max_topic_duration_sec=40.0,
                min_topic_duration_sec=10.0,
                max_pause_between_segments_sec=2.5
            )

            save_json(topic_dir / "topic_segments_raw.json", topic_segments)

            enriched_topic_segments = enrich_topic_segments(topic_segments)
            save_json(topic_dir / "topic_segments.json", enriched_topic_segments)

            best_topics = select_best_topics_for_shorts(enriched_topic_segments, top_k=3)
            save_json(topic_dir / "best_topics.json", best_topics)

            topic_clips = export_topic_video_clips_with_subtitles(
                video_path,
                enriched_topic_segments,
                fw_segments,
                topic_dir / "topic_clips"
            )
            save_json(topic_dir / "topic_clips" / "topic_clips_manifest.json", topic_clips)

            best_topic_clips = export_topic_video_clips_with_subtitles(
                video_path,
                best_topics,
                fw_segments,
                topic_dir / "best_topic_clips"
            )
            save_json(topic_dir / "best_topic_clips" / "best_topic_clips_manifest.json", best_topic_clips)

    # ---------- SUMMARIZATION ----------
    sum_dir = OUTPUT_DIR / "summary"
    sum_dir.mkdir(exist_ok=True, parents=True)

    summary_results = []

    if chosen_transcript.strip():
        gpt4all_dir = sum_dir / "gpt4all"
        gpt4all_dir.mkdir(exist_ok=True, parents=True)
        summary_results.append(run_gpt4all_summary(chosen_transcript, gpt4all_dir, ref_summary))

        textrank_dir = sum_dir / "textrank"
        textrank_dir.mkdir(exist_ok=True, parents=True)
        summary_results.append(run_textrank_summary(chosen_transcript, textrank_dir, ref_summary))
    else:
        summary_results.append({
            "tool": "summary-stage",
            "status": "skipped",
            "error": "No transcript available from ASR stage"
        })

    sum_df = pd.DataFrame(summary_results)
    sum_df.to_csv(OUTPUT_DIR / "summary_benchmark.csv", index=False, encoding="utf-8-sig")

    # ---------- OVERALL RANKING ----------
    rows = []

    for row in asr_results:
        if row.get("status") == "ok":
            score = 0.0
            if row.get("wer") is not None:
                score += (1.0 - row["wer"]) * 0.6
            if row.get("rtf") is not None:
                score += max(0.0, 1.0 - min(row["rtf"], 1.0)) * 0.4
            rows.append({
                "category": "ASR",
                "tool": row["tool"],
                "score": round(score, 4)
            })

    for row in seg_results:
        if row.get("status") == "ok":
            if row.get("parsed_segments") == 0:
                score = 0.0
            else:
                score = 0.0
                if row.get("seg_f1") is not None:
                    score += row["seg_f1"] * 0.7
                if row.get("rtf") is not None:
                    score += max(0.0, 1.0 - min(row["rtf"], 1.0)) * 0.3

            rows.append({
                "category": "Segmentation",
                "tool": row["tool"],
                "score": round(score, 4)
            })

    for row in summary_results:
        if row.get("status") == "ok":
            score = 0.0
            if row.get("rougeL_f") is not None:
                score += row["rougeL_f"] * 0.7
            if row.get("time_sec") is not None:
                score += max(0.0, 1.0 - min(row["time_sec"] / 60.0, 1.0)) * 0.3
            rows.append({
                "category": "Summary",
                "tool": row["tool"],
                "score": round(score, 4)
            })

    system_snapshot = get_system_usage_snapshot()
    gpu_snapshot = get_nvidia_gpu_snapshot()

    system_report = {**system_snapshot, **gpu_snapshot}
    save_json(OUTPUT_DIR / "system_usage_summary.json", system_report)

    overall_df = pd.DataFrame(rows)
    overall_df.to_csv(OUTPUT_DIR / "overall_scores.csv", index=False, encoding="utf-8-sig")

    print("\n=== ASR ===")
    print(asr_df)
    print("\n=== SEGMENTATION ===")
    print(seg_df)
    print("\n=== SUMMARY ===")
    print(sum_df)
    print("\n=== OVERALL ===")
    print(overall_df)

    print("\n=== SYSTEM USAGE ===")
    for k, v in system_report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()