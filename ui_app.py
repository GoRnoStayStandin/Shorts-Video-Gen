import html
import re
import time
from pathlib import Path
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

from benchmark_tools import (
    OUTPUT_DIR,
    GPT4ALL_MODEL_PATH,
    GPT4All,
    MODELS_DIR,
    ASR_ALLOW_MODEL_DOWNGRADE,
    ASR_AUTO_TUNE,
    ASR_DEVICE,
    FASTER_WHISPER_MODEL,
    FW_GPU_BATCH_SIZE,
    CHAPTER_BLOCK_MAX_CHARS,
    CHAPTER_BLOCK_TARGET_SEC,
    CHAPTER_BOUNDARY_MAX_TOKENS,
    CHAPTER_BOUNDARY_OVERLAP_BLOCKS,
    CHAPTER_BOUNDARY_SENSITIVITY,
    CHAPTER_BOUNDARY_WINDOW_BLOCKS,
    CHAPTER_FULL_TRANSCRIPT_MAX_CHARS,
    CHAPTER_MAX_BLOCKS_PER_WINDOW,
    CHAPTER_MAX_TOKENS,
    CHAPTER_OVERLAP_BLOCKS,
    CHAPTER_SUMMARY_WINDOW_BLOCKS,
    CHAPTERING_VERSION,
    GPT4ALL_DEVICE,
    GPT4ALL_FAST_TOPIC_MAX_TOKENS,
    GPT4ALL_GPU_BACKENDS,
    GPT4ALL_MIN_FREE_VRAM_MB,
    GPT4ALL_TOPIC_MAX_TOKENS,
    LAST_ASR_RUNTIME,
    LAST_CHAPTERING_DEBUG,
    LAST_GPT4ALL_RUNTIME,
    TOPIC_FAST_MAX_SEGMENTS_PER_WINDOW,
    TOPIC_FAST_OVERLAP_SEGMENTS,
    TOPIC_MAX_SEGMENTS_PER_WINDOW,
    TOPIC_OVERLAP_SEGMENTS,
    USE_NVENC_FOR_EXPORT,
    ResourceMonitor,
    analyze_topic_with_gpt4all_model,
    build_asr_profile_ladder,
    build_srt_for_clip,
    clean_text_for_topic_analysis,
    download_video_from_url,
    export_clip_with_optional_srt,
    format_timestamp,
    get_video_duration_sec,
    is_supported_video_url,
    llm_chapter_segmentation,
    llm_topic_segmentation,
    load_gpt4all_model,
    make_topic_label,
    normalize_subtitle_split_mode,
    read_json_if_exists,
    safe_filename,
    save_json,
    score_topic_segment_v2,
    transcribe_with_faster_whisper,
    write_pipeline_text_report,
)
from timestamp_eval import (
    block_to_intervals,
    evaluate_segments,
    format_timestamp as format_eval_timestamp,
    infer_duration,
    parse_blocks as parse_timestamp_blocks,
    parse_timestamp as parse_eval_timestamp,
)


# =========================
# UI CONFIG
# =========================

st.set_page_config(
    page_title="AI Video Clip Cutter",
    layout="wide",
    initial_sidebar_state="collapsed"
)

UI_UPLOAD_DIR = OUTPUT_DIR / "ui_uploads"
UI_DOWNLOAD_DIR = OUTPUT_DIR / "ui_downloads"
UI_PROJECTS_DIR = OUTPUT_DIR / "ui_projects"

UI_UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
UI_DOWNLOAD_DIR.mkdir(exist_ok=True, parents=True)
UI_PROJECTS_DIR.mkdir(exist_ok=True, parents=True)

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def get_models_dir() -> Path:
    if MODELS_DIR.is_absolute():
        return MODELS_DIR
    return Path(__file__).resolve().parent / MODELS_DIR


def get_available_gpt4all_models():
    candidates = []
    seen = set()

    default_path = Path(GPT4ALL_MODEL_PATH)
    if default_path.exists():
        resolved = default_path.resolve()
        candidates.append(resolved)
        seen.add(str(resolved).lower())

    models_dir = get_models_dir()
    if models_dir.exists():
        for path in sorted(models_dir.rglob("*.gguf")):
            resolved = path.resolve()
            key = str(resolved).lower()
            if key not in seen:
                candidates.append(resolved)
                seen.add(key)

    return candidates


def format_model_option(model_path: str) -> str:
    path = Path(model_path)
    models_dir = get_models_dir().resolve()

    try:
        return str(path.resolve().relative_to(models_dir))
    except ValueError:
        return path.name


@st.cache_data(ttl=30, show_spinner=False)
def get_asr_profile_preview():
    return build_asr_profile_ladder()


# Локальная страховка: чтобы ui_app.py не падал, даже если safe_filename не импортировался из benchmark_tools.
def ui_safe_filename(name: str, max_len: int = 90) -> str:
    name = str(name or "file").strip()
    name = re.sub(r"[^\w\dа-яА-ЯёЁ\-. ]+", "_", name)
    name = re.sub(r"\s+", "_", name)
    return (name[:max_len] or "file").strip("_") or "file"


# =========================
# SESSION HELPERS
# =========================

def reset_processing_state():
    keys_to_remove = [
        "video_path",
        "project_dir",
        "media_duration",
        "fw_segments",
        "topic_segments",
        "selected_position",
        "generated_srt_by_clip",
        "exported_clip_by_clip",
        "preview_clip_by_clip",
        "upload_signature",
        "source_url",
        "source_path",
        "source_type",
    ]

    for key in keys_to_remove:
        st.session_state.pop(key, None)

    for key in list(st.session_state.keys()):
        if key.startswith(("title_", "summary_", "text_", "start_", "end_", "range_", "aspect_", "subtitle_")):
            st.session_state.pop(key, None)


def save_uploaded_video(uploaded_file):
    signature = f"{uploaded_file.name}:{uploaded_file.size}"

    if st.session_state.get("upload_signature") == signature and st.session_state.get("video_path"):
        return

    reset_processing_state()

    timestamp = int(time.time())
    clean_name = ui_safe_filename(uploaded_file.name)
    upload_path = UI_UPLOAD_DIR / f"{timestamp}_{clean_name}"

    upload_path.write_bytes(uploaded_file.getbuffer())

    project_name = f"{Path(clean_name).stem}_{timestamp}"
    project_dir = UI_PROJECTS_DIR / project_name
    project_dir.mkdir(exist_ok=True, parents=True)

    st.session_state.upload_signature = signature
    st.session_state.video_path = str(upload_path)
    st.session_state.project_dir = str(project_dir)
    st.session_state.source_url = None
    st.session_state.source_path = None
    st.session_state.source_type = "upload"
    st.session_state.generated_srt_by_clip = {}
    st.session_state.exported_clip_by_clip = {}
    st.session_state.preview_clip_by_clip = {}


def select_video_from_local_path(path_text: str):
    raw_path = str(path_text or "").strip().strip('"').strip("'")

    if not raw_path:
        st.error("Укажите путь к видеофайлу на этом компьютере.")
        return

    video_path = Path(raw_path).expanduser()

    if not video_path.exists():
        st.error(f"Файл не найден: {video_path}")
        return

    if not video_path.is_file():
        st.error(f"Указанный путь не является файлом: {video_path}")
        return

    if video_path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        st.error("Поддерживаются видеофайлы: mp4, mov, mkv, avi, webm, m4v.")
        return

    video_path = video_path.resolve()
    stat = video_path.stat()
    signature = f"path:{str(video_path).lower()}:{stat.st_size}:{stat.st_mtime_ns}"

    if st.session_state.get("upload_signature") == signature and st.session_state.get("video_path"):
        return

    reset_processing_state()

    timestamp = int(time.time())
    project_name = f"{ui_safe_filename(video_path.stem)}_{timestamp}"
    project_dir = UI_PROJECTS_DIR / project_name
    project_dir.mkdir(exist_ok=True, parents=True)

    st.session_state.upload_signature = signature
    st.session_state.video_path = str(video_path)
    st.session_state.project_dir = str(project_dir)
    st.session_state.source_url = None
    st.session_state.source_path = str(video_path)
    st.session_state.source_type = "path"
    st.session_state.generated_srt_by_clip = {}
    st.session_state.exported_clip_by_clip = {}
    st.session_state.preview_clip_by_clip = {}

    st.success(f"Видео выбрано: {video_path.name}")


def download_video_from_link(video_url: str):
    video_url = str(video_url or "").strip()

    if not video_url:
        st.error("Вставьте ссылку на публичное видео YouTube или VK.")
        return

    if not is_supported_video_url(video_url):
        st.error("Поддерживаются только публичные ссылки YouTube, YouTube Shorts, VK и VK Video.")
        return

    signature = f"url:{video_url}"

    if st.session_state.get("upload_signature") == signature and st.session_state.get("video_path"):
        return

    reset_processing_state()

    progress_bar = st.progress(0)
    status_box = st.empty()

    def download_progress(progress_value, message):
        progress_bar.progress(min(max(float(progress_value), 0.0), 1.0))
        status_box.info(message)

    try:
        video_path = download_video_from_url(
            url=video_url,
            out_dir=UI_DOWNLOAD_DIR,
            progress_callback=download_progress
        )
    except Exception as e:
        status_box.error(f"Не удалось скачать видео: {e}")
        progress_bar.progress(1.0)
        return

    timestamp = int(time.time())
    project_name = f"{ui_safe_filename(video_path.stem)}_{timestamp}"
    project_dir = UI_PROJECTS_DIR / project_name
    project_dir.mkdir(exist_ok=True, parents=True)

    st.session_state.upload_signature = signature
    st.session_state.video_path = str(video_path)
    st.session_state.project_dir = str(project_dir)
    st.session_state.source_url = video_url
    st.session_state.source_path = None
    st.session_state.source_type = "url"
    st.session_state.generated_srt_by_clip = {}
    st.session_state.exported_clip_by_clip = {}
    st.session_state.preview_clip_by_clip = {}

    status_box.success(f"Видео скачано: {video_path.name}")


# =========================
# TIMELINE
# =========================

def build_timeline_html(segments, duration, selected_clip_id=None, timeline_title="Таймлайн найденных смысловых фрагментов"):
    if not duration or duration <= 0:
        return "<p>Нет данных для таймлайна</p>"

    markers_html = ""

    for index, seg in enumerate(segments):
        start = float(seg.get("start", 0))
        end = float(seg.get("end", start))
        clip_id = seg.get("id")
        score = float(seg.get("score", 0) or 0)

        left = max(0, min(100, start / duration * 100))
        width = max(0.4, min(100 - left, (end - start) / duration * 100))

        is_selected = clip_id == selected_clip_id

        color = "#f59e0b" if is_selected else "#2563eb"
        opacity = 0.95 if is_selected else min(0.85, 0.30 + score * 0.55)
        height = "34px" if is_selected else "26px"
        top = "13px" if is_selected else "17px"

        title = html.escape(seg.get("title") or f"Фрагмент {index + 1}")
        tooltip = html.escape(
            f"{title} | {format_timestamp(start)} - {format_timestamp(end)} | score: {score:.2f}"
        )

        markers_html += f"""
        <div
            class="clip-marker"
            title="{tooltip}"
            style="
                left:{left:.3f}%;
                width:{width:.3f}%;
                background:{color};
                opacity:{opacity};
                height:{height};
                top:{top};
            "
        ></div>
        """

    tick_0 = "0:00"
    tick_25 = format_timestamp(duration * 0.25)
    tick_50 = format_timestamp(duration * 0.50)
    tick_75 = format_timestamp(duration * 0.75)
    tick_100 = format_timestamp(duration)

    return f"""
    <style>
        .timeline-box {{
            width: 100%;
            border: 1px solid #d0d7de;
            border-radius: 14px;
            padding: 16px 18px 12px 18px;
            background: #ffffff;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .timeline-title {{
            font-size: 15px;
            font-weight: 700;
            margin-bottom: 12px;
            color: #1f2937;
        }}

        .timeline-bar {{
            position: relative;
            height: 60px;
            border-radius: 12px;
            background: linear-gradient(90deg, #f3f4f6, #e5e7eb);
            overflow: hidden;
            border: 1px solid #cbd5e1;
        }}

        .clip-marker {{
            position: absolute;
            border-radius: 7px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.20);
            cursor: pointer;
            transition: 0.15s ease;
        }}

        .clip-marker:hover {{
            opacity: 1 !important;
            transform: translateY(-2px);
        }}

        .timeline-ticks {{
            position: relative;
            height: 24px;
            margin-top: 8px;
            font-size: 12px;
            color: #6b7280;
        }}

        .tick {{
            position: absolute;
            transform: translateX(-50%);
            white-space: nowrap;
        }}
    </style>

    <div class="timeline-box">
        <div class="timeline-title">{html.escape(timeline_title)}</div>
        <div class="timeline-bar">
            {markers_html}
        </div>
        <div class="timeline-ticks">
            <span class="tick" style="left:0%; transform:translateX(0);">{tick_0}</span>
            <span class="tick" style="left:25%;">{tick_25}</span>
            <span class="tick" style="left:50%;">{tick_50}</span>
            <span class="tick" style="left:75%;">{tick_75}</span>
            <span class="tick" style="left:100%; transform:translateX(-100%);">{tick_100}</span>
        </div>
    </div>
    """


def build_eval_comparison_timeline_html(reference, prediction, duration, iou_threshold=0.5):
    if not duration or duration <= 0:
        return "<p>Нет данных для сравнительного таймлайна</p>"

    result = evaluate_segments(reference, prediction, iou_threshold=iou_threshold)
    matched_ref_ids = {id(match["reference"]) for match in result["matches"]}
    matched_pred_ids = {id(match["prediction"]) for match in result["matches"]}
    ref_iou_by_id = {id(match["reference"]): float(match["iou"]) for match in result["matches"]}
    pred_iou_by_id = {id(match["prediction"]): float(match["iou"]) for match in result["matches"]}

    def marker_html(items, matched_ids, iou_by_id, kind):
        markers = []

        for index, item in enumerate(items, start=1):
            start = max(0.0, float(item.get("start", 0) or 0))
            end = max(start, float(item.get("end", start) or start))

            if end <= start:
                continue

            left = max(0.0, min(100.0, start / duration * 100.0))
            width = max(0.35, min(100.0 - left, (end - start) / duration * 100.0))
            is_match = id(item) in matched_ids
            iou = iou_by_id.get(id(item))

            if kind == "reference":
                color = "#16a34a" if is_match else "#f97316"
                status = "TP" if is_match else "FN"
            else:
                color = "#2563eb" if is_match else "#dc2626"
                status = "TP" if is_match else "FP"

            label = str(item.get("label") or f"{kind} {index}").strip()
            short_label = html.escape(label[:30] + ("..." if len(label) > 30 else ""))
            tooltip_parts = [
                status,
                label,
                f"{format_eval_timestamp(start)} - {format_eval_timestamp(end)}",
                f"duration {(end - start):.1f}s",
            ]

            if iou is not None:
                tooltip_parts.append(f"IoU {iou:.3f}")

            tooltip = html.escape(" | ".join(tooltip_parts))

            markers.append(f"""
            <div
                class="eval-segment eval-segment-{status.lower()}"
                title="{tooltip}"
                style="left:{left:.3f}%; width:{width:.3f}%; background:{color};"
            >
                <span>{html.escape(status)}</span>
                <em>{short_label}</em>
            </div>
            """)

        return "\n".join(markers)

    tick_0 = "0:00"
    tick_25 = format_eval_timestamp(duration * 0.25)
    tick_50 = format_eval_timestamp(duration * 0.50)
    tick_75 = format_eval_timestamp(duration * 0.75)
    tick_100 = format_eval_timestamp(duration)

    reference_markers = marker_html(reference, matched_ref_ids, ref_iou_by_id, "reference")
    prediction_markers = marker_html(prediction, matched_pred_ids, pred_iou_by_id, "prediction")

    return f"""
    <style>
        .eval-timeline-box {{
            width: 100%;
            border: 1px solid #d0d7de;
            border-radius: 14px;
            padding: 16px 18px 12px 18px;
            background: #ffffff;
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .eval-title {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            margin-bottom: 12px;
            color: #111827;
            font-size: 15px;
            font-weight: 700;
        }}

        .eval-legend {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            font-size: 12px;
            color: #4b5563;
            font-weight: 500;
        }}

        .eval-dot {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 999px;
            margin-right: 4px;
            vertical-align: -1px;
        }}

        .eval-row {{
            display: grid;
            grid-template-columns: 88px 1fr;
            gap: 12px;
            align-items: center;
            margin: 10px 0;
        }}

        .eval-row-label {{
            font-size: 13px;
            font-weight: 700;
            color: #374151;
            text-align: right;
        }}

        .eval-track {{
            position: relative;
            height: 56px;
            border-radius: 12px;
            overflow: hidden;
            background: repeating-linear-gradient(
                90deg,
                #f8fafc 0,
                #f8fafc 24.6%,
                #eef2f7 25%,
                #f8fafc 25.4%
            );
            border: 1px solid #cbd5e1;
        }}

        .eval-segment {{
            position: absolute;
            top: 10px;
            height: 36px;
            border-radius: 8px;
            color: #ffffff;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.18);
            overflow: hidden;
            white-space: nowrap;
            display: flex;
            align-items: center;
            gap: 5px;
            padding: 0 7px;
            box-sizing: border-box;
            font-size: 11px;
            line-height: 1;
        }}

        .eval-segment span {{
            font-weight: 800;
            letter-spacing: 0.02em;
        }}

        .eval-segment em {{
            font-style: normal;
            opacity: 0.92;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .eval-ticks {{
            position: relative;
            height: 24px;
            margin-left: 100px;
            margin-top: 8px;
            color: #6b7280;
            font-size: 12px;
        }}

        .eval-tick {{
            position: absolute;
            transform: translateX(-50%);
            white-space: nowrap;
        }}
    </style>

    <div class="eval-timeline-box">
        <div class="eval-title">
            <div>Сравнение таймингов при IoU {iou_threshold:.2f}</div>
            <div class="eval-legend">
                <span><i class="eval-dot" style="background:#16a34a"></i>Эталон TP</span>
                <span><i class="eval-dot" style="background:#f97316"></i>Эталон FN</span>
                <span><i class="eval-dot" style="background:#2563eb"></i>ИИ TP</span>
                <span><i class="eval-dot" style="background:#dc2626"></i>ИИ FP</span>
            </div>
        </div>
        <div class="eval-row">
            <div class="eval-row-label">Эталон</div>
            <div class="eval-track">{reference_markers}</div>
        </div>
        <div class="eval-row">
            <div class="eval-row-label">ИИ</div>
            <div class="eval-track">{prediction_markers}</div>
        </div>
        <div class="eval-ticks">
            <span class="eval-tick" style="left:0%; transform:translateX(0);">{tick_0}</span>
            <span class="eval-tick" style="left:25%;">{tick_25}</span>
            <span class="eval-tick" style="left:50%;">{tick_50}</span>
            <span class="eval-tick" style="left:75%;">{tick_75}</span>
            <span class="eval-tick" style="left:100%; transform:translateX(-100%);">{tick_100}</span>
        </div>
    </div>
    """


# =========================
# PIPELINE
# =========================

def process_video_pipeline(video_path: Path, project_dir: Path):
    project_dir.mkdir(exist_ok=True, parents=True)
    monitor = ResourceMonitor(project_dir / "monitoring", interval_sec=2.0).start()
    pipeline_started = time.perf_counter()
    stage_name = None
    stage_started = None
    stage_times = {}
    finalized = False
    report_context = {
        "video_path": str(video_path),
        "project_dir": str(project_dir),
    }

    def set_stage(name: str):
        nonlocal stage_name, stage_started
        now = time.perf_counter()
        if stage_name is not None and stage_started is not None:
            stage_times[stage_name] = stage_times.get(stage_name, 0.0) + (now - stage_started)
        stage_name = name
        stage_started = now
        monitor.set_stage(name)

    def finalize_pipeline(status: str, message: str = ""):
        nonlocal finalized, stage_name, stage_started
        if finalized:
            return

        now = time.perf_counter()
        if stage_name is not None and stage_started is not None:
            stage_times[stage_name] = stage_times.get(stage_name, 0.0) + (now - stage_started)
            stage_name = None
            stage_started = None

        resource_summary = monitor.stop()
        report_path = project_dir / "pipeline_run_report.json"
        text_report_path = project_dir / "pipeline_run_report.txt"
        report = {
            "status": status,
            "message": message,
            "started_at": resource_summary.get("started_at"),
            "ended_at": resource_summary.get("ended_at"),
            "elapsed_sec": round(time.perf_counter() - pipeline_started, 3),
            "stage_times_sec": {key: round(value, 3) for key, value in stage_times.items()},
            "context": report_context,
            "asr_runtime": dict(LAST_ASR_RUNTIME),
            "gpt4all_runtime": dict(LAST_GPT4ALL_RUNTIME),
            "resource_summary": resource_summary,
            "report_json_path": str(report_path),
            "report_txt_path": str(text_report_path),
        }
        save_json(report_path, report)
        write_pipeline_text_report(text_report_path, report)
        st.session_state.pipeline_run_report = report
        st.session_state.pipeline_run_report_path = str(report_path)
        st.session_state.pipeline_run_text_report_path = str(text_report_path)
        finalized = True

    progress_bar = st.progress(0)
    status_box = st.empty()

    try:
        set_stage("duration_probe")
        status_box.info("Получаю длительность видео...")
        media_duration = get_video_duration_sec(str(video_path))
        st.session_state.media_duration = media_duration
        report_context["media_duration_sec"] = media_duration
        progress_bar.progress(0.04)

        asr_dir = project_dir / "asr" / "faster_whisper"

        def asr_progress(progress_value, message):
            progress_bar.progress(min(max(progress_value, 0.0), 0.65))
            status_box.info(message)

        set_stage("asr")
        fw_segments = transcribe_with_faster_whisper(
            video_path=str(video_path),
            out_dir=asr_dir,
            progress_callback=asr_progress
        )
        report_context["asr_segment_count"] = len(fw_segments or [])

        st.session_state.fw_segments = fw_segments

        if not fw_segments:
            st.session_state.topic_segments = []
            message = "Речь не распознана или ASR-сегменты пустые."
            status_box.error(message)
            progress_bar.progress(1.0)
            finalize_pipeline("failed", message)
            return

        set_stage("segmentation_setup")
        topic_dir = project_dir / "topic_segmentation"
        topic_dir.mkdir(exist_ok=True, parents=True)

        segmentation_goal = st.session_state.get("segmentation_goal", "chapters")
        is_chapter_goal = segmentation_goal == "chapters"
        goal_label = "главы видео" if is_chapter_goal else "подтемы/Shorts"

        topic_mode = st.session_state.get("topic_mode", "fast")
        is_fast_topic_mode = topic_mode == "fast"
        chapter_sensitivity = str(st.session_state.get("chapter_sensitivity", CHAPTER_BOUNDARY_SENSITIVITY))

        if is_fast_topic_mode:
            topic_mode_label = "быстрый"
            topic_max_segments = TOPIC_FAST_MAX_SEGMENTS_PER_WINDOW
            topic_overlap = TOPIC_FAST_OVERLAP_SEGMENTS
        else:
            topic_mode_label = "качественный"
            topic_max_segments = TOPIC_MAX_SEGMENTS_PER_WINDOW
            topic_overlap = TOPIC_OVERLAP_SEGMENTS

        generate_metadata = bool(st.session_state.get("generate_topic_metadata", False))
        effective_generate_metadata = False if is_chapter_goal else generate_metadata
        metadata_label = "metadata" if effective_generate_metadata else "bounds"
        if is_chapter_goal:
            topic_max_tokens = CHAPTER_BOUNDARY_MAX_TOKENS
        else:
            topic_max_tokens = GPT4ALL_TOPIC_MAX_TOKENS if effective_generate_metadata else GPT4ALL_FAST_TOPIC_MAX_TOKENS

        selected_model_path = str(st.session_state.get("gpt4all_model_path") or GPT4ALL_MODEL_PATH)
        selected_model_name = Path(selected_model_path).stem
        model_cache_key = ui_safe_filename(selected_model_name, max_len=70)
        segmentation_cache_version = CHAPTERING_VERSION if is_chapter_goal else "topics_v1"
        sensitivity_cache_part = f"_{chapter_sensitivity}" if is_chapter_goal else ""
        topic_cache_path = topic_dir / f"topic_segments_{segmentation_goal}_{segmentation_cache_version}{sensitivity_cache_part}_{topic_mode}_{metadata_label}_{model_cache_key}.json"
        report_context.update({
            "segmentation_goal": segmentation_goal,
            "segmentation_goal_label": goal_label,
            "topic_mode": topic_mode,
            "topic_mode_label": topic_mode_label,
            "chapter_sensitivity": chapter_sensitivity if is_chapter_goal else None,
            "generate_metadata_requested": generate_metadata,
            "generate_metadata_effective": effective_generate_metadata,
            "model_path": selected_model_path,
            "model_name": selected_model_name,
            "topic_cache_path": str(topic_cache_path),
        })

        cached_topics = read_json_if_exists(str(topic_cache_path))

        if isinstance(cached_topics, list) and cached_topics:
            save_json(topic_dir / "topic_segments.json", cached_topics)
            st.session_state.topic_segments = cached_topics
            st.session_state.selected_position = 0
            report_context["topic_segment_count"] = len(cached_topics)
            progress_bar.progress(1.0)
            message = f"Использую кэш ИИ-сегментации ({goal_label}, {topic_mode_label} режим, {selected_model_name})."
            status_box.success(message)
            finalize_pipeline("cached", message)
            return

        status_box.info(f"Выделяю {goal_label} через GPT4All ({topic_mode_label} режим, {metadata_label}, {selected_model_name})...")
        progress_bar.progress(0.68)

        if GPT4All is None:
            st.session_state.topic_segments = []
            message = f"gpt4all не установлен. Невозможно выполнить ИИ-сегментацию: {goal_label}."
            status_box.error(message)
            progress_bar.progress(1.0)
            finalize_pipeline("failed", message)
            return

        if not Path(selected_model_path).exists():
            st.session_state.topic_segments = []
            message = f"Модель GPT4All не найдена: {selected_model_path}"
            status_box.error(message)
            progress_bar.progress(1.0)
            finalize_pipeline("failed", message)
            return

        set_stage("model_load")
        try:
            model = load_gpt4all_model(n_ctx=4096, model_path=selected_model_path)
        except Exception as e:
            st.session_state.topic_segments = []
            message = f"Не удалось загрузить GPT4All: {e}"
            status_box.error(message)
            progress_bar.progress(1.0)
            finalize_pipeline("failed", message)
            return

        if LAST_GPT4ALL_RUNTIME:
            status_box.info(
                "GPT4All загружен: "
                f"backend `{LAST_GPT4ALL_RUNTIME.get('backend', 'unknown')}`, "
                f"device `{LAST_GPT4ALL_RUNTIME.get('device') or LAST_GPT4ALL_RUNTIME.get('attempted_device', 'cpu')}`."
            )

        def topic_progress(done, total, message):
            local_progress = done / total if total else 1.0
            progress_bar.progress(0.68 + min(local_progress, 1.0) * 0.29)
            status_box.info(message)

        set_stage("llm_processing")
        try:
            if is_chapter_goal:
                enriched = llm_chapter_segmentation(
                    whisper_segments=fw_segments,
                    model=model,
                    target_block_duration_sec=CHAPTER_BLOCK_TARGET_SEC,
                    max_block_text_chars=CHAPTER_BLOCK_MAX_CHARS,
                    max_blocks_per_window=CHAPTER_MAX_BLOCKS_PER_WINDOW,
                    overlap_blocks=CHAPTER_OVERLAP_BLOCKS,
                    min_chapter_duration_sec=20.0,
                    generate_metadata=False,
                    max_tokens=topic_max_tokens,
                    sensitivity=chapter_sensitivity,
                    progress_callback=topic_progress
                )
            else:
                enriched = llm_topic_segmentation(
                    whisper_segments=fw_segments,
                    model=model,
                    max_segments_per_window=topic_max_segments,
                    overlap_segments=topic_overlap,
                    min_topic_duration_sec=8.0,
                    max_topic_duration_sec=240.0,
                    fast_mode=is_fast_topic_mode,
                    generate_metadata=effective_generate_metadata,
                    topic_max_tokens=topic_max_tokens,
                    progress_callback=topic_progress
                )

        except Exception as e:
            st.session_state.topic_segments = []
            message = f"Ошибка ИИ-сегментации ({goal_label}): {e}"
            status_box.error(message)
            progress_bar.progress(1.0)
            finalize_pipeline("failed", message)
            return

        report_context["topic_segment_count"] = len(enriched or [])

        if not enriched:
            st.session_state.topic_segments = []
            message = (
                f"ИИ не нашёл подходящие {goal_label}. "
                "Обычно это значит, что GPT4All не смог корректно вернуть JSON или контекст модели всё ещё маловат."
            )
            status_box.warning(message)
            progress_bar.progress(1.0)
            finalize_pipeline("failed", message)
            return

        set_stage("save_results")
        save_json(topic_cache_path, enriched)
        save_json(topic_dir / "topic_segments.json", enriched)
        save_json(
            topic_dir / "topic_segmentation_settings.json",
            {
                "segmentation_goal": segmentation_goal,
                "segmentation_cache_version": segmentation_cache_version,
                "chaptering_version": CHAPTERING_VERSION if is_chapter_goal else None,
                "goal_label": goal_label,
                "mode": topic_mode,
                "mode_label": topic_mode_label,
                "max_segments_per_window": topic_max_segments,
                "overlap_segments": topic_overlap,
                "chapter_block_target_sec": CHAPTER_BLOCK_TARGET_SEC if is_chapter_goal else None,
                "chapter_block_max_chars": CHAPTER_BLOCK_MAX_CHARS if is_chapter_goal else None,
                "chapter_max_blocks_per_window": CHAPTER_MAX_BLOCKS_PER_WINDOW if is_chapter_goal else None,
                "chapter_overlap_blocks": CHAPTER_OVERLAP_BLOCKS if is_chapter_goal else None,
                "chapter_boundary_window_blocks": CHAPTER_BOUNDARY_WINDOW_BLOCKS if is_chapter_goal else None,
                "chapter_boundary_overlap_blocks": CHAPTER_BOUNDARY_OVERLAP_BLOCKS if is_chapter_goal else None,
                "chapter_boundary_sensitivity": chapter_sensitivity if is_chapter_goal else None,
                "chapter_full_transcript_max_chars": CHAPTER_FULL_TRANSCRIPT_MAX_CHARS if is_chapter_goal else None,
                "chapter_summary_window_blocks": CHAPTER_SUMMARY_WINDOW_BLOCKS if is_chapter_goal else None,
                "max_tokens": topic_max_tokens,
                "fast_mode": is_fast_topic_mode,
                "generate_metadata_requested": generate_metadata,
                "generate_metadata_effective": effective_generate_metadata,
                "model_path": selected_model_path,
                "model_name": selected_model_name,
            }
        )

        if is_chapter_goal and LAST_CHAPTERING_DEBUG:
            save_json(topic_dir / "debug_chaptering_v3.json", LAST_CHAPTERING_DEBUG)

        st.session_state.topic_segments = enriched
        st.session_state.selected_position = 0

        progress_bar.progress(1.0)
        status_box.success("Обработка завершена.")
        finalize_pipeline("success", "Обработка завершена.")
    except Exception as e:
        finalize_pipeline("error", str(e))
        raise


def render_pipeline_report(project_dir: Path):
    report_path = project_dir / "pipeline_run_report.json"
    report = st.session_state.get("pipeline_run_report") or read_json_if_exists(str(report_path))

    if not isinstance(report, dict):
        return

    resource_summary = report.get("resource_summary") or {}
    stage_times = report.get("stage_times_sec") or {}
    text_report_path = Path(report.get("report_txt_path") or (project_dir / "pipeline_run_report.txt"))

    with st.expander("Отчет запуска и ресурсы", expanded=False):
        st.caption(f"JSON: `{report_path}`")
        if text_report_path.exists():
            st.caption(f"TXT: `{text_report_path}`")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Статус", str(report.get("status", "unknown")))
        m2.metric("Время", f"{float(report.get('elapsed_sec') or 0):.1f} сек")
        m3.metric("CPU peak", f"{float(resource_summary.get('cpu_percent_peak') or 0):.0f}%")
        m4.metric("GPU peak", f"{float(resource_summary.get('gpu_util_percent_peak') or 0):.0f}%")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("RAM peak", f"{float(resource_summary.get('ram_used_mb_peak') or 0):.0f} MB")
        r2.metric("VRAM peak", f"{float(resource_summary.get('gpu_memory_used_mb_peak') or 0):.0f} MB")
        r3.metric("Disk read", f"{float(resource_summary.get('disk_read_mb_total') or 0):.0f} MB")
        r4.metric("Disk write", f"{float(resource_summary.get('disk_write_mb_total') or 0):.0f} MB")

        if report.get("message"):
            st.write(report["message"])

        if stage_times:
            st.write("Время по этапам")
            st.dataframe(
                [{"stage": key, "seconds": round(float(value), 2)} for key, value in stage_times.items()],
                use_container_width=True,
                hide_index=True,
            )

        stages = resource_summary.get("stages") or {}
        if stages:
            st.write("Нагрузка по этапам")
            stage_rows = []
            for stage, data in stages.items():
                stage_rows.append({
                    "stage": stage,
                    "CPU avg": data.get("cpu_percent_avg"),
                    "CPU peak": data.get("cpu_percent_peak"),
                    "GPU avg": data.get("gpu_util_percent_avg"),
                    "GPU peak": data.get("gpu_util_percent_peak"),
                    "VRAM peak MB": data.get("gpu_memory_used_mb_peak"),
                    "RAM peak MB": data.get("ram_used_mb_peak"),
                })
            st.dataframe(stage_rows, use_container_width=True, hide_index=True)

        download_col_1, download_col_2 = st.columns(2)
        if text_report_path.exists():
            with open(text_report_path, "rb") as f:
                download_col_1.download_button(
                    "Скачать TXT отчет",
                    data=f,
                    file_name=text_report_path.name,
                    mime="text/plain",
                    use_container_width=True,
                )

        samples_csv = resource_summary.get("samples_csv")
        if samples_csv and Path(samples_csv).exists():
            with open(samples_csv, "rb") as f:
                download_col_2.download_button(
                    "Скачать CSV ресурсов",
                    data=f,
                    file_name=Path(samples_csv).name,
                    mime="text/csv",
                    use_container_width=True,
                )


# =========================
# CLIP EDITING HELPERS
# =========================

def ensure_clip_widget_defaults(seg):
    clip_id = seg["id"]

    title_key = f"title_{clip_id}"
    summary_key = f"summary_{clip_id}"
    text_key = f"text_{clip_id}"
    start_key = f"start_{clip_id}"
    end_key = f"end_{clip_id}"
    range_key = f"range_{clip_id}"

    refresh_widgets = st.session_state.pop(f"refresh_clip_widgets_{clip_id}", False)

    if refresh_widgets or title_key not in st.session_state:
        st.session_state[title_key] = seg.get("title") or ""

    if refresh_widgets or summary_key not in st.session_state:
        st.session_state[summary_key] = seg.get("summary") or ""

    if refresh_widgets or text_key not in st.session_state:
        st.session_state[text_key] = seg.get("text") or ""

    if refresh_widgets or start_key not in st.session_state:
        st.session_state[start_key] = format_timestamp(seg.get("start", 0))

    if refresh_widgets or end_key not in st.session_state:
        st.session_state[end_key] = format_timestamp(seg.get("end", 0))

    if refresh_widgets or range_key not in st.session_state:
        st.session_state[range_key] = (float(seg.get("start", 0) or 0), float(seg.get("end", 0) or 0))

    return title_key, summary_key, text_key, start_key, end_key, range_key


def apply_widget_values_to_segment(position: int):
    seg = st.session_state.topic_segments[position]
    clip_id = seg["id"]

    title_key = f"title_{clip_id}"
    summary_key = f"summary_{clip_id}"
    text_key = f"text_{clip_id}"

    seg["title"] = st.session_state.get(title_key, seg.get("title", ""))
    seg["summary"] = st.session_state.get(summary_key, seg.get("summary", ""))
    seg["text"] = st.session_state.get(text_key, seg.get("text", ""))

    st.session_state.topic_segments[position] = seg

    edited_path = Path(st.session_state.project_dir) / "topic_segmentation" / "topic_segments_edited.json"
    save_json(edited_path, st.session_state.topic_segments)

    return seg


SUBTITLE_SPLIT_LABELS = {
    "ASR-сегмент": "segment",
    "1 слово": "words_1",
    "2 слова": "words_2",
    "3 слова": "words_3",
    "5 слов": "words_5",
    "Целое предложение": "sentence",
}

SUBTITLE_POSITION_LABELS = {
    "Снизу": "bottom",
    "Выше снизу": "upper_bottom",
    "По центру": "center",
}

SUBTITLE_ANIMATION_LABELS = {
    "Без анимации": "none",
    "Fade in/out": "fade",
    "Pop-up": "popup",
}

SUBTITLE_FONT_OPTIONS = ["Arial", "Segoe UI", "Tahoma", "Verdana"]


def subtitle_cache_key(clip_id, subtitle_split_mode: str, namespace: str = "clip") -> str:
    return f"{namespace}:{clip_id}:{normalize_subtitle_split_mode(subtitle_split_mode)}"


def default_subtitle_margin_for_position(position_label: str) -> int:
    if position_label == "По центру":
        return 0
    if position_label == "Выше снизу":
        return 220
    return 32


def get_export_settings_for_clip(clip_id):
    aspect_mode = "vertical_9_16" if st.session_state.get(f"aspect_{clip_id}") == "Вертикальный 9:16" else "original"
    subtitle_mode = st.session_state.get(f"subtitle_mode_{clip_id}", "Отдельный SRT рядом")
    subtitle_position_label = st.session_state.get(f"subtitle_position_{clip_id}", "Снизу")
    subtitle_split_label = st.session_state.get(f"subtitle_split_{clip_id}", "3 слова")
    subtitle_animation_label = st.session_state.get(f"subtitle_animation_{clip_id}", "Без анимации")
    subtitle_style = {
        "subtitle_font_name": st.session_state.get(f"subtitle_font_{clip_id}", "Arial"),
        "subtitle_primary_color": st.session_state.get(f"subtitle_color_{clip_id}", "#FFFFFF"),
        "subtitle_outline_color": st.session_state.get(f"subtitle_outline_color_{clip_id}", "#000000"),
        "subtitle_outline_width": int(st.session_state.get(f"subtitle_outline_width_{clip_id}", 2)),
        "subtitle_shadow": int(st.session_state.get(f"subtitle_shadow_{clip_id}", 1)),
        "subtitle_margin_v": int(
            st.session_state.get(
                f"subtitle_margin_{clip_id}",
                default_subtitle_margin_for_position(subtitle_position_label),
            )
        ),
        "subtitle_animation": SUBTITLE_ANIMATION_LABELS.get(subtitle_animation_label, "none"),
    }
    return (
        aspect_mode,
        subtitle_mode,
        SUBTITLE_POSITION_LABELS.get(subtitle_position_label, "bottom"),
        int(st.session_state.get(f"subtitle_size_{clip_id}", 28)),
        SUBTITLE_SPLIT_LABELS.get(subtitle_split_label, "words_3"),
        subtitle_style,
    )


def render_export_settings_controls(clip_id):
    aspect_key = f"aspect_{clip_id}"
    subtitle_mode_key = f"subtitle_mode_{clip_id}"
    subtitle_position_key = f"subtitle_position_{clip_id}"
    subtitle_size_key = f"subtitle_size_{clip_id}"
    subtitle_split_key = f"subtitle_split_{clip_id}"

    st.radio(
        "Формат видео",
        options=["Оригинальный", "Вертикальный 9:16"],
        horizontal=True,
        key=aspect_key
    )

    st.radio(
        "Субтитры при экспорте",
        options=["Не добавлять", "Отдельный SRT рядом", "Вшить в видео"],
        horizontal=True,
        index=1,
        key=subtitle_mode_key
    )

    st.selectbox(
        "Разбиение субтитров",
        options=list(SUBTITLE_SPLIT_LABELS.keys()),
        index=3,
        key=subtitle_split_key,
        help="Уменьшает количество слов на экране. Тайминги внутри ASR-сегмента распределяются пропорционально."
    )

    if st.session_state.get(subtitle_mode_key) != "Вшить в видео":
        return

    sub_col_1, sub_col_2, sub_col_3 = st.columns(3)

    with sub_col_1:
        st.selectbox(
            "Положение субтитров",
            options=list(SUBTITLE_POSITION_LABELS.keys()),
            key=subtitle_position_key
        )

    with sub_col_2:
        st.slider(
            "Размер субтитров",
            min_value=16,
            max_value=52,
            value=28,
            step=2,
            key=subtitle_size_key
        )

    with sub_col_3:
        current_position_label = st.session_state.get(subtitle_position_key, "Снизу")
        st.slider(
            "Отступ по вертикали",
            min_value=0,
            max_value=500,
            value=default_subtitle_margin_for_position(current_position_label),
            step=4,
            key=f"subtitle_margin_{clip_id}",
            help="Для нижнего положения это отступ от нижнего края."
        )

    style_col_1, style_col_2, style_col_3 = st.columns(3)

    with style_col_1:
        st.selectbox(
            "Шрифт",
            options=SUBTITLE_FONT_OPTIONS,
            key=f"subtitle_font_{clip_id}"
        )

    with style_col_2:
        st.color_picker(
            "Цвет текста",
            value="#FFFFFF",
            key=f"subtitle_color_{clip_id}"
        )

    with style_col_3:
        st.color_picker(
            "Цвет обводки",
            value="#000000",
            key=f"subtitle_outline_color_{clip_id}"
        )

    effect_col_1, effect_col_2, effect_col_3 = st.columns(3)

    with effect_col_1:
        st.slider(
            "Толщина обводки",
            min_value=0,
            max_value=8,
            value=2,
            step=1,
            key=f"subtitle_outline_width_{clip_id}"
        )

    with effect_col_2:
        st.slider(
            "Тень",
            min_value=0,
            max_value=5,
            value=1,
            step=1,
            key=f"subtitle_shadow_{clip_id}"
        )

    with effect_col_3:
        st.selectbox(
            "Анимация burn-in",
            options=list(SUBTITLE_ANIMATION_LABELS.keys()),
            key=f"subtitle_animation_{clip_id}",
            help="Анимация применяется только при вшивании субтитров в видео."
        )


def generate_srt_for_clip(seg, fw_segments, project_dir: Path, subtitle_split_mode: str = "segment", subdir: str = "selected_subtitles"):
    out_dir = project_dir / subdir
    out_dir.mkdir(exist_ok=True, parents=True)

    clip_id = seg["id"]
    title = ui_safe_filename(seg.get("title") or f"clip_{clip_id}")
    start = float(seg["start"])
    end = float(seg["end"])
    subtitle_split_mode = normalize_subtitle_split_mode(subtitle_split_mode)

    srt_path = out_dir / f"clip_{int(clip_id):03d}_{title}_{subtitle_split_mode}.srt"

    build_srt_for_clip(
        whisper_segments=fw_segments,
        clip_start=start,
        clip_end=end,
        srt_path=srt_path,
        subtitle_split_mode=subtitle_split_mode
    )

    return srt_path


def ensure_srt_for_export(
    seg,
    fw_segments,
    project_dir: Path,
    subtitle_mode: str,
    subtitle_split_mode: str,
    cache_id,
    subdir: str = "selected_subtitles",
    cache_namespace: str = "clip",
):
    if subtitle_mode == "Не добавлять":
        return None

    st.session_state.setdefault("generated_srt_by_clip", {})
    cache_key = subtitle_cache_key(cache_id, subtitle_split_mode, namespace=cache_namespace)
    saved_srt = st.session_state.generated_srt_by_clip.get(cache_key)

    if saved_srt and Path(saved_srt).exists():
        return Path(saved_srt)

    srt_path = generate_srt_for_clip(
        seg=seg,
        fw_segments=fw_segments,
        project_dir=project_dir,
        subtitle_split_mode=subtitle_split_mode,
        subdir=subdir,
    )
    st.session_state.generated_srt_by_clip[cache_key] = str(srt_path)
    return srt_path


def parse_ui_timestamp(value: str) -> float:
    value = str(value or "").strip().replace(",", ".")

    if not value:
        raise ValueError("пустой таймкод")

    if ":" not in value:
        return float(value)

    parts = value.split(":")

    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)

    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    raise ValueError(f"некорректный таймкод: {value}")


def build_clip_text_from_range(fw_segments, clip_start: float, clip_end: float) -> str:
    parts = []

    for seg in fw_segments:
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", seg_start))

        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        text = str(seg.get("text", "")).strip()
        if text:
            parts.append(text)

    return clean_text_for_topic_analysis(" ".join(parts))


def find_asr_ids_for_range(fw_segments, clip_start: float, clip_end: float):
    matched_ids = []

    for idx, seg in enumerate(fw_segments):
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", seg_start))

        if seg_end > clip_start and seg_start < clip_end:
            matched_ids.append(idx)

    if not matched_ids:
        return None, None

    return matched_ids[0], matched_ids[-1]


def clear_clip_outputs(clip_id):
    clip_key = str(clip_id)
    generated_srt = st.session_state.setdefault("generated_srt_by_clip", {})
    for key in list(generated_srt):
        key_text = str(key)
        if key_text == clip_key or key_text.startswith(f"{clip_key}:") or key_text.startswith(f"clip:{clip_key}:"):
            generated_srt.pop(key, None)
    st.session_state.setdefault("exported_clip_by_clip", {}).pop(clip_key, None)
    st.session_state.setdefault("preview_clip_by_clip", {}).pop(clip_key, None)


def apply_clip_boundaries(position: int, new_start: float, new_end: float, fw_segments, media_duration: float):
    seg = apply_widget_values_to_segment(position)
    clip_id = seg["id"]

    media_duration = float(media_duration or 0)
    new_start = max(0.0, float(new_start))
    new_end = max(0.0, float(new_end))

    if media_duration > 0:
        new_start = min(new_start, media_duration)
        new_end = min(new_end, media_duration)

    if new_end <= new_start:
        raise ValueError("Конец клипа должен быть позже начала.")

    seg["start"] = new_start
    seg["end"] = new_end
    seg["duration"] = new_end - new_start
    seg["text"] = build_clip_text_from_range(fw_segments, new_start, new_end) or seg.get("text", "")

    start_id, end_id = find_asr_ids_for_range(fw_segments, new_start, new_end)
    if start_id is not None:
        seg["start_id"] = start_id
        seg["end_id"] = end_id

    seg["score"] = score_topic_segment_v2(seg)
    seg["boundaries_edited"] = True

    st.session_state.topic_segments[position] = seg
    st.session_state[f"refresh_clip_widgets_{clip_id}"] = True

    clear_clip_outputs(clip_id)
    save_json(Path(st.session_state.project_dir) / "topic_segmentation" / "topic_segments_edited.json", st.session_state.topic_segments)
    return seg


def filter_segments_for_range(fw_segments, clip_start: float, clip_end: float):
    filtered = []

    for seg in fw_segments:
        seg_start = float(seg.get("start", 0) or 0)
        seg_end = float(seg.get("end", seg_start) or seg_start)

        if seg_end > clip_start and seg_start < clip_end:
            filtered.append(dict(seg))

    return filtered


def chapter_shorts_cache_path(project_dir: Path, chapter_seg: dict, model_path: str, topic_mode: str) -> Path:
    chapter_id = int(chapter_seg.get("id", 0) or 0)
    model_key = ui_safe_filename(Path(model_path).stem, max_len=70)
    mode_key = str(topic_mode or "fast")
    return project_dir / "topic_segmentation" / "chapter_shorts" / f"chapter_{chapter_id:03d}_shorts_{mode_key}_{model_key}.json"


def format_short_label(index: int, short: dict) -> str:
    title = str(short.get("title") or f"Short {index + 1}").strip()
    if len(title) > 42:
        title = title[:42] + "..."
    return f"{index + 1}. score {float(short.get('score', 0) or 0):.2f} · {format_timestamp(short.get('start', 0))}-{format_timestamp(short.get('end', 0))} · {title}"


def render_chapter_shorts_panel(selected_seg, fw_segments, project_dir: Path):
    chapter_id = int(selected_seg.get("id", 0) or 0)
    selected_model_path = str(st.session_state.get("gpt4all_model_path") or GPT4ALL_MODEL_PATH)
    topic_mode = st.session_state.get("topic_mode", "fast")
    is_fast_topic_mode = topic_mode == "fast"
    cache_path = chapter_shorts_cache_path(project_dir, selected_seg, selected_model_path, topic_mode)
    session_key = f"chapter_shorts_{chapter_id}_{topic_mode}_{ui_safe_filename(Path(selected_model_path).stem, max_len=70)}"

    with st.expander("Shorts внутри выбранной главы", expanded=False):
        st.caption(
            "Ищет короткие смысловые фрагменты только внутри выбранной главы. "
            "Список глав и F1 по главам не перезаписываются."
        )
        st.caption(f"Кэш: `{cache_path}`")

        shorts = st.session_state.get(session_key)
        if shorts is None:
            cached = read_json_if_exists(str(cache_path))
            shorts = cached if isinstance(cached, list) else []
            st.session_state[session_key] = shorts

        run_col, reload_col = st.columns(2)

        with run_col:
            if st.button("Найти Shorts в этой главе", use_container_width=True, key=f"find_chapter_shorts_{chapter_id}"):
                chapter_start = float(selected_seg.get("start", 0) or 0)
                chapter_end = float(selected_seg.get("end", chapter_start) or chapter_start)
                chapter_segments = filter_segments_for_range(fw_segments, chapter_start, chapter_end)

                if len(chapter_segments) < 3:
                    st.warning("В главе слишком мало ASR-сегментов для поиска Shorts.")
                elif GPT4All is None:
                    st.error("gpt4all не установлен.")
                elif not Path(selected_model_path).exists():
                    st.error(f"Модель GPT4All не найдена: {selected_model_path}")
                else:
                    with st.spinner("Ищу Shorts внутри выбранной главы..."):
                        try:
                            model = load_gpt4all_model(n_ctx=4096, model_path=selected_model_path)
                            raw_shorts = llm_topic_segmentation(
                                whisper_segments=chapter_segments,
                                model=model,
                                max_segments_per_window=TOPIC_FAST_MAX_SEGMENTS_PER_WINDOW if is_fast_topic_mode else TOPIC_MAX_SEGMENTS_PER_WINDOW,
                                overlap_segments=TOPIC_FAST_OVERLAP_SEGMENTS if is_fast_topic_mode else TOPIC_OVERLAP_SEGMENTS,
                                min_topic_duration_sec=8.0,
                                max_topic_duration_sec=180.0,
                                fast_mode=is_fast_topic_mode,
                                generate_metadata=bool(st.session_state.get("generate_topic_metadata", False)),
                                topic_max_tokens=GPT4ALL_FAST_TOPIC_MAX_TOKENS if is_fast_topic_mode else GPT4ALL_TOPIC_MAX_TOKENS,
                            )
                        except Exception as e:
                            raw_shorts = []
                            st.error(f"Не удалось найти Shorts внутри главы: {e}")

                    if raw_shorts:
                        prepared = []
                        for index, item in enumerate(raw_shorts, start=1):
                            short = dict(item)
                            short["id"] = chapter_id * 1000 + index
                            short["short_id"] = index
                            short["segment_kind"] = "chapter_short"
                            short["parent_chapter_id"] = chapter_id
                            short["parent_chapter_title"] = selected_seg.get("title")
                            short["parent_chapter_start"] = chapter_start
                            short["parent_chapter_end"] = chapter_end
                            prepared.append(short)

                        cache_path.parent.mkdir(exist_ok=True, parents=True)
                        save_json(cache_path, prepared)
                        st.session_state[session_key] = prepared
                        st.success(f"Найдено Shorts: {len(prepared)}")
                        st.rerun()
                    elif "raw_shorts" in locals():
                        st.warning("Shorts внутри этой главы не найдены.")

        with reload_col:
            if st.button("Загрузить сохраненные Shorts", use_container_width=True, key=f"load_chapter_shorts_{chapter_id}"):
                cached = read_json_if_exists(str(cache_path))
                if isinstance(cached, list) and cached:
                    st.session_state[session_key] = cached
                    st.success(f"Загружено Shorts: {len(cached)}")
                    st.rerun()
                else:
                    st.info("Сохраненных Shorts для этой главы пока нет.")

        shorts = st.session_state.get(session_key) or []
        if not shorts:
            st.info("Shorts внутри выбранной главы пока не найдены.")
            return

        short_rows = [
            {
                "#": index,
                "score": round(float(short.get("score", 0) or 0), 3),
                "start": format_timestamp(short.get("start", 0)),
                "end": format_timestamp(short.get("end", 0)),
                "duration": round(float(short.get("duration", 0) or 0), 1),
                "title": short.get("title", ""),
            }
            for index, short in enumerate(shorts, start=1)
        ]
        st.dataframe(short_rows, use_container_width=True, hide_index=True)

        selected_short_index = st.radio(
            "Выбранный Short",
            options=list(range(len(shorts))),
            format_func=lambda index: format_short_label(index, shorts[index]),
            key=f"selected_chapter_short_{chapter_id}"
        )
        selected_short = dict(shorts[selected_short_index])
        try:
            short_number = int(selected_short.get("short_id") or selected_short_index + 1)
        except (TypeError, ValueError):
            short_number = selected_short_index + 1
        selected_short["short_id"] = short_number
        selected_short.setdefault("id", chapter_id * 1000 + short_number)
        selected_short.setdefault("segment_kind", "chapter_short")
        st.write(
            f"**{selected_short.get('title') or f'Short {selected_short_index + 1}'}** · "
            f"{format_timestamp(selected_short.get('start', 0))}-{format_timestamp(selected_short.get('end', 0))} · "
            f"score {float(selected_short.get('score', 0) or 0):.2f}"
        )
        st.text_area(
            "Текст выбранного Short",
            value=selected_short.get("text", ""),
            height=140,
            key=f"selected_chapter_short_text_{chapter_id}_{selected_short_index}",
            disabled=True,
        )

        short_id = selected_short.get("short_id", selected_short_index + 1)
        short_cache_id = f"chapter_short_{chapter_id}_{short_id}"
        short_namespace = "chapter_short"
        short_subdir = "selected_subtitles/chapter_shorts"
        st.session_state.setdefault("generated_srt_by_clip", {})
        st.session_state.setdefault("exported_clip_by_clip", {})
        st.session_state.setdefault("preview_clip_by_clip", {})

        with st.expander("Настройки предпросмотра и экспорта Short", expanded=False):
            render_export_settings_controls(short_cache_id)

        short_action_col_1, short_action_col_2, short_action_col_3 = st.columns(3)

        with short_action_col_1:
            if st.button("Сгенерировать SRT для Short", use_container_width=True, key=f"srt_chapter_short_{chapter_id}_{short_id}"):
                _, _, _, _, subtitle_split_mode, _ = get_export_settings_for_clip(short_cache_id)

                with st.spinner("Генерирую SRT для выбранного Short..."):
                    srt_path = generate_srt_for_clip(
                        seg=selected_short,
                        fw_segments=fw_segments,
                        project_dir=project_dir,
                        subtitle_split_mode=subtitle_split_mode,
                        subdir=short_subdir,
                    )

                st.session_state.generated_srt_by_clip[
                    subtitle_cache_key(short_cache_id, subtitle_split_mode, namespace=short_namespace)
                ] = str(srt_path)
                st.success(f"SRT для Short создан: {srt_path}")

        with short_action_col_2:
            if st.button("Собрать предпросмотр Short", use_container_width=True, key=f"preview_chapter_short_{chapter_id}_{short_id}"):
                aspect_mode, subtitle_mode, subtitle_position, subtitle_size, subtitle_split_mode, subtitle_style = get_export_settings_for_clip(short_cache_id)

                with st.spinner("Собираю предпросмотр Short..."):
                    try:
                        srt_path = ensure_srt_for_export(
                            seg=selected_short,
                            fw_segments=fw_segments,
                            project_dir=project_dir,
                            subtitle_mode=subtitle_mode,
                            subtitle_split_mode=subtitle_split_mode,
                            cache_id=short_cache_id,
                            subdir=short_subdir,
                            cache_namespace=short_namespace,
                        ) if subtitle_mode == "Вшить в видео" else None
                        preview_path = export_clip_with_optional_srt(
                            video_path=Path(st.session_state.video_path),
                            seg=selected_short,
                            out_dir=project_dir / "previews" / "chapter_shorts",
                            srt_path=srt_path,
                            burn_subtitles=subtitle_mode == "Вшить в видео",
                            aspect_mode=aspect_mode,
                            subtitle_position=subtitle_position,
                            subtitle_font_size=subtitle_size,
                            **subtitle_style,
                        )
                        st.session_state.preview_clip_by_clip[short_cache_id] = str(preview_path)
                        st.success(f"Предпросмотр Short создан: {preview_path}")
                    except Exception as e:
                        st.error(f"Не удалось собрать предпросмотр Short: {e}")

        with short_action_col_3:
            if st.button("Экспортировать Short", type="primary", use_container_width=True, key=f"export_chapter_short_{chapter_id}_{short_id}"):
                aspect_mode, subtitle_mode, subtitle_position, subtitle_size, subtitle_split_mode, subtitle_style = get_export_settings_for_clip(short_cache_id)

                with st.spinner("Экспортирую выбранный Short..."):
                    try:
                        srt_path = ensure_srt_for_export(
                            seg=selected_short,
                            fw_segments=fw_segments,
                            project_dir=project_dir,
                            subtitle_mode=subtitle_mode,
                            subtitle_split_mode=subtitle_split_mode,
                            cache_id=short_cache_id,
                            subdir=short_subdir,
                            cache_namespace=short_namespace,
                        )
                        clip_path = export_clip_with_optional_srt(
                            video_path=Path(st.session_state.video_path),
                            seg=selected_short,
                            out_dir=project_dir / "exports" / "chapter_shorts",
                            srt_path=srt_path,
                            burn_subtitles=subtitle_mode == "Вшить в видео",
                            aspect_mode=aspect_mode,
                            subtitle_position=subtitle_position,
                            subtitle_font_size=subtitle_size,
                            **subtitle_style,
                        )

                        st.session_state.exported_clip_by_clip[short_cache_id] = str(clip_path)
                        st.success(f"Short экспортирован: {clip_path}")
                    except Exception as e:
                        st.error(f"Не удалось экспортировать Short: {e}")

        short_split_label = st.session_state.get(f"subtitle_split_{short_cache_id}", "3 слова")
        short_split_mode = SUBTITLE_SPLIT_LABELS.get(short_split_label, "words_3")
        short_srt = st.session_state.generated_srt_by_clip.get(
            subtitle_cache_key(short_cache_id, short_split_mode, namespace=short_namespace)
        )
        short_export = st.session_state.exported_clip_by_clip.get(short_cache_id)
        short_preview = st.session_state.preview_clip_by_clip.get(short_cache_id)

        if short_srt and Path(short_srt).exists():
            srt_path = Path(short_srt)

            with open(srt_path, "rb") as f:
                st.download_button(
                    label="Скачать SRT Short",
                    data=f,
                    file_name=srt_path.name,
                    mime="text/plain",
                    use_container_width=True,
                    key=f"download_srt_chapter_short_{chapter_id}_{short_id}",
                )

        if short_export and Path(short_export).exists():
            clip_path = Path(short_export)
            st.video(str(clip_path))

            with open(clip_path, "rb") as f:
                st.download_button(
                    label="Скачать экспортированный Short MP4",
                    data=f,
                    file_name=clip_path.name,
                    mime="video/mp4",
                    use_container_width=True,
                    key=f"download_export_chapter_short_{chapter_id}_{short_id}",
                )

        elif short_preview and Path(short_preview).exists():
            st.caption("Предпросмотр выбранного Short")
            st.video(str(short_preview))


# =========================
# WORKSPACE
# =========================

def render_workspace():
    segments = st.session_state.get("topic_segments", [])
    fw_segments = st.session_state.get("fw_segments", [])
    media_duration = st.session_state.get("media_duration", 0)
    is_chapter_goal = st.session_state.get("segmentation_goal", "chapters") == "chapters"
    segment_plural = "главы" if is_chapter_goal else "фрагменты"
    segment_single = "главе" if is_chapter_goal else "фрагменте"
    clip_word = "главы" if is_chapter_goal else "клипа"
    st.session_state.setdefault("generated_srt_by_clip", {})
    st.session_state.setdefault("exported_clip_by_clip", {})
    st.session_state.setdefault("preview_clip_by_clip", {})

    if not segments:
        st.warning(f"Пока нет найденных {segment_plural}. Сначала запусти обработку видео.")
        return

    st.divider()

    left_col, right_col = st.columns([0.32, 0.68], gap="large")

    with left_col:
        st.subheader(f"Найденные {segment_plural}")

        def format_clip_label(i):
            seg = segments[i]
            score = float(seg.get("score", 0) or 0)
            title = seg.get("title") or (f"Глава {seg.get('id', i + 1)}" if is_chapter_goal else f"Клип {seg.get('id', i + 1)}")
            start = format_timestamp(seg.get("start", 0))
            end = format_timestamp(seg.get("end", 0))

            if len(title) > 42:
                title = title[:42] + "..."

            if is_chapter_goal:
                return f"{i + 1}. {start}-{end} · {title}"

            return f"{i + 1}. score {score:.2f} · {start}-{end} · {title}"

        selected_position = st.radio(
            label="Главы отсортированы по времени" if is_chapter_goal else "Кандидаты отсортированы по убыванию score",
            options=list(range(len(segments))),
            index=min(st.session_state.get("selected_position", 0), len(segments) - 1),
            format_func=format_clip_label,
            label_visibility="visible"
        )

        st.session_state.selected_position = selected_position

        sort_label = "Отсортировать по времени" if is_chapter_goal else "Пересортировать по score"

        if st.button(sort_label, use_container_width=True):
            if is_chapter_goal:
                st.session_state.topic_segments.sort(key=lambda x: float(x.get("start", 0) or 0))
            else:
                st.session_state.topic_segments.sort(
                    key=lambda x: float(x.get("score", 0) or 0),
                    reverse=True
                )
            st.rerun()

    selected_seg = segments[selected_position]
    selected_clip_id = selected_seg["id"]

    with right_col:
        components.html(
            build_timeline_html(
                segments=segments,
                duration=media_duration,
                selected_clip_id=selected_clip_id,
                timeline_title="Таймлайн найденных глав" if is_chapter_goal else "Таймлайн найденных смысловых фрагментов"
            ),
            height=150
        )

        title_key, summary_key, text_key, start_key, end_key, range_key = ensure_clip_widget_defaults(selected_seg)

        st.subheader(f"Подробная информация о {segment_single}")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.metric("Score", f"{float(selected_seg.get('score', 0) or 0):.2f}")

        with m2:
            st.metric("Начало", format_timestamp(selected_seg.get("start", 0)))

        with m3:
            st.metric("Конец", format_timestamp(selected_seg.get("end", 0)))

        with m4:
            st.metric("Длительность", f"{float(selected_seg.get('duration', 0) or 0):.1f} сек.")

        if selected_seg.get("analysis_error"):
            st.warning(f"Предупреждение анализа: {selected_seg.get('analysis_error')}")

        with st.expander("Технические границы ASR-сегментов", expanded=False):
            st.write(
                f"start_id: `{selected_seg.get('start_id')}` · "
                f"end_id: `{selected_seg.get('end_id')}`"
            )

        with st.expander(f"Редактирование границ {clip_word}", expanded=True):
            max_slider_value = max(float(media_duration or 0), float(selected_seg.get("end", 0) or 0), 1.0)
            current_start = max(0.0, min(float(selected_seg.get("start", 0) or 0), max_slider_value))
            current_end = max(current_start + 0.1, min(float(selected_seg.get("end", current_start + 1) or current_start + 1), max_slider_value))

            range_start, range_end = st.session_state.get(range_key, (current_start, current_end))
            range_start = max(0.0, min(float(range_start), max_slider_value))
            range_end = max(range_start + 0.1, min(float(range_end), max_slider_value))
            st.session_state[range_key] = (range_start, range_end)

            st.slider(
                f"Границы {clip_word}, сек.",
                min_value=0.0,
                max_value=float(max_slider_value),
                value=st.session_state[range_key],
                step=1.0,
                key=range_key
            )

            boundary_col_1, boundary_col_2 = st.columns(2)

            with boundary_col_1:
                st.text_input("Начало клипа", key=start_key, help="Например: 5:45 или 0:05:45")

            with boundary_col_2:
                st.text_input("Конец клипа", key=end_key, help="Например: 7:35 или 0:07:35")

            apply_slider_col, apply_text_col = st.columns(2)

            with apply_slider_col:
                if st.button("Применить границы из ползунка", use_container_width=True):
                    try:
                        slider_start, slider_end = st.session_state[range_key]
                        apply_clip_boundaries(selected_position, slider_start, slider_end, fw_segments, media_duration)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Не удалось применить границы: {e}")

            with apply_text_col:
                if st.button("Применить границы из полей", use_container_width=True):
                    try:
                        apply_clip_boundaries(
                            selected_position,
                            parse_ui_timestamp(st.session_state.get(start_key)),
                            parse_ui_timestamp(st.session_state.get(end_key)),
                            fw_segments,
                            media_duration
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"Не удалось применить границы: {e}")

        if st.button("Перегенерировать название и summary", use_container_width=True):
            current_text = st.session_state.get(text_key, selected_seg.get("text", ""))
            selected_model_path = str(st.session_state.get("gpt4all_model_path") or GPT4ALL_MODEL_PATH)

            if GPT4All is None:
                st.error("gpt4all не установлен.")
            elif not Path(selected_model_path).exists():
                st.error(f"Модель GPT4All не найдена: {selected_model_path}")
            else:
                with st.spinner("Перегенерирую описание текущего клипа..."):
                    try:
                        model = load_gpt4all_model(n_ctx=4096, model_path=selected_model_path)
                        analysis = analyze_topic_with_gpt4all_model(
                            model=model,
                            text=current_text
                        )
                    except Exception as e:
                        analysis = {
                            "title": make_topic_label(current_text, max_words=6).replace("_", " "),
                            "summary": None,
                            "error": str(e)
                        }

                selected_seg["title"] = analysis.get("title")
                selected_seg["summary"] = analysis.get("summary")
                selected_seg["analysis_error"] = analysis.get("error")
                selected_seg["analysis_model"] = selected_model_path

                st.session_state[title_key] = selected_seg["title"] or ""
                st.session_state[summary_key] = selected_seg["summary"] or ""

                st.session_state.topic_segments[selected_position] = selected_seg
                st.rerun()

        st.text_input(
            "Название главы" if is_chapter_goal else "Название клипа",
            key=title_key
        )

        st.text_area(
            "Summary главы" if is_chapter_goal else "Summary клипа",
            key=summary_key,
            height=100
        )

        st.text_area(
            "Текст главы / что говорится в этом разделе" if is_chapter_goal else "Текст клипа / что говорится во фрагменте",
            key=text_key,
            height=220
        )

        if is_chapter_goal:
            render_chapter_shorts_panel(
                selected_seg=selected_seg,
                fw_segments=fw_segments,
                project_dir=Path(st.session_state.project_dir),
            )

        with st.expander("Настройки предпросмотра и экспорта", expanded=False):
            render_export_settings_controls(selected_clip_id)

        def get_export_settings():
            return get_export_settings_for_clip(selected_clip_id)

        def ensure_srt_if_needed(seg, subtitle_mode, subtitle_split_mode):
            return ensure_srt_for_export(
                seg=seg,
                fw_segments=fw_segments,
                project_dir=Path(st.session_state.project_dir),
                subtitle_mode=subtitle_mode,
                subtitle_split_mode=subtitle_split_mode,
                cache_id=selected_clip_id,
            )

        preview_button_label = "Собрать предпросмотр главы" if is_chapter_goal else "Собрать предпросмотр"
        export_button_label = "Экспортировать главу" if is_chapter_goal else "Экспортировать клип"

        action_col_1, action_col_2, action_col_3, action_col_4 = st.columns(4)

        with action_col_1:
            if st.button("Применить правки", use_container_width=True):
                apply_widget_values_to_segment(selected_position)
                st.success("Правки сохранены в карточку клипа.")

        with action_col_2:
            if st.button("Сгенерировать SRT", use_container_width=True):
                seg = apply_widget_values_to_segment(selected_position)
                _, _, _, _, subtitle_split_mode, _ = get_export_settings()

                with st.spinner("Генерирую SRT по таймкодам faster-whisper..."):
                    srt_path = generate_srt_for_clip(
                        seg=seg,
                        fw_segments=fw_segments,
                        project_dir=Path(st.session_state.project_dir),
                        subtitle_split_mode=subtitle_split_mode
                )

                st.session_state.generated_srt_by_clip[subtitle_cache_key(selected_clip_id, subtitle_split_mode)] = str(srt_path)
                st.success(f"SRT создан: {srt_path}")

        with action_col_3:
            if st.button(preview_button_label, use_container_width=True):
                seg = apply_widget_values_to_segment(selected_position)
                aspect_mode, subtitle_mode, subtitle_position, subtitle_size, subtitle_split_mode, subtitle_style = get_export_settings()

                with st.spinner("Собираю предпросмотр клипа..."):
                    try:
                        srt_path = ensure_srt_if_needed(seg, subtitle_mode, subtitle_split_mode) if subtitle_mode == "Вшить в видео" else None
                        preview_path = export_clip_with_optional_srt(
                            video_path=Path(st.session_state.video_path),
                            seg=seg,
                            out_dir=Path(st.session_state.project_dir) / "previews",
                            srt_path=srt_path,
                            burn_subtitles=subtitle_mode == "Вшить в видео",
                            aspect_mode=aspect_mode,
                            subtitle_position=subtitle_position,
                            subtitle_font_size=subtitle_size,
                            **subtitle_style,
                        )
                        st.session_state.preview_clip_by_clip[str(selected_clip_id)] = str(preview_path)
                        st.success(f"Предпросмотр создан: {preview_path}")
                    except Exception as e:
                        st.error(f"Не удалось собрать предпросмотр: {e}")

        with action_col_4:
            if st.button(export_button_label, type="primary", use_container_width=True):
                seg = apply_widget_values_to_segment(selected_position)
                aspect_mode, subtitle_mode, subtitle_position, subtitle_size, subtitle_split_mode, subtitle_style = get_export_settings()

                with st.spinner("Экспортирую выбранный клип..."):
                    try:
                        srt_path = ensure_srt_if_needed(seg, subtitle_mode, subtitle_split_mode)
                        clip_path = export_clip_with_optional_srt(
                            video_path=Path(st.session_state.video_path),
                            seg=seg,
                            out_dir=Path(st.session_state.project_dir) / "exports",
                            srt_path=srt_path,
                            burn_subtitles=subtitle_mode == "Вшить в видео",
                            aspect_mode=aspect_mode,
                            subtitle_position=subtitle_position,
                            subtitle_font_size=subtitle_size,
                            **subtitle_style,
                        )

                        st.session_state.exported_clip_by_clip[str(selected_clip_id)] = str(clip_path)
                        st.success(f"Клип экспортирован: {clip_path}")

                    except Exception as e:
                        st.error(f"Не удалось экспортировать клип: {e}")

        current_split_label = st.session_state.get(f"subtitle_split_{selected_clip_id}", "3 слова")
        current_split_mode = SUBTITLE_SPLIT_LABELS.get(current_split_label, "words_3")
        current_srt = st.session_state.generated_srt_by_clip.get(subtitle_cache_key(selected_clip_id, current_split_mode))
        current_export = st.session_state.exported_clip_by_clip.get(str(selected_clip_id))
        current_preview = st.session_state.preview_clip_by_clip.get(str(selected_clip_id))

        if current_srt and Path(current_srt).exists():
            srt_path = Path(current_srt)

            with open(srt_path, "rb") as f:
                st.download_button(
                    label="Скачать SRT",
                    data=f,
                    file_name=srt_path.name,
                    mime="text/plain",
                    use_container_width=True
                )

        if current_export and Path(current_export).exists():
            clip_path = Path(current_export)

            st.video(str(clip_path))

            with open(clip_path, "rb") as f:
                st.download_button(
                    label="Скачать экспортированный MP4",
                    data=f,
                    file_name=clip_path.name,
                    mime="video/mp4",
                    use_container_width=True
                )

        elif current_preview and Path(current_preview).exists():
            st.caption("Предпросмотр выбранного клипа")
            st.video(str(current_preview))

    if is_chapter_goal:
        render_chapter_debug_panel(Path(st.session_state.project_dir))


def load_chapter_debug(project_dir: Path):
    debug_path = project_dir / "topic_segmentation" / "debug_chaptering_v3.json"
    data = read_json_if_exists(str(debug_path))

    if isinstance(data, dict):
        return debug_path, data

    return debug_path, None


def format_boundary_sources(boundary: dict) -> str:
    sources = boundary.get("sources") or []
    if not sources and boundary.get("source"):
        sources = [boundary.get("source")]
    return ", ".join(str(item) for item in sources if str(item).strip()) or "unknown"


def render_chapter_debug_panel(project_dir: Path):
    debug_path, debug_data = load_chapter_debug(project_dir)

    with st.expander("Диагностика границ глав", expanded=False):
        if not debug_data:
            st.info(f"Debug-файл пока не найден: `{debug_path}`")
            return

        blocks = debug_data.get("blocks") or []
        selected_boundaries = debug_data.get("selected_boundaries") or []
        raw_candidates = debug_data.get("raw_candidates") or []
        raw_answers = debug_data.get("raw_answers") or []
        ranges = debug_data.get("ranges") or []
        sensitivity = debug_data.get("sensitivity") or {}
        blocks_by_id = {int(block.get("id")): block for block in blocks if "id" in block}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Блоков", debug_data.get("block_count", len(blocks)))
        c2.metric("Окон LLM", debug_data.get("boundary_window_count", len(raw_answers)))
        c3.metric("Кандидатов", debug_data.get("raw_candidate_count", len(raw_candidates)))
        c4.metric("Выбрано границ", len(selected_boundaries))

        st.caption(
            f"Версия `{debug_data.get('version', 'unknown')}`, метод `{debug_data.get('method', 'unknown')}`, "
            f"чувствительность `{sensitivity.get('name', 'unknown')}`. Debug: `{debug_path}`"
        )

        if selected_boundaries:
            boundary_rows = []
            for index, boundary in enumerate(selected_boundaries, start=1):
                block_id = int(boundary.get("block_id", 0))
                block = blocks_by_id.get(block_id, {})
                boundary_rows.append({
                    "#": index,
                    "block_id": block_id,
                    "time": format_timestamp(block.get("start", 0)),
                    "segment_id": boundary.get("refined_segment_id"),
                    "segment_time": format_timestamp(boundary.get("refined_time", block.get("start", 0)) or 0),
                    "confidence": round(float(boundary.get("confidence", 0) or 0), 3),
                    "sources": format_boundary_sources(boundary),
                    "reason": boundary.get("reason", ""),
                    "segment_text": str(boundary.get("refined_text", ""))[:140],
                    "block_text": str(block.get("text", ""))[:160],
                })

            st.write("Выбранные границы")
            st.dataframe(boundary_rows, use_container_width=True, hide_index=True)

            boundary_options = [
                f"#{row['#']} · block {row['block_id']} · {row['time']} · conf {row['confidence']}"
                for row in boundary_rows
            ]
            selected_boundary_label = st.selectbox("Контекст вокруг границы", boundary_options)
            selected_boundary_index = boundary_options.index(selected_boundary_label)
            selected_block_id = int(boundary_rows[selected_boundary_index]["block_id"])
            context_radius = st.slider("Блоков вокруг границы", 1, 8, 3, key="chapter_debug_context_radius")

            context_rows = []
            for block_id in range(selected_block_id - context_radius, selected_block_id + context_radius + 1):
                block = blocks_by_id.get(block_id)
                if not block:
                    continue

                context_rows.append({
                    "mark": "START" if block_id == selected_block_id else "",
                    "block_id": block_id,
                    "time": f"{format_timestamp(block.get('start', 0))}-{format_timestamp(block.get('end', 0))}",
                    "gap_before": "" if block.get("gap_before") is None else round(float(block.get("gap_before") or 0), 2),
                    "text": block.get("text", ""),
                })

            st.dataframe(context_rows, use_container_width=True, hide_index=True)
        else:
            st.warning("Выбранных границ нет. Проверь raw-ответы и кандидаты ниже.")

        if ranges:
            range_rows = []
            for index, item in enumerate(ranges, start=1):
                if "start_id" in item:
                    range_rows.append({
                        "#": index,
                        "start_segment": item.get("start_id"),
                        "end_segment": item.get("end_id"),
                        "start_time": format_timestamp(item.get("start", 0) or 0),
                        "end_time": format_timestamp(item.get("end", 0) or 0),
                    })
                else:
                    range_rows.append({
                        "#": index,
                        "start_block": item.get("start_block_id"),
                        "end_block": item.get("end_block_id"),
                        "start_time": format_timestamp(blocks_by_id.get(int(item.get("start_block_id", 0)), {}).get("start", 0)),
                        "end_time": format_timestamp(blocks_by_id.get(int(item.get("end_block_id", 0)), {}).get("end", 0)),
                    })

            st.write("Собранные диапазоны глав")
            st.dataframe(range_rows, use_container_width=True, hide_index=True)

        with st.expander("Все кандидаты границ", expanded=False):
            candidate_rows = []
            for candidate in raw_candidates:
                block_id = int(candidate.get("block_id", 0))
                block = blocks_by_id.get(block_id, {})
                candidate_rows.append({
                    "block_id": block_id,
                    "time": format_timestamp(block.get("start", 0)),
                    "confidence": round(float(candidate.get("confidence", 0) or 0), 3),
                    "sources": format_boundary_sources(candidate),
                    "reason": candidate.get("reason", ""),
                })

            if candidate_rows:
                st.dataframe(candidate_rows, use_container_width=True, hide_index=True)
            else:
                st.info("Кандидаты не сохранены или не найдены.")

        with st.expander("Raw-ответы LLM", expanded=False):
            if not raw_answers:
                st.info("Raw-ответов нет.")
            else:
                answer_labels = [
                    f"Окно {item.get('window_index')} · blocks {item.get('start_block_id')}-{item.get('end_block_id')}"
                    for item in raw_answers
                ]
                selected_answer_label = st.selectbox("LLM-ответ", answer_labels, key="chapter_debug_raw_answer_select")
                selected_answer = raw_answers[answer_labels.index(selected_answer_label)]
                st.text_area(
                    "Ответ модели",
                    value=selected_answer.get("answer", ""),
                    height=220,
                    key="chapter_debug_raw_answer_text"
                )


def decode_uploaded_text(uploaded_file) -> str:
    raw = uploaded_file.getvalue()

    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="ignore")


def model_segments_as_intervals():
    intervals = []
    is_chapter_goal = st.session_state.get("segmentation_goal", "chapters") == "chapters"

    for seg in st.session_state.get("topic_segments", []) or []:
        start = float(seg.get("start", 0) or 0)
        end = float(seg.get("end", start) or start)

        if end <= start:
            continue

        intervals.append({
            "start": start,
            "end": end,
            "label": seg.get("title") or (f"Глава {seg.get('id', len(intervals) + 1)}" if is_chapter_goal else f"Клип {seg.get('id', len(intervals) + 1)}"),
            "source": "model",
        })

    return sorted(intervals, key=lambda item: item["start"])


def format_interval_for_option(index: int, intervals: list, source_label: str) -> str:
    if intervals:
        first = intervals[0]
        last = intervals[-1]
        return (
            f"{source_label}: {len(intervals)} сегм. "
            f"({format_eval_timestamp(first['start'])}-{format_eval_timestamp(last['end'])})"
        )

    return f"{source_label}: 0 сегм."


def render_timestamp_evaluation():
    is_chapter_goal = st.session_state.get("segmentation_goal", "chapters") == "chapters"
    eval_title = "Оценка глав / F1-score" if is_chapter_goal else "Оценка таймингов / F1-score"

    with st.expander(eval_title, expanded=False):
        st.caption(
            "Загрузите или вставьте тайминги глав, выберите эталон и сравниваемый блок. " if is_chapter_goal else
            "Загрузите или вставьте тайминги, выберите эталон и сравниваемый блок. "
        )
        st.caption(
            "Сегмент считается найденным, если IoU по времени выше заданного порога."
        )

        uploaded_timestamps = st.file_uploader(
            "Файл с таймингами (.txt)",
            type=["txt"],
            key="timestamps_eval_file"
        )

        if uploaded_timestamps is not None:
            file_signature = f"{uploaded_timestamps.name}:{uploaded_timestamps.size}"

            if st.session_state.get("timestamps_eval_file_signature") != file_signature:
                st.session_state.timestamps_eval_text = decode_uploaded_text(uploaded_timestamps)
                st.session_state.timestamps_eval_file_signature = file_signature

        st.text_area(
            "Тайминги",
            key="timestamps_eval_text",
            height=220,
            placeholder=(
                "00:00 - 00:46 Вступление\n"
                "00:47 - 03:30 Ультрадиффузные галактики\n\n"
                "00:00 - Начало\n00:46 - Следующая тема"
            )
        )

        timing_text = st.session_state.get("timestamps_eval_text", "")
        model_intervals = model_segments_as_intervals()

        try:
            blocks = parse_timestamp_blocks(timing_text) if timing_text.strip() else []
        except Exception as e:
            st.error(f"Не удалось разобрать тайминги: {e}")
            return

        inferred_duration = infer_duration(blocks) if blocks else None
        default_duration = float(st.session_state.get("media_duration") or inferred_duration or 0)

        duration_value = st.text_input(
            "Длительность видео для последнего маркера без конца",
            value=format_eval_timestamp(default_duration) if default_duration > 0 else "",
            help="Нужно для блоков вида `19:27 - Завершение`, где нет конечного таймкода."
        )

        try:
            fallback_duration = parse_eval_timestamp(duration_value) if duration_value.strip() else inferred_duration
        except Exception as e:
            st.error(f"Некорректная длительность видео: {e}")
            return

        sources = []

        for index, block in enumerate(blocks):
            intervals = block_to_intervals(block, fallback_duration=fallback_duration)
            sources.append({
                "label": format_interval_for_option(index, intervals, f"Блок {index}"),
                "intervals": intervals,
                "kind": "text",
                "index": index,
            })

        if model_intervals:
            model_source_label = "Текущие главы модели" if is_chapter_goal else "Текущие сегменты модели"
            sources.append({
                "label": format_interval_for_option(len(sources), model_intervals, model_source_label),
                "intervals": model_intervals,
                "kind": "model",
                "index": None,
            })

        if not sources:
            st.info("Добавьте тайминги или сначала выполните ИИ-сегментацию видео.")
            return

        if len(sources) < 2:
            st.info("Для F1 нужны минимум два набора интервалов: эталон и сравниваемые тайминги.")
            return

        option_labels = [source["label"] for source in sources]
        default_prediction_index = len(sources) - 1

        eval_col_1, eval_col_2, eval_col_3 = st.columns([0.38, 0.38, 0.24])

        with eval_col_1:
            reference_label = st.selectbox("Эталон", option_labels, index=0)

        with eval_col_2:
            prediction_label = st.selectbox("Сравниваем", option_labels, index=default_prediction_index)

        with eval_col_3:
            iou_threshold = st.slider("IoU", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

        reference = sources[option_labels.index(reference_label)]["intervals"]
        prediction = sources[option_labels.index(prediction_label)]["intervals"]

        timeline_duration = max(
            [
                float(st.session_state.get("media_duration") or 0),
                float(fallback_duration or 0),
            ] + [
                float(item.get("end", 0) or 0)
                for item in reference + prediction
            ]
        )

        components.html(
            build_eval_comparison_timeline_html(
                reference=reference,
                prediction=prediction,
                duration=timeline_duration,
                iou_threshold=iou_threshold,
            ),
            height=230,
        )

        quick_rows = []
        for quick_iou in (0.30, 0.50, 0.70):
            quick_result = evaluate_segments(reference, prediction, iou_threshold=quick_iou)
            quick_rows.append({
                "IoU": f"{quick_iou:.2f}",
                "TP": quick_result["tp"],
                "FP": quick_result["fp"],
                "FN": quick_result["fn"],
                "Precision": round(quick_result["precision"], 3),
                "Recall": round(quick_result["recall"], 3),
                "F1": round(quick_result["f1"], 3),
            })

        st.write("Быстрое сравнение по порогам IoU")
        st.dataframe(quick_rows, use_container_width=True, hide_index=True)

        if st.button("Посчитать F1", type="primary", use_container_width=True):
            result = evaluate_segments(reference, prediction, iou_threshold=iou_threshold)

            metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5, metric_col_6 = st.columns(6)
            metric_col_1.metric("TP", result["tp"])
            metric_col_2.metric("FP", result["fp"])
            metric_col_3.metric("FN", result["fn"])
            metric_col_4.metric("Precision", f"{result['precision']:.3f}")
            metric_col_5.metric("Recall", f"{result['recall']:.3f}")
            metric_col_6.metric("F1", f"{result['f1']:.3f}")

            matched_prediction_ids = {id(match["prediction"]) for match in result["matches"]}
            matched_reference_ids = {id(match["reference"]) for match in result["matches"]}

            match_rows = []
            for match in result["matches"]:
                pred = match["prediction"]
                ref = match["reference"]
                match_rows.append({
                    "IoU": round(match["iou"], 3),
                    "prediction": f"{format_eval_timestamp(pred['start'])}-{format_eval_timestamp(pred['end'])} {pred.get('label', '')}",
                    "reference": f"{format_eval_timestamp(ref['start'])}-{format_eval_timestamp(ref['end'])} {ref.get('label', '')}",
                })

            if match_rows:
                st.write("Совпавшие сегменты")
                st.dataframe(match_rows, use_container_width=True, hide_index=True)

            false_positive_rows = [
                {
                    "prediction": f"{format_eval_timestamp(item['start'])}-{format_eval_timestamp(item['end'])} {item.get('label', '')}"
                }
                for item in prediction
                if id(item) not in matched_prediction_ids
            ]

            false_negative_rows = [
                {
                    "reference": f"{format_eval_timestamp(item['start'])}-{format_eval_timestamp(item['end'])} {item.get('label', '')}"
                }
                for item in reference
                if id(item) not in matched_reference_ids
            ]

            if false_positive_rows:
                st.write("FP: лишние сегменты")
                st.dataframe(false_positive_rows, use_container_width=True, hide_index=True)

            if false_negative_rows:
                st.write("FN: пропущенные сегменты")
                st.dataframe(false_negative_rows, use_container_width=True, hide_index=True)

            if st.session_state.get("project_dir"):
                eval_json_path = Path(st.session_state.project_dir) / "timestamp_evaluation_report.json"
                eval_txt_path = Path(st.session_state.project_dir) / "timestamp_evaluation_report.txt"
                eval_report = {
                    "reference_label": reference_label,
                    "prediction_label": prediction_label,
                    "iou_threshold": iou_threshold,
                    "metrics": {
                        "tp": result["tp"],
                        "fp": result["fp"],
                        "fn": result["fn"],
                        "precision": result["precision"],
                        "recall": result["recall"],
                        "f1": result["f1"],
                    },
                    "quick_metrics": quick_rows,
                    "matches": match_rows,
                    "false_positives": false_positive_rows,
                    "false_negatives": false_negative_rows,
                }
                save_json(eval_json_path, eval_report)
                eval_lines = [
                    "F1 ОЦЕНКА ТАЙМИНГОВ",
                    "===================",
                    "",
                    f"Эталон: {reference_label}",
                    f"Сравниваем: {prediction_label}",
                    f"IoU threshold: {iou_threshold:.2f}",
                    "",
                    f"TP: {result['tp']}",
                    f"FP: {result['fp']}",
                    f"FN: {result['fn']}",
                    f"Precision: {result['precision']:.3f}",
                    f"Recall: {result['recall']:.3f}",
                    f"F1: {result['f1']:.3f}",
                    "",
                    "Быстрое сравнение:",
                ]
                for row in quick_rows:
                    eval_lines.append(
                        f"IoU {row['IoU']}: TP={row['TP']} FP={row['FP']} FN={row['FN']} "
                        f"Precision={row['Precision']} Recall={row['Recall']} F1={row['F1']}"
                    )
                eval_txt_path.write_text("\n".join(eval_lines), encoding="utf-8")
                st.caption(f"F1 отчет сохранен: `{eval_txt_path}`")


# =========================
# MAIN UI
# =========================

st.title("AI-инструмент для создания коротких клипов из видео")

st.caption(
    "Загрузка видео → распознавание речи → ИИ-сегментация глав или подтем → оценка пригодности → "
    "редактирование карточки → генерация субтитров → экспорт выбранного клипа."
)

with st.expander("Настройки ускорения", expanded=False):
    asr_hardware, asr_profiles = get_asr_profile_preview()
    first_asr_profile = asr_profiles[0] if asr_profiles else {}

    st.write(
        f"**ASR:** auto-tune `{'on' if ASR_AUTO_TUNE else 'off'}`, "
        f"запрос устройства `{ASR_DEVICE}`, downgrade модели `{'on' if ASR_ALLOW_MODEL_DOWNGRADE else 'off'}`"
    )

    if first_asr_profile:
        st.write(
            "**ASR первый профиль:** "
            f"`{first_asr_profile.get('model')}` · `{first_asr_profile.get('device')}` · "
            f"`{first_asr_profile.get('compute_type')}` · batch `{first_asr_profile.get('batch_size')}`"
        )

    if asr_hardware.get("gpu_available"):
        st.caption(
            f"GPU: {asr_hardware.get('gpu_name')} · "
            f"VRAM free/total: {asr_hardware.get('gpu_memory_free_mb')}/{asr_hardware.get('gpu_memory_total_mb')} MB · "
            f"RAM available/total: {asr_hardware.get('ram_available_mb')}/{asr_hardware.get('ram_total_mb')} MB"
        )
    else:
        gpu_error = asr_hardware.get("gpu_error") or "NVIDIA GPU не обнаружена"
        st.caption(f"GPU: {gpu_error}. ASR будет использовать CPU fallback.")

    st.write(
        f"**GPT4All:** запрос устройства `{GPT4ALL_DEVICE}` · "
        f"GPU backends `{', '.join(GPT4ALL_GPU_BACKENDS)}` · "
        f"min free VRAM `{GPT4ALL_MIN_FREE_VRAM_MB}` MB"
    )

    if LAST_GPT4ALL_RUNTIME:
        st.caption(
            "Последняя загрузка GPT4All: "
            f"attempt `{LAST_GPT4ALL_RUNTIME.get('attempted_device', 'default')}`, "
            f"backend `{LAST_GPT4ALL_RUNTIME.get('backend', 'unknown')}`, "
            f"device `{LAST_GPT4ALL_RUNTIME.get('device') or 'cpu/none'}`."
        )

    st.caption(
        f"ИИ-сегментация: окно `{TOPIC_MAX_SEGMENTS_PER_WINDOW}` сегм., overlap `{TOPIC_OVERLAP_SEGMENTS}`. "
        "Если GPU не подходит или backend не загрузился, GPT4All автоматически откатится на CPU."
    )
    st.write(f"**Экспорт:** NVENC {'включён' if USE_NVENC_FOR_EXPORT else 'выключен'} с fallback на libx264")
    st.caption(
        "Эти значения можно менять через переменные окружения без правки кода: "
        "ASR_DEVICE, ASR_AUTO_TUNE, ASR_ALLOW_MODEL_DOWNGRADE, FASTER_WHISPER_MODEL, "
        "FW_GPU_BATCH_SIZE, GPT4ALL_DEVICE, GPT4ALL_GPU_BACKENDS, GPT4ALL_MIN_FREE_VRAM_MB, USE_NVENC_FOR_EXPORT."
    )

available_model_paths = [str(path) for path in get_available_gpt4all_models()]
previous_model_path = st.session_state.get("gpt4all_model_path")

if available_model_paths:
    model_index = 0
    if previous_model_path in available_model_paths:
        model_index = available_model_paths.index(previous_model_path)

    selected_gpt4all_model_path = st.selectbox(
        "LLM-модель для GPT4All",
        options=available_model_paths,
        index=model_index,
        format_func=format_model_option,
        help="Положите новые .gguf модели в папку models, затем перезапустите Streamlit или обновите страницу."
    )
else:
    selected_gpt4all_model_path = str(GPT4ALL_MODEL_PATH)
    st.warning(f"В папке `{get_models_dir()}` не найдены .gguf модели. Текущий путь: `{selected_gpt4all_model_path}`")

model_changed = previous_model_path and previous_model_path != selected_gpt4all_model_path
st.session_state.gpt4all_model_path = selected_gpt4all_model_path

previous_segmentation_goal = st.session_state.get("segmentation_goal", "chapters")
segmentation_goal_label = st.radio(
    "Что ищем на этом проходе",
    options=["Главы видео", "Подтемы / Shorts"],
    index=0 if previous_segmentation_goal == "chapters" else 1,
    horizontal=True,
    help=(
        "Главы видео подходят для сравнения с авторскими или человеческими таймингами. "
        "Подтемы / Shorts — старый режим поиска коротких смысловых фрагментов."
    )
)
selected_segmentation_goal = "chapters" if segmentation_goal_label == "Главы видео" else "topics"

previous_chapter_sensitivity = st.session_state.get("chapter_sensitivity", CHAPTER_BOUNDARY_SENSITIVITY)
chapter_sensitivity_labels = {
    "detailed": "Подробные",
    "balanced": "Баланс",
    "coarse": "Крупные",
}
chapter_sensitivity_by_label = {label: key for key, label in chapter_sensitivity_labels.items()}

if selected_segmentation_goal == "chapters":
    previous_chapter_label = chapter_sensitivity_labels.get(previous_chapter_sensitivity, "Подробные")
    selected_chapter_label = st.radio(
        "Чувствительность глав",
        options=["Подробные", "Баланс", "Крупные"],
        index=["Подробные", "Баланс", "Крупные"].index(previous_chapter_label),
        horizontal=True,
        help=(
            "Подробные повышают Recall и чаще находят переходы. "
            "Крупные оставляют только сильные смены темы."
        )
    )
    selected_chapter_sensitivity = chapter_sensitivity_by_label[selected_chapter_label]
else:
    selected_chapter_sensitivity = previous_chapter_sensitivity

topic_mode_label = st.radio(
    "Режим ИИ-сегментации",
    options=["Быстрый", "Качественный"],
    index=0,
    horizontal=True,
    help=(
        "Быстрый режим просит GPT4All только найти границы, без title/summary для каждого окна. "
        "Качественный режим может генерировать title и summary сразу, но работает заметно дольше."
    )
)

selected_topic_mode = "fast" if topic_mode_label == "Быстрый" else "quality"
generate_topic_metadata = st.checkbox(
    "Генерировать title/summary сразу",
    value=False,
    help="Если выключено, GPT4All ищет только границы. Название и summary можно сгенерировать позже для выбранного фрагмента."
)

if selected_segmentation_goal == "chapters" and generate_topic_metadata:
    st.info("Для режима глав поиск границ всегда выполняется без title/summary. Описания можно сгенерировать после выбора главы.")

goal_changed = previous_segmentation_goal != selected_segmentation_goal
chapter_sensitivity_changed = previous_chapter_sensitivity != selected_chapter_sensitivity
mode_changed = st.session_state.get("topic_mode") and st.session_state.topic_mode != selected_topic_mode
metadata_changed = (
    "generate_topic_metadata" in st.session_state
    and st.session_state.generate_topic_metadata != generate_topic_metadata
)

if goal_changed or chapter_sensitivity_changed or mode_changed or metadata_changed or model_changed:
    for key in ["topic_segments", "selected_position", "generated_srt_by_clip", "exported_clip_by_clip", "preview_clip_by_clip"]:
        st.session_state.pop(key, None)

st.session_state.segmentation_goal = selected_segmentation_goal
st.session_state.chapter_sensitivity = selected_chapter_sensitivity
st.session_state.topic_mode = selected_topic_mode
st.session_state.generate_topic_metadata = generate_topic_metadata

if selected_segmentation_goal == "chapters":
    st.caption(
        f"Главы `{CHAPTERING_VERSION}`: ASR сжимается в блоки по `{CHAPTER_BLOCK_TARGET_SEC}` сек., "
        f"до `{CHAPTER_BLOCK_MAX_CHARS}` символов на блок. LLM ищет локальные границы в окнах "
        f"по `{CHAPTER_BOUNDARY_WINDOW_BLOCKS}` блоков, overlap `{CHAPTER_BOUNDARY_OVERLAP_BLOCKS}`. "
        "Этот режим используйте для F1 по таймингам глав."
    )
elif selected_topic_mode == "fast":
    st.caption(
        f"Быстрый режим: окно `{TOPIC_FAST_MAX_SEGMENTS_PER_WINDOW}` ASR-сегм., "
        f"overlap `{TOPIC_FAST_OVERLAP_SEGMENTS}`, max tokens `{GPT4ALL_FAST_TOPIC_MAX_TOKENS}`. "
        "Если title/summary не генерируются сразу, их можно сделать для выбранного клипа."
    )
else:
    st.caption(
        f"Качественный режим: окно `{TOPIC_MAX_SEGMENTS_PER_WINDOW}` ASR-сегм., "
        f"overlap `{TOPIC_OVERLAP_SEGMENTS}`, max tokens `{GPT4ALL_TOPIC_MAX_TOKENS}`."
    )

source_mode = st.radio(
    "Источник видео",
    options=["Локальный файл", "Путь к файлу", "YouTube / VK"],
    horizontal=True
)

if source_mode == "Локальный файл":
    uploaded_video = st.file_uploader(
        "Загрузите видеофайл",
        type=["mp4", "mov", "mkv", "avi", "webm", "m4v"],
        accept_multiple_files=False
    )

    if uploaded_video is not None:
        save_uploaded_video(uploaded_video)

elif source_mode == "Путь к файлу":
    local_path = st.text_input(
        "Путь к видеофайлу",
        placeholder="C:\\Videos\\example.mp4"
    )

    st.caption(
        "Для больших локальных файлов это лучше, чем загрузка через браузер: "
        "файл не копируется в outputs и не ограничивается upload-лимитом."
    )

    if st.button("Выбрать файл", type="primary", use_container_width=True):
        select_video_from_local_path(local_path)

else:
    video_url = st.text_input(
        "Ссылка на публичное видео",
        placeholder="https://www.youtube.com/watch?v=... или https://vk.com/video..."
    )

    st.caption(
        "Поддерживаются публичные видео YouTube, YouTube Shorts, VK и VK Video. "
        "Закрытые видео и авторизация пока не используются."
    )

    if st.button("Скачать видео", type="primary", use_container_width=True):
        download_video_from_link(video_url)

if st.session_state.get("video_path"):
    video_path = Path(st.session_state.video_path)
    project_dir = Path(st.session_state.project_dir)

    st.success(f"Видео загружено: {video_path.name}")

    if st.session_state.get("source_url"):
        st.caption(f"Источник: {st.session_state.source_url}")

    if st.session_state.get("source_path"):
        st.caption(f"Локальный путь: {st.session_state.source_path}")

    with st.expander("Предпросмотр исходного видео", expanded=False):
        st.video(str(video_path))

    start_col, info_col = st.columns([0.25, 0.75])

    with start_col:
        if st.button("Начать обработку", type="primary", use_container_width=True):
            try:
                process_video_pipeline(
                    video_path=video_path,
                    project_dir=project_dir
                )
            except Exception as e:
                st.error(f"Ошибка обработки: {e}")

    with info_col:
        st.info(
            "Во время обработки клипы физически не экспортируются. "
            "Сохраняются только промежуточные данные анализа. "
            "MP4-файл выбранного клипа создается только после нажатия кнопки «Экспортировать клип»."
        )

    render_pipeline_report(project_dir)

render_timestamp_evaluation()

if st.session_state.get("topic_segments"):
    render_workspace()
