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
    ASR_DEVICE,
    FASTER_WHISPER_MODEL,
    FW_GPU_BATCH_SIZE,
    GPT4ALL_DEVICE,
    GPT4ALL_FAST_TOPIC_MAX_TOKENS,
    GPT4ALL_TOPIC_MAX_TOKENS,
    TOPIC_FAST_MAX_SEGMENTS_PER_WINDOW,
    TOPIC_FAST_OVERLAP_SEGMENTS,
    TOPIC_MAX_SEGMENTS_PER_WINDOW,
    TOPIC_OVERLAP_SEGMENTS,
    USE_NVENC_FOR_EXPORT,
    analyze_topic_with_gpt4all_model,
    build_srt_for_clip,
    clean_text_for_topic_analysis,
    download_video_from_url,
    export_clip_with_optional_srt,
    format_timestamp,
    get_video_duration_sec,
    is_supported_video_url,
    llm_topic_segmentation,
    load_gpt4all_model,
    make_topic_label,
    read_json_if_exists,
    safe_filename,
    save_json,
    score_topic_segment_v2,
    transcribe_with_faster_whisper,
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

def build_timeline_html(segments, duration, selected_clip_id=None):
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

        title = html.escape(seg.get("title") or f"Клип {index + 1}")
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
        <div class="timeline-title">Таймлайн найденных смысловых фрагментов</div>
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


# =========================
# PIPELINE
# =========================

def process_video_pipeline(video_path: Path, project_dir: Path):
    progress_bar = st.progress(0)
    status_box = st.empty()

    status_box.info("Получаю длительность видео...")
    media_duration = get_video_duration_sec(str(video_path))
    st.session_state.media_duration = media_duration
    progress_bar.progress(0.04)

    asr_dir = project_dir / "asr" / "faster_whisper"

    def asr_progress(progress_value, message):
        progress_bar.progress(min(max(progress_value, 0.0), 0.65))
        status_box.info(message)

    fw_segments = transcribe_with_faster_whisper(
        video_path=str(video_path),
        out_dir=asr_dir,
        progress_callback=asr_progress
    )

    st.session_state.fw_segments = fw_segments

    if not fw_segments:
        st.session_state.topic_segments = []
        status_box.error("Речь не распознана или ASR-сегменты пустые.")
        progress_bar.progress(1.0)
        return

    topic_dir = project_dir / "topic_segmentation"
    topic_dir.mkdir(exist_ok=True, parents=True)

    topic_mode = st.session_state.get("topic_mode", "fast")
    is_fast_topic_mode = topic_mode == "fast"

    if is_fast_topic_mode:
        topic_mode_label = "быстрый"
        topic_max_segments = TOPIC_FAST_MAX_SEGMENTS_PER_WINDOW
        topic_overlap = TOPIC_FAST_OVERLAP_SEGMENTS
    else:
        topic_mode_label = "качественный"
        topic_max_segments = TOPIC_MAX_SEGMENTS_PER_WINDOW
        topic_overlap = TOPIC_OVERLAP_SEGMENTS

    generate_metadata = bool(st.session_state.get("generate_topic_metadata", False))
    metadata_label = "metadata" if generate_metadata else "bounds"
    topic_max_tokens = GPT4ALL_TOPIC_MAX_TOKENS if generate_metadata else GPT4ALL_FAST_TOPIC_MAX_TOKENS
    topic_cache_path = topic_dir / f"topic_segments_{topic_mode}_{metadata_label}.json"

    cached_topics = read_json_if_exists(str(topic_cache_path))

    if isinstance(cached_topics, list) and cached_topics:
        save_json(topic_dir / "topic_segments.json", cached_topics)
        st.session_state.topic_segments = cached_topics
        st.session_state.selected_position = 0
        progress_bar.progress(1.0)
        status_box.success(f"Использую кэш ИИ-сегментации ({topic_mode_label} режим).")
        return

    status_box.info(f"Выделяю смысловые подтемы через GPT4All ({topic_mode_label} режим, {metadata_label})...")
    progress_bar.progress(0.68)

    if GPT4All is None:
        st.session_state.topic_segments = []
        status_box.error("gpt4all не установлен. Невозможно выполнить ИИ-сегментацию подтем.")
        progress_bar.progress(1.0)
        return

    if not Path(GPT4ALL_MODEL_PATH).exists():
        st.session_state.topic_segments = []
        status_box.error(f"Модель GPT4All не найдена: {GPT4ALL_MODEL_PATH}")
        progress_bar.progress(1.0)
        return

    try:
        model = load_gpt4all_model(n_ctx=4096)
    except Exception as e:
        st.session_state.topic_segments = []
        status_box.error(f"Не удалось загрузить GPT4All: {e}")
        progress_bar.progress(1.0)
        return

    def topic_progress(done, total, message):
        local_progress = done / total if total else 1.0
        progress_bar.progress(0.68 + min(local_progress, 1.0) * 0.29)
        status_box.info(message)

    try:
        enriched = llm_topic_segmentation(
            whisper_segments=fw_segments,
            model=model,
            max_segments_per_window=topic_max_segments,
            overlap_segments=topic_overlap,
            min_topic_duration_sec=8.0,
            max_topic_duration_sec=240.0,
            fast_mode=is_fast_topic_mode,
            generate_metadata=generate_metadata,
            topic_max_tokens=topic_max_tokens,
            progress_callback=topic_progress
        )

    except Exception as e:
        st.session_state.topic_segments = []
        status_box.error(f"Ошибка ИИ-сегментации подтем: {e}")
        progress_bar.progress(1.0)
        return

    if not enriched:
        st.session_state.topic_segments = []
        status_box.warning(
            "ИИ не нашёл подходящие смысловые подтемы. "
            "Обычно это значит, что GPT4All не смог корректно вернуть JSON или контекст модели всё ещё маловат."
        )
        progress_bar.progress(1.0)
        return

    save_json(topic_cache_path, enriched)
    save_json(topic_dir / "topic_segments.json", enriched)
    save_json(
        topic_dir / "topic_segmentation_settings.json",
        {
            "mode": topic_mode,
            "mode_label": topic_mode_label,
            "max_segments_per_window": topic_max_segments,
            "overlap_segments": topic_overlap,
            "max_tokens": topic_max_tokens,
            "fast_mode": is_fast_topic_mode,
            "generate_metadata": generate_metadata,
        }
    )

    st.session_state.topic_segments = enriched
    st.session_state.selected_position = 0

    progress_bar.progress(1.0)
    status_box.success("Обработка завершена.")


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


def generate_srt_for_clip(seg, fw_segments, project_dir: Path):
    out_dir = project_dir / "selected_subtitles"
    out_dir.mkdir(exist_ok=True, parents=True)

    clip_id = seg["id"]
    title = ui_safe_filename(seg.get("title") or f"clip_{clip_id}")
    start = float(seg["start"])
    end = float(seg["end"])

    srt_path = out_dir / f"clip_{clip_id:03d}_{title}.srt"

    build_srt_for_clip(
        whisper_segments=fw_segments,
        clip_start=start,
        clip_end=end,
        srt_path=srt_path
    )

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
    st.session_state.setdefault("generated_srt_by_clip", {}).pop(clip_key, None)
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


# =========================
# WORKSPACE
# =========================

def render_workspace():
    segments = st.session_state.get("topic_segments", [])
    fw_segments = st.session_state.get("fw_segments", [])
    media_duration = st.session_state.get("media_duration", 0)
    st.session_state.setdefault("generated_srt_by_clip", {})
    st.session_state.setdefault("exported_clip_by_clip", {})
    st.session_state.setdefault("preview_clip_by_clip", {})

    if not segments:
        st.warning("Пока нет найденных клипов. Сначала запусти обработку видео.")
        return

    st.divider()

    left_col, right_col = st.columns([0.32, 0.68], gap="large")

    with left_col:
        st.subheader("Найденные фрагменты")

        def format_clip_label(i):
            seg = segments[i]
            score = float(seg.get("score", 0) or 0)
            title = seg.get("title") or f"Клип {seg.get('id', i + 1)}"
            start = format_timestamp(seg.get("start", 0))
            end = format_timestamp(seg.get("end", 0))

            if len(title) > 42:
                title = title[:42] + "..."

            return f"{i + 1}. score {score:.2f} · {start}-{end} · {title}"

        selected_position = st.radio(
            label="Кандидаты отсортированы по убыванию score",
            options=list(range(len(segments))),
            index=min(st.session_state.get("selected_position", 0), len(segments) - 1),
            format_func=format_clip_label,
            label_visibility="visible"
        )

        st.session_state.selected_position = selected_position

        if st.button("Пересортировать по score", use_container_width=True):
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
                selected_clip_id=selected_clip_id
            ),
            height=150
        )

        title_key, summary_key, text_key, start_key, end_key, range_key = ensure_clip_widget_defaults(selected_seg)

        st.subheader("Подробная информация о фрагменте")

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

        with st.expander("Редактирование границ клипа", expanded=True):
            max_slider_value = max(float(media_duration or 0), float(selected_seg.get("end", 0) or 0), 1.0)
            current_start = max(0.0, min(float(selected_seg.get("start", 0) or 0), max_slider_value))
            current_end = max(current_start + 0.1, min(float(selected_seg.get("end", current_start + 1) or current_start + 1), max_slider_value))

            range_start, range_end = st.session_state.get(range_key, (current_start, current_end))
            range_start = max(0.0, min(float(range_start), max_slider_value))
            range_end = max(range_start + 0.1, min(float(range_end), max_slider_value))
            st.session_state[range_key] = (range_start, range_end)

            st.slider(
                "Границы клипа, сек.",
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

            if GPT4All is None:
                st.error("gpt4all не установлен.")
            elif not Path(GPT4ALL_MODEL_PATH).exists():
                st.error(f"Модель GPT4All не найдена: {GPT4ALL_MODEL_PATH}")
            else:
                with st.spinner("Перегенерирую описание текущего клипа..."):
                    try:
                        model = load_gpt4all_model(n_ctx=4096)
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

                st.session_state[title_key] = selected_seg["title"] or ""
                st.session_state[summary_key] = selected_seg["summary"] or ""

                st.session_state.topic_segments[selected_position] = selected_seg
                st.rerun()

        st.text_input(
            "Название клипа",
            key=title_key
        )

        st.text_area(
            "Summary клипа",
            key=summary_key,
            height=100
        )

        st.text_area(
            "Текст клипа / что говорится во фрагменте",
            key=text_key,
            height=220
        )

        with st.expander("Настройки предпросмотра и экспорта", expanded=False):
            aspect_key = f"aspect_{selected_clip_id}"
            subtitle_mode_key = f"subtitle_mode_{selected_clip_id}"
            subtitle_position_key = f"subtitle_position_{selected_clip_id}"
            subtitle_size_key = f"subtitle_size_{selected_clip_id}"

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

            if st.session_state.get(subtitle_mode_key) == "Вшить в видео":
                sub_col_1, sub_col_2 = st.columns(2)

                with sub_col_1:
                    st.selectbox(
                        "Положение субтитров",
                        options=["Снизу", "Выше снизу", "По центру"],
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

        def get_export_settings():
            aspect_mode = "vertical_9_16" if st.session_state.get(f"aspect_{selected_clip_id}") == "Вертикальный 9:16" else "original"
            subtitle_mode = st.session_state.get(f"subtitle_mode_{selected_clip_id}", "Отдельный SRT рядом")
            subtitle_position_label = st.session_state.get(f"subtitle_position_{selected_clip_id}", "Снизу")
            subtitle_position = {
                "Снизу": "bottom",
                "Выше снизу": "upper_bottom",
                "По центру": "center",
            }.get(subtitle_position_label, "bottom")
            subtitle_size = int(st.session_state.get(f"subtitle_size_{selected_clip_id}", 28))

            return aspect_mode, subtitle_mode, subtitle_position, subtitle_size

        def ensure_srt_if_needed(seg, subtitle_mode):
            if subtitle_mode == "Не добавлять":
                return None

            saved_srt = st.session_state.generated_srt_by_clip.get(str(selected_clip_id))

            if saved_srt and Path(saved_srt).exists():
                return Path(saved_srt)

            srt_path = generate_srt_for_clip(
                seg=seg,
                fw_segments=fw_segments,
                project_dir=Path(st.session_state.project_dir)
            )
            st.session_state.generated_srt_by_clip[str(selected_clip_id)] = str(srt_path)
            return srt_path

        action_col_1, action_col_2, action_col_3, action_col_4 = st.columns(4)

        with action_col_1:
            if st.button("Применить правки", use_container_width=True):
                apply_widget_values_to_segment(selected_position)
                st.success("Правки сохранены в карточку клипа.")

        with action_col_2:
            if st.button("Сгенерировать SRT", use_container_width=True):
                seg = apply_widget_values_to_segment(selected_position)

                with st.spinner("Генерирую SRT по таймкодам faster-whisper..."):
                    srt_path = generate_srt_for_clip(
                        seg=seg,
                        fw_segments=fw_segments,
                        project_dir=Path(st.session_state.project_dir)
                    )

                st.session_state.generated_srt_by_clip[str(selected_clip_id)] = str(srt_path)
                st.success(f"SRT создан: {srt_path}")

        with action_col_3:
            if st.button("Собрать предпросмотр", use_container_width=True):
                seg = apply_widget_values_to_segment(selected_position)
                aspect_mode, subtitle_mode, subtitle_position, subtitle_size = get_export_settings()

                with st.spinner("Собираю предпросмотр клипа..."):
                    try:
                        srt_path = ensure_srt_if_needed(seg, subtitle_mode) if subtitle_mode == "Вшить в видео" else None
                        preview_path = export_clip_with_optional_srt(
                            video_path=Path(st.session_state.video_path),
                            seg=seg,
                            out_dir=Path(st.session_state.project_dir) / "previews",
                            srt_path=srt_path,
                            burn_subtitles=subtitle_mode == "Вшить в видео",
                            aspect_mode=aspect_mode,
                            subtitle_position=subtitle_position,
                            subtitle_font_size=subtitle_size
                        )
                        st.session_state.preview_clip_by_clip[str(selected_clip_id)] = str(preview_path)
                        st.success(f"Предпросмотр создан: {preview_path}")
                    except Exception as e:
                        st.error(f"Не удалось собрать предпросмотр: {e}")

        with action_col_4:
            if st.button("Экспортировать клип", type="primary", use_container_width=True):
                seg = apply_widget_values_to_segment(selected_position)
                aspect_mode, subtitle_mode, subtitle_position, subtitle_size = get_export_settings()

                with st.spinner("Экспортирую выбранный клип..."):
                    try:
                        srt_path = ensure_srt_if_needed(seg, subtitle_mode)
                        clip_path = export_clip_with_optional_srt(
                            video_path=Path(st.session_state.video_path),
                            seg=seg,
                            out_dir=Path(st.session_state.project_dir) / "exports",
                            srt_path=srt_path,
                            burn_subtitles=subtitle_mode == "Вшить в видео",
                            aspect_mode=aspect_mode,
                            subtitle_position=subtitle_position,
                            subtitle_font_size=subtitle_size
                        )

                        st.session_state.exported_clip_by_clip[str(selected_clip_id)] = str(clip_path)
                        st.success(f"Клип экспортирован: {clip_path}")

                    except Exception as e:
                        st.error(f"Не удалось экспортировать клип: {e}")

        current_srt = st.session_state.generated_srt_by_clip.get(str(selected_clip_id))
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

    for seg in st.session_state.get("topic_segments", []) or []:
        start = float(seg.get("start", 0) or 0)
        end = float(seg.get("end", start) or start)

        if end <= start:
            continue

        intervals.append({
            "start": start,
            "end": end,
            "label": seg.get("title") or f"Клип {seg.get('id', len(intervals) + 1)}",
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
    with st.expander("Оценка таймингов / F1-score", expanded=False):
        st.caption(
            "Загрузите или вставьте тайминги, выберите эталон и сравниваемый блок. "
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
            sources.append({
                "label": format_interval_for_option(len(sources), model_intervals, "Текущие сегменты модели"),
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


# =========================
# MAIN UI
# =========================

st.title("AI-инструмент для создания коротких клипов из видео")

st.caption(
    "Загрузка видео → распознавание речи → ИИ-сегментация подтем → оценка пригодности → "
    "редактирование карточки → генерация субтитров → экспорт выбранного клипа."
)

with st.expander("Настройки ускорения", expanded=False):
    st.write(
        f"**ASR:** faster-whisper `{FASTER_WHISPER_MODEL}`, устройство `{ASR_DEVICE}`, "
        f"GPU batch `{FW_GPU_BATCH_SIZE}`"
    )
    st.write(
        f"**GPT4All:** устройство `{GPT4ALL_DEVICE}` · "
        f"окно сегментации `{TOPIC_MAX_SEGMENTS_PER_WINDOW}` сегм., overlap `{TOPIC_OVERLAP_SEGMENTS}`"
    )
    st.write(f"**Экспорт:** NVENC {'включён' if USE_NVENC_FOR_EXPORT else 'выключен'} с fallback на libx264")
    st.caption(
        "Эти значения можно менять через переменные окружения без правки кода: "
        "ASR_DEVICE, FW_GPU_BATCH_SIZE, GPT4ALL_DEVICE, USE_NVENC_FOR_EXPORT."
    )

topic_mode_label = st.radio(
    "Режим ИИ-сегментации",
    options=["Быстрый", "Качественный"],
    index=0,
    horizontal=True,
    help=(
        "Быстрый режим просит GPT4All только найти границы подтем, без title/summary для каждого окна. "
        "Качественный режим генерирует title и summary сразу, но работает заметно дольше."
    )
)

selected_topic_mode = "fast" if topic_mode_label == "Быстрый" else "quality"
generate_topic_metadata = st.checkbox(
    "Генерировать title/summary сразу",
    value=False,
    help="Если выключено, GPT4All ищет только границы подтем. Название и summary можно сгенерировать позже для выбранного клипа."
)

mode_changed = st.session_state.get("topic_mode") and st.session_state.topic_mode != selected_topic_mode
metadata_changed = (
    "generate_topic_metadata" in st.session_state
    and st.session_state.generate_topic_metadata != generate_topic_metadata
)

if mode_changed or metadata_changed:
    for key in ["topic_segments", "selected_position", "generated_srt_by_clip", "exported_clip_by_clip", "preview_clip_by_clip"]:
        st.session_state.pop(key, None)

st.session_state.topic_mode = selected_topic_mode
st.session_state.generate_topic_metadata = generate_topic_metadata

if selected_topic_mode == "fast":
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

render_timestamp_evaluation()

if st.session_state.get("topic_segments"):
    render_workspace()
