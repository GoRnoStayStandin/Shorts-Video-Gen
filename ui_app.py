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
    analyze_topic_with_gpt4all_model,
    build_srt_for_clip,
    clean_text_for_topic_analysis,
    export_clip_with_optional_srt,
    format_timestamp,
    get_video_duration_sec,
    llm_topic_segmentation,
    load_gpt4all_model,
    make_topic_label,
    safe_filename,
    save_json,
    transcribe_with_faster_whisper,
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
UI_PROJECTS_DIR = OUTPUT_DIR / "ui_projects"

UI_UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
UI_PROJECTS_DIR.mkdir(exist_ok=True, parents=True)


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
    ]

    for key in keys_to_remove:
        st.session_state.pop(key, None)

    for key in list(st.session_state.keys()):
        if key.startswith("title_") or key.startswith("summary_") or key.startswith("text_"):
            st.session_state.pop(key, None)


def save_uploaded_video(uploaded_file):
    signature = f"{uploaded_file.name}:{uploaded_file.size}"

    if st.session_state.get("upload_signature") == signature:
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
    st.session_state.generated_srt_by_clip = {}
    st.session_state.exported_clip_by_clip = {}


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

    status_box.info("Выделяю смысловые подтемы через GPT4All...")
    progress_bar.progress(0.68)

    topic_dir = project_dir / "topic_segmentation"
    topic_dir.mkdir(exist_ok=True, parents=True)

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
            max_segments_per_window=8,
            overlap_segments=2,
            min_topic_duration_sec=8.0,
            max_topic_duration_sec=240.0,
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

    save_json(topic_dir / "topic_segments.json", enriched)

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

    if title_key not in st.session_state:
        st.session_state[title_key] = seg.get("title") or ""

    if summary_key not in st.session_state:
        st.session_state[summary_key] = seg.get("summary") or ""

    if text_key not in st.session_state:
        st.session_state[text_key] = seg.get("text") or ""

    return title_key, summary_key, text_key


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


# =========================
# WORKSPACE
# =========================

def render_workspace():
    segments = st.session_state.get("topic_segments", [])
    fw_segments = st.session_state.get("fw_segments", [])
    media_duration = st.session_state.get("media_duration", 0)

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

        title_key, summary_key, text_key = ensure_clip_widget_defaults(selected_seg)

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
            if st.button("16:9 → 9:16", use_container_width=True):
                st.info(
                    "Пока это заглушка. Позже сюда можно добавить вертикальное кадрирование: "
                    "center crop, blurred background или AI-tracking лица/объекта."
                )

        with action_col_4:
            if st.button("Экспортировать клип", type="primary", use_container_width=True):
                seg = apply_widget_values_to_segment(selected_position)

                srt_path: Optional[Path] = None
                saved_srt = st.session_state.generated_srt_by_clip.get(str(selected_clip_id))

                if saved_srt and Path(saved_srt).exists():
                    srt_path = Path(saved_srt)

                with st.spinner("Экспортирую выбранный клип..."):
                    try:
                        clip_path = export_clip_with_optional_srt(
                            video_path=Path(st.session_state.video_path),
                            seg=seg,
                            out_dir=Path(st.session_state.project_dir) / "exports",
                            srt_path=srt_path,
                            burn_subtitles=False
                        )

                        st.session_state.exported_clip_by_clip[str(selected_clip_id)] = str(clip_path)
                        st.success(f"Клип экспортирован: {clip_path}")

                    except Exception as e:
                        st.error(f"Не удалось экспортировать клип: {e}")

        current_srt = st.session_state.generated_srt_by_clip.get(str(selected_clip_id))
        current_export = st.session_state.exported_clip_by_clip.get(str(selected_clip_id))

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


# =========================
# MAIN UI
# =========================

st.title("AI-инструмент для создания коротких клипов из видео")

st.caption(
    "Загрузка видео → распознавание речи → ИИ-сегментация подтем → оценка пригодности → "
    "редактирование карточки → генерация субтитров → экспорт выбранного клипа."
)

upload_col, future_col = st.columns([0.58, 0.42], gap="large")

with upload_col:
    uploaded_video = st.file_uploader(
        "Загрузите видеофайл",
        type=["mp4", "mov", "mkv", "avi", "webm"],
        accept_multiple_files=False
    )

with future_col:
    st.text_input(
        "Загрузка по ссылке",
        placeholder="VK / RuTube / YouTube — будет добавлено позже",
        disabled=True
    )

    st.caption(
        "Сейчас работает загрузка локального файла. "
        "Поддержку ссылок лучше добавить отдельным модулем загрузки."
    )

if uploaded_video is not None:
    save_uploaded_video(uploaded_video)

if st.session_state.get("video_path"):
    video_path = Path(st.session_state.video_path)
    project_dir = Path(st.session_state.project_dir)

    st.success(f"Видео загружено: {video_path.name}")

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

if st.session_state.get("topic_segments"):
    render_workspace()
