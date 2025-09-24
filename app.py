# app.py  (Streamlit UI)
import tempfile, time, os
from pathlib import Path
import streamlit as st
import numpy as np
import cv2

from final_consolidado import run_stream

st.set_page_config(page_title="Semaforización Inteligente", layout="wide")

# --- Sidebar: fuente de video y opciones ---
st.sidebar.title("Fuente de video")
option = st.sidebar.radio("¿Qué quieres usar?", ["Subir archivo", "Webcam (0)", "RTSP URL"])

video_source = None
uploaded_file = None

if option == "Subir archivo":
    uploaded_file = st.sidebar.file_uploader("Sube un video (.mp4, .avi)", type=["mp4", "avi", "mov", "mkv"])
elif option == "Webcam (0)":
    video_source = "0"
else:
    video_source = st.sidebar.text_input("rtsp://usuario:pass@host:554/...", "")

st.sidebar.title("Parámetros de Control")
min_green = st.sidebar.slider("Verde mínimo (s)", 5, 60, 12, 1)
max_green = st.sidebar.slider("Verde máximo (s)", 10, 120, 60, 1)
clearance = st.sidebar.slider("Amarillo/Clearance (s)", 2, 10, 4, 1)
fixed_red = st.sidebar.slider("Rojo fijo (s)", 5, 30, 10, 1)
phase_penalty = st.sidebar.number_input("Penalización cambio de fase", value=0.0, step=0.5)

col_start, col_save = st.sidebar.columns(2)
start_btn = col_start.button("▶️ Iniciar")
stop_btn = col_save.button("⏹️ Detener")
save_output = st.sidebar.checkbox("Guardar video procesado", value=False)

# --- Tabs ---
tab_det, tab_counts, tab_params, tab_logs = st.tabs(["Detecciones", "Conteos y Tasas", "Parámetros activos", "Logs"])

with tab_det:
    st.header("Detección + Fases (en vivo)")
    video_placeholder = st.empty()
    info_placeholder = st.empty()
with tab_counts:
    st.header("Conteos y tasas")
    counts_placeholder = st.empty()
    rate_placeholder = st.empty()
with tab_params:
    st.header("Parámetros activos")
    params_placeholder = st.empty()
with tab_logs:
    st.header("Logs")
    logs_area = st.empty()

# --- Estado de sesión ---
if "running" not in st.session_state:
    st.session_state.running = False
if "tmpfile" not in st.session_state:
    st.session_state.tmpfile = None

# --- Preparar fuente si es upload ---
def prepare_uploaded(uploaded_file):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name  # ruta local temporal

# --- Botón iniciar ---
if start_btn and not st.session_state.running:
    # resolver source
    if option == "Subir archivo":
        if uploaded_file is None:
            st.warning("Sube un video primero.")
        else:
            local_path = prepare_uploaded(uploaded_file)
            video_source = local_path
    elif option == "RTSP URL":
        if not video_source:
            st.warning("Ingresa la URL RTSP.")
    # construir overrides de control
    ctrl_over = {
        "min_green": int(min_green),
        "max_green": int(max_green),
        "clearance": int(clearance),
        "fixed_red_time": int(fixed_red),
        "phase_change_penalty": float(phase_penalty),
    }
    # path de guardado si aplica
    save_path = None
    if save_output:
        out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True, parents=True)
        ts = int(time.time())
        save_path = str(out_dir / f"streamlit_out_{ts}.mp4")
    # guardar en estado y arrancar
    st.session_state.running = True
    st.session_state.gen = run_stream(
        video_source=video_source,
        headless=True,
        save_path=save_path,
        ctrl_overrides=ctrl_over,
        app_overrides=None,
    )
    logs_area.write("⏱️ Procesamiento iniciado.")

# --- Botón detener ---
if stop_btn and st.session_state.running:
    st.session_state.running = False
    st.session_state.gen = None
    logs_area.write("⏹️ Procesamiento detenido por el usuario.")

# --- Loop de render mientras running ---
if st.session_state.running:
    try:
        for frame_bgr, stats in st.session_state.gen:
            # mostrar detección
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            # info rápida
            info_placeholder.write(
                f"**Fase:** {stats['fase']} | **Estado:** {stats['estado']} | **Tiempo restante:** {stats['t_restante']}s"
            )

            # conteos
            live_counts = stats.get("live_counts", {})
            counts_placeholder.write(live_counts)

            # tasas
            rates = {k: f"{v:.1f} vpm" for k, v in stats.get("rates_vpm", {}).items()}
            rate_placeholder.write(rates)

            # params activos
            params_placeholder.write({
                "min_green": min_green,
                "max_green": max_green,
                "clearance": clearance,
                "fixed_red_time": fixed_red,
                "phase_change_penalty": phase_penalty,
            })

            # pequeño sleep para no saturar UI
            time.sleep(0.01)

            # permitir parar
            if not st.session_state.running:
                break
    except Exception as e:
        st.error(f"Error en el stream: {e}")
        st.session_state.running = False
        st.session_state.gen = None
        logs_area.write("❌ Error: ver detalles arriba.")

# Si se guardó salida, ofrecer descarga
if save_output and not st.session_state.running:
    outs = sorted(Path("outputs").glob("streamlit_out_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if outs:
        latest = outs[0]
        with open(latest, "rb") as f:
            st.download_button("⬇️ Descargar último procesado", f, file_name=latest.name, mime="video/mp4")
