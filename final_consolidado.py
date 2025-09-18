from pathlib import Path
import sys, os, time, cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_yaml, yolo_class_names
from src.vision.detector import Detector
from src.vision.counter import LineCounter
from src.utils.dib import dib_cajas, dib_lineas, dib_panel_cont
from src.control.cont_heuristico import cont_heuristico, ControlParams


def open_video(abs_path_or_cam):
    if str(abs_path_or_cam) == "0":
        return cv2.VideoCapture(0)
    for be in [cv2.CAP_FFMPEG, cv2.CAP_MSMF, cv2.CAP_DSHOW, 0]:
        cap = cv2.VideoCapture(str(abs_path_or_cam), be)
        if cap.isOpened():
            print(f"[OK] Abierto con backend={be} -> {abs_path_or_cam}")
            return cap
    return None


def main():
    app = load_yaml(ROOT / "config" / "app.yaml")
    data_yaml = ROOT / "config" / "data.yaml"
    class_names = yolo_class_names(data_yaml) if data_yaml.exists() else None

    HEADLESS = os.getenv("HEADLESS", "0") == "1"
    if os.getenv("VIDEO_SOURCE"):
        app["video_source"] = os.getenv("VIDEO_SOURCE")

    WEIGHTS = ROOT / "models" / "yolo12" / "modelo_final" / "weights" / "best.pt"

    det = Detector(
        weights=str(WEIGHTS),
        conf=app.get("conf", 0.3),
        iou=app.get("iou", 0.5),
        tracker_cfg=app.get("tracker_cfg"),
        classes=app.get("classes_to_track"),
    )

    lc_cfg = app.get("line_counting", {}) or {}
    lanes_path = (ROOT / lc_cfg.get("lanes_config", "config/lanes.example.json")).resolve()
    counter = LineCounter(str(lanes_path))

    ctrl_cfg = app.get("control", {}) or {}
    params = ControlParams(
        min_green=ctrl_cfg.get("min_green", 12),
        max_green=ctrl_cfg.get("max_green", 60),
        green_extension=ctrl_cfg.get("green_extension", 3),
        clearance=ctrl_cfg.get("clearance", 4),
        thresholds=ctrl_cfg.get("density_thresholds", {"low": 6, "medium": 12, "high": 20}),
        class_weights=ctrl_cfg.get(
            "class_weights",
            {
                "car": 1.0,
                "motorbike": 0.6,
                "bus": 2.5,
                "truck": 2.0,
                "van": 1.2,
                "bicycle": 0.3,
                "rickshaw": 0.8,
            },
        ),
        phase_change_penalty=float(ctrl_cfg.get("phase_change_penalty", 0.0)),
    )
    controller = cont_heuristico(params)

    rojo_fijo = int(ctrl_cfg.get("fixed_red_time", 10))

    src = app.get("video_source", "0")
    if str(src) == "0":
        video_path = "0"
    else:
        video_path = str(src)
        if not os.path.isabs(video_path):
            video_path = str((ROOT / video_path).resolve())

    cap = open_video(video_path)
    if cap is None or not cap.isOpened():
        raise RuntimeError(f"Error apertura de video: {video_path}")

    writer = None
    out_dir = Path(os.getenv("OUT_DIR", "/outputs"))
    if HEADLESS:
        if not out_dir.exists():
            out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "out.mp4"

    active_phase = None          
    verde_fin_ts = 0.0
    amarillo_fin_ts = 0.0
    rojo_fin_ts = 0.0

    estado = "VERDE"

    tiempos = {"verde": 0.0, "amarillo": 0.0, "rojo": 0.0}
    t0_sesion = time.time()
    last_ts = t0_sesion

    def decidir_con_vpm(rates_vpm, detail_by_approach, live_counts):
        nonlocal active_phase, verde_fin_ts, amarillo_fin_ts
        plan = controller.decide(
            rates_vpm,
            detail_by_approach=detail_by_approach,
            live_counts=live_counts,
        )
        active_phase = plan["next_phase"] if plan["next_phase"] else "Principal"
        greens = plan["greens"]
        verde_duracion = int(greens.get(active_phase, params.min_green))
        now2 = time.time()
        verde_fin_ts = now2 + verde_duracion
        amarillo_fin_ts = verde_fin_ts + params.clearance
        print(
            f"[DECIDIR] vpm={rates_vpm} verdes={greens} "
            f"fase={active_phase} VERDE={verde_duracion}s AMARILLO={params.clearance}s"
        )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        dt = now - last_ts
        last_ts = now

        results = det.infer(frame)
        frame = dib_cajas(frame, results, class_names)

        info = counter.update(results, class_names)
        rates_vpm = info.get("veh_per_min", {})
        detail_by_approach = info.get("by_approach", {})
        live_counts = info.get("live_counts", {})

        if not rates_vpm:
            total_now = float(sum(live_counts.values()))
            rates_vpm = {"Principal": total_now}

        if active_phase is None:
            estado = "VERDE"
            decidir_con_vpm(rates_vpm, detail_by_approach, live_counts)

        if estado == "VERDE":
            tiempos["verde"] += dt
            if now >= verde_fin_ts:
                estado = "AMARILLO"

        elif estado == "AMARILLO":
            tiempos["amarillo"] += dt
            if now >= amarillo_fin_ts:
                estado = "ROJO"
                rojo_fin_ts = now + rojo_fijo

        elif estado == "ROJO":
            tiempos["rojo"] += dt
            if now >= rojo_fin_ts:
                estado = "VERDE"
                decidir_con_vpm(rates_vpm, detail_by_approach, live_counts)

        frame = dib_lineas(frame, counter.lines_for_draw())
        frame = dib_panel_cont(frame, live_counts, origin=(15, 30))
        try:
            rates_overlay = {f"{ap} vpm": f"{v:.1f}" for ap, v in rates_vpm.items()}
            frame = dib_panel_cont(frame, rates_overlay, origin=(15, 120))
        except Exception:
            pass

        if estado == "VERDE":
            t_restante = max(0, int(verde_fin_ts - now))
            texto_fase = f"Fase: {active_phase} | Verde | Tiempo restante: {t_restante}s"
        elif estado == "AMARILLO":
            t_restante = max(0, int(amarillo_fin_ts - now))
            texto_fase = f"Fase: {active_phase} | Amarillo | Tiempo restante: {t_restante}s"
        else:  
            t_restante = max(0, int(rojo_fin_ts - now))
            texto_fase = f"Fase: {active_phase} | Rojo | Tiempo restante: {t_restante}s"

        cv2.putText(
            frame,
            texto_fase,
            (15, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        try:
            flujo_total = int(sum(rates_vpm.values()))
            resumen = {
                "flujo": flujo_total,
                "verde": int(tiempos["verde"]),
                "amarillo": int(tiempos["amarillo"]),
                "rojo": int(tiempos["rojo"]),
            }
            frame = dib_panel_cont(frame, resumen, origin=(15, 200))
        except Exception:
            pass

        if HEADLESS:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = frame.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 25.0
                writer = cv2.VideoWriter(str(out_file), fourcc, float(fps), (w, h))
                print(f"[HEADLESS] Guardando salida en: {out_file} @ {fps}fps")
            writer.write(frame)
        else:
            cv2.imshow("Control Semaforo", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    if writer is not None:
        writer.release()
    if not HEADLESS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
