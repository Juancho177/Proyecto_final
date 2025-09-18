import cv2, time, json
from pathlib import Path
from src.vision.detector import Detector
from src.vision.counter import LineCounter
from src.utils.dib import dib_cajas, dib_lineas, dib_panel_cont
from src.utils.io import yolo_class_names, load_yaml

ROOT = Path(__file__).resolve().parent

WEIGHTS = ROOT / "models" / "yolo12" / "modelo_final" / "weights" / "best.pt"

def adjust_count_line_y(cap, lanes_path: Path, y_pixels: int | None, y_ratio: float | None):
  
    ok, sample = cap.read()
    if not ok:
        print("No se pudo leer el frame")
        return
    H, W = sample.shape[:2]
    # calcula Y objetivo
    if y_pixels is not None:
        y = max(0, min(H - 1, int(y_pixels)))
    else:
        y_ratio = 0.25 if (y_ratio is None) else float(y_ratio)
        y = max(0, min(H - 1, int(H * y_ratio)))

    try:
        cfg = json.loads(lanes_path.read_text(encoding="utf-8"))
        changed = False
        for ap in cfg.get("approaches", []):
            for ln in ap.get("count_lines", []):
                new_p1 = [0, y]
                new_p2 = [max(0, W - 1), y]
                if ln.get("p1") != new_p1 or ln.get("p2") != new_p2:
                    ln["p1"], ln["p2"] = new_p1, new_p2
                    changed = True
        if changed:
            lanes_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            print(f"[INFO] Línea ajustada a Y={y} y ancho W={W} en {lanes_path.name}.")
        else:
            print(f"[INFO] La línea ya estaba en el ancho. Y={y}, W={W}.")
    finally:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 

def main():
    app = load_yaml(ROOT/"config"/"app.yaml")
    data_yaml = ROOT/"config"/"data.yaml"
    class_names = yolo_class_names(data_yaml) if data_yaml.exists() else None

    det = Detector(
        weights=str(WEIGHTS),
        conf=app["conf"],
        iou=app["iou"],
        tracker_cfg=app.get("tracker_cfg"),
        classes=app.get("classes_to_track")
    )
    src = app["video_source"]
    cap = cv2.VideoCapture(0 if str(src)=="0" else str(src))
    if not cap.isOpened():
        raise RuntimeError(f"No puedo abrir el video: {src}")

    lc_cfg = app.get("line_counting", {}) or {}
    lanes_path = (ROOT / lc_cfg.get("lanes_config", "config/lanes.example.json")).resolve()
    if lc_cfg.get("enabled", True) and lanes_path.exists():
        adjust_count_line_y(
            cap,
            lanes_path,
            y_pixels=lc_cfg.get("y_pixels"),
            y_ratio=lc_cfg.get("y_ratio")
        )

    counter = LineCounter(str(lanes_path)) if (lc_cfg.get("enabled", True) and lanes_path.exists()) else None

    last_print = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = det.infer(frame)
        frame = dib_cajas(frame, results, class_names)

        if counter is not None:
            info  = counter.update(results, class_names)
            frame = dib_lineas(frame, counter.lines_for_draw())
            frame = dib_panel_cont(frame, info["live_counts"])

            if time.time() - last_print > 10:
                print("Live_Counts:", info["live_counts"])
                last_print = time.time()

        cv2.imshow("Conteo", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
