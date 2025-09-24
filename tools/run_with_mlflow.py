import argparse
import os
from pathlib import Path
import time
import csv
import mlflow
import sys

# --- asegurar import del proyecto desde tools/ ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from final_consolidado import run_stream, ROOT
from src.utils.io import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Ruta a video local o '0' o URL RTSP")
    parser.add_argument("--experiment", default="semaforizacion_yolo12", help="Nombre experimento MLflow")
    parser.add_argument("--run-name", default=None, help="Nombre amigable del run")
    parser.add_argument("--save-output", action="store_true", help="Guardar MP4 procesado")
    parser.add_argument("--log-interval", type=int, default=25, help="Intervalo de steps para log de métricas")
    parser.add_argument("--lanes-config", default=None, help="Override ruta lanes.json")
    parser.add_argument("--weights-path", default=None, help="Override ruta weights YOLO")
    args = parser.parse_args()

    # ---- Normaliza/valida la ruta del video ----
    raw_video = args.video
    if str(raw_video) != "0":
        p = Path(raw_video)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Video no encontrado: {p}")
        args.video = str(p)  # usar ruta absoluta normalizada
        print(f"[INFO] Usando video: {args.video}")
    else:
        print("[INFO] Usando cámara (0)")

    # 1) Tracking local/servidor:
    #    - Si MLFLOW_TRACKING_URI está definida (p. ej., http://127.0.0.1:8080), úsala.
    #    - Si no, usar ./mlruns local con URI file:// seguro para Windows.
    tracking_uri_env = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri_env:
        tracking_uri = tracking_uri_env
        mlflow.set_tracking_uri(tracking_uri_env)
    else:
        tracking_uri = (Path.cwd() / "mlruns").resolve().as_uri()
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(args.experiment)

    # 2) Lee configuración para registrar como artefacto
    app_yaml = ROOT / "config" / "app.yaml"
    app_cfg = load_yaml(app_yaml)

    # 3) Params de control a loggear (del YAML)
    ctrl = app_cfg.get("control", {}) or {}
    det_conf = app_cfg.get("conf", 0.3)
    det_iou = app_cfg.get("iou", 0.5)

    run_name = args.run_name or f"video={Path(args.video).name if str(raw_video) != '0' else 'cam0'}"
    with mlflow.start_run(run_name=run_name):
        # --- Params: detector ---
        mlflow.log_param("detector_conf", det_conf)
        mlflow.log_param("detector_iou", det_iou)
        mlflow.log_param("weights_path", args.weights_path or "default_best.pt")

        # --- Params: control ---
        mlflow.log_param("min_green", ctrl.get("min_green", 12))
        mlflow.log_param("max_green", ctrl.get("max_green", 60))
        mlflow.log_param("clearance", ctrl.get("clearance", 4))
        mlflow.log_param("fixed_red_time", ctrl.get("fixed_red_time", 10))
        mlflow.log_param("phase_change_penalty", ctrl.get("phase_change_penalty", 0.0))

        # Pesos por clase (si los hay)
        cw = (ctrl.get("class_weights") or {}) if isinstance(ctrl.get("class_weights"), dict) else {}
        for k, v in cw.items():
            mlflow.log_param(f"class_weight__{k}", v)

        # --- Tags útiles ---
        mlflow.set_tags({
            "video_source": args.video if str(raw_video) != "0" else "camera_0",
            "lanes_config": args.lanes_config or app_cfg.get("line_counting", {}).get("lanes_config", "default"),
            "mode": "inference+control",
            "tracking_uri": tracking_uri,
        })

        # --- Artefactos de configuración ---
        mlflow.log_artifact(str(app_yaml), artifact_path="config")
        if args.lanes_config:
            mlflow.log_artifact(str(Path(args.lanes_config)), artifact_path="config")
        else:
            lanes = ROOT / (app_cfg.get("line_counting", {}).get("lanes_config", "config/lanes.example.json"))
            if lanes.exists():
                mlflow.log_artifact(str(lanes), artifact_path="config")

        # --- Opcional: salida MP4 ---
        save_path = None
        if args.save_output:
            out_dir = Path("outputs"); out_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            save_path = str(out_dir / f"mlflow_out_{ts}.mp4")

        # --- Loop de procesamiento ---
        step = 0
        csv_path = Path("outputs") / f"metrics_{int(time.time())}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow([
                "step","fase","estado","t_restante","flujo_total",
                "vpm_por_aproximacion","verde","amarillo","rojo"
            ])

            for frame_bgr, stats in run_stream(
                video_source=args.video,
                weights_path=args.weights_path,
                lanes_config=args.lanes_config,
                headless=True,
                save_path=save_path,
                ctrl_overrides=None,
                app_overrides=None,
            ):
                # métricas instantáneas
                fase = stats.get("fase") or "Principal"
                estado = stats.get("estado")
                t_rest = stats.get("t_restante", 0)
                rates = stats.get("rates_vpm", {})
                tiempos = stats.get("tiempos", {})
                flujo_total = int(sum(rates.values())) if rates else 0

                # Log nominal cada 'log_interval'
                if step % args.log_interval == 0:
                    # Métricas escalares globales
                    mlflow.log_metrics({
                        "flujo_total": flujo_total,
                        "t_restante": t_rest,
                        "tiempo_verde_s": tiempos.get("verde", 0.0),
                        "tiempo_amarillo_s": tiempos.get("amarillo", 0.0),
                        "tiempo_rojo_s": tiempos.get("rojo", 0.0),
                    }, step=step)

                    # Métricas por aproximación (prefijo)
                    for ap, v in (rates.items() if rates else []):
                        mlflow.log_metric(f"vpm__{ap}", float(v), step=step)

                # Guardar fila CSV (útil para inspección posterior)
                writer.writerow([
                    step, fase, estado, t_rest, flujo_total,
                    "|".join([f"{k}:{v:.2f}" for k, v in (rates.items() if rates else [])]),
                    int(tiempos.get("verde", 0.0)),
                    int(tiempos.get("amarillo", 0.0)),
                    int(tiempos.get("rojo", 0.0)),
                ])

                step += 1

        # --- Subir artefactos finales ---
        if Path(csv_path).exists():
            mlflow.log_artifact(str(csv_path), artifact_path="logs")

        if args.save_output and save_path and Path(save_path).exists():
            mlflow.log_artifact(save_path, artifact_path="videos")

        print(f"[MLflow] Run completado. Experimento: {args.experiment}")


if __name__ == "__main__":
    main()


