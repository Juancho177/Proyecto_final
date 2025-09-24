# src_grpc/server.py
import time
from concurrent import futures
from pathlib import Path
import sys
import grpc

# --- Hacer visible la raíz del proyecto (donde está final_consolidado.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Stubs generados por protoc (proto/traffic.proto)
import traffic_pb2 as pb
import traffic_pb2_grpc as pbg

# Tu pipeline (expuesto como generador en final_consolidado.py)
from final_consolidado import run_stream


class TrafficService(pbg.TrafficServiceServicer):
    """
    Implementa el servicio definido en traffic.proto:
      service TrafficService {
        rpc ProcessVideo (ProcessRequest) returns (stream ProcessResponse);
        rpc HealthCheck (Empty) returns (Health);
      }
    """

    def ProcessVideo(self, request, context):
        """
        Server-streaming: emite muchos 'tick' durante el procesamiento
        y al final un 'summary'. Usa tu run_stream(headless=True).
        """
        start = time.time()
        last_tiempos = {"verde": 0.0, "amarillo": 0.0, "rojo": 0.0}
        frame_index = 0

        # Si el cliente quiere guardar salida, definimos un MP4
        save_path = None
        if getattr(request, "save_output", False):
            out_dir = PROJECT_ROOT / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(out_dir / f"grpc_out_{int(time.time())}.mp4")

        try:
            gen = run_stream(
                video_source=request.video_source,
                weights_path=request.weights_path if getattr(request, "weights_path", "") else None,
                lanes_config=request.lanes_config if getattr(request, "lanes_config", "") else None,
                headless=True,
                save_path=save_path,
                ctrl_overrides=None,
                app_overrides=None,
            )

            for _, stats in gen:
                fase = stats.get("fase") or "Principal"
                estado = stats.get("estado") or ""
                t_rest = int(stats.get("t_restante", 0))
                rates = stats.get("rates_vpm", {}) or {}
                tiempos = stats.get("tiempos", {}) or {}
                last_tiempos = tiempos

                tick_msg = pb.Tick(
                    fase=fase,
                    estado=estado,
                    t_restante=t_rest,
                    rates=pb.Rates(vpm={k: float(v) for k, v in rates.items()}),
                    tiempos=pb.PhaseTimes(
                        verde=float(tiempos.get("verde", 0.0)),
                        amarillo=float(tiempos.get("amarillo", 0.0)),
                        rojo=float(tiempos.get("rojo", 0.0)),
                    ),
                    frame_index=frame_index,
                )
                frame_index += 1
                yield pb.ProcessResponse(tick=tick_msg)

        except Exception as e:
            context.set_details(f"ProcessVideo error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            return

        dur = time.time() - start
        summary_msg = pb.Summary(
            duracion_sesion=float(dur),
            tiempos=pb.PhaseTimes(
                verde=float(last_tiempos.get("verde", 0.0)),
                amarillo=float(last_tiempos.get("amarillo", 0.0)),
                rojo=float(last_tiempos.get("rojo", 0.0)),
            ),
            output_video_path=save_path or "",
        )
        yield pb.ProcessResponse(summary=summary_msg)

    def HealthCheck(self, request, context):
        return pb.Health(status="SERVING", version="v1")


def serve(port: int = 50051):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pbg.add_TrafficServiceServicer_to_server(TrafficService(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"🚦 gRPC TrafficService escuchando en 0.0.0.0:{port}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
