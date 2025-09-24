# src_grpc/client_example.py
import argparse
import grpc
import traffic_pb2 as pb
import traffic_pb2_grpc as pbg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostport", default="localhost:50051", help="host:port del servidor gRPC")
    parser.add_argument("--video", required=True, help="Ruta de video o '0' para webcam o RTSP")
    parser.add_argument("--save-output", action="store_true", help="Pedir al servidor que guarde un MP4")
    parser.add_argument("--weights-path", default="", help="Override ruta pesos YOLO (opcional)")
    parser.add_argument("--lanes-config", default="", help="Override lanes.json (opcional)")
    args = parser.parse_args()

    # Canal gRPC
    channel = grpc.insecure_channel(args.hostport)
    stub = pbg.TrafficServiceStub(channel)

    # Petición (debe coincidir con tu traffic.proto: ProcessRequest)
    req = pb.ProcessRequest(
        video_source=args.video,
        save_output=bool(args.save_output),
        weights_path=args.weights_path,
        lanes_config=args.lanes_config,
    )

    # Llamada server-streaming: iteramos respuestas
    print(f"[CLIENT] Enviando petición a {args.hostport} ...")
    try:
        for resp in stub.ProcessVideo(req):
            # resp es ProcessResponse con uno de los campos: tick | summary
            if resp.HasField("tick"):
                t = resp.tick
                print(
                    f"[TICK] frame={t.frame_index} fase={t.fase} estado={t.estado} "
                    f"t_restante={t.t_restante}s vpm={dict(t.rates.vpm)} "
                    f"tiempos={{verde:{t.tiempos.verde:.1f}, amarillo:{t.tiempos.amarillo:.1f}, rojo:{t.tiempos.rojo:.1f}}}"
                )
            if resp.HasField("summary"):
                s = resp.summary
                print(
                    f"[SUMMARY] duracion={s.duracion_sesion:.1f}s "
                    f"tiempos={{verde:{s.tiempos.verde:.1f}, amarillo:{s.tiempos.amarillo:.1f}, rojo:{s.tiempos.rojo:.1f}}} "
                    f"output='{s.output_video_path}'"
                )
    except grpc.RpcError as e:
        print(f"[CLIENT][ERROR] {e.code().name}: {e.details()}")


if __name__ == "__main__":
    main()
