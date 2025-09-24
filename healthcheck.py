import sys
def main():
    try:
        import cv2, yaml, json
        from src.vision.detector import Detector
        from src.vision.counter import LineCounter
        from src.control.cont_heuristico import cont_heuristico, ControlParams
    except Exception as e:
        print(f"[HEALTHCHECK] Import failed: {e}", file=sys.stderr)
        sys.exit(2)
    try:
        _ = cont_heuristico(ControlParams())
        print("[HEALTHCHECK] OK: imports y objetos básicos")
        sys.exit(0)
    except Exception as e:
        print(f"[HEALTHCHECK] Objects failed: {e}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
