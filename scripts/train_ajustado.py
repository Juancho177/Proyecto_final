from ultralytics import YOLO
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "config" / "data.yaml"
OUT  = ROOT / "models" / "yolo12"

def main():
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU?:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))

    model = YOLO("yolo12n.pt")

    results = model.train(
        data=str(DATA),
        epochs=180,             
        patience=10,            
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        pretrained=True,
        cache=True,
        project=str(OUT),
        name="modelo_final",
        deterministic=False,    
        cos_lr=True,           
        lr0=0.01,               
        lrf=0.01,              
        warmup_epochs=3,
        
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        fliplr=0.5, flipud=0.0,
        scale=0.5,
        mosaic=1.0,
        close_mosaic=15,
        mixup=0.10,
        val=True
    )

    print("Listo. Resultados en:", OUT / "modelo_final")

if __name__ == "__main__":
    main()
