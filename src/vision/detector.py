from pathlib import Path
from typing import Optional, Any
from ultralytics import YOLO

class Detector:

    def __init__(
        self,
        weights: Optional[str],
        conf: float = 0.3,
        iou: float = 0.5,
        tracker_cfg: Optional[str] = None,
        classes: Optional[list[int]] = None,
        model: Any = None,               
    ):
        self.conf = conf
        self.iou = iou
        self.tracker_cfg = tracker_cfg
        self.classes = classes

        if model is not None:             
            self.model = model
        else:
            if weights is None:
                raise ValueError("weights=None y model=None: se requiere al menos uno para inicializar Detector")
            self.model = YOLO(str(weights))

    def infer(self, frame):
    
        if self.tracker_cfg:
            return self.model.track(
                frame,
                persist=True,
                conf=self.conf,
                iou=self.iou,
                tracker=self.tracker_cfg
            )
        else:
            return self.model(
                frame,
                conf=self.conf,
                iou=self.iou
            )


