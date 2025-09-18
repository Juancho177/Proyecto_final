from src.vision.detector import Detector
from src.vision.counter import LineCounter
from src.control.cont_heuristico import cont_heuristico, ControlParams

class FakeYOLO:
    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def track(self, frame, **kwargs):
        out = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return out

    def __call__(self, frame, **kwargs):
        return self.track(frame, **kwargs)

def test_pipeline_smoke(lanes_file, make_result):
    frames = [
        make_result([{"xyxy":[100,10,140,30], "cls":0, "id":1}]),
        make_result([{"xyxy":[100,60,140,80], "cls":0, "id":1}]),
    ]
    det = Detector(weights=None, model=FakeYOLO(frames))
    cnt = LineCounter(str(lanes_file))
    ctl = cont_heuristico(ControlParams())

    res1 = det.infer(None)
    cnt.update(res1)
    res2 = det.infer(None)
    info = cnt.update(res2)

    assert info["counts"].get("0", 0) == 1

    plan = ctl.decide({"Principal": 5})
    assert "next_phase" in plan and "greens" in plan
