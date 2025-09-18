import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest, json, numpy as np

@pytest.fixture
def lanes_file(tmp_path: Path):
  
    cfg = {
        "approaches": [
            {"name": "Principal", "count_lines": [
                {"p1": [0, 40], "p2": [200, 40], "dir": "down"}
            ]}
        ],
        "rate_window_s": 30,
        "zones": []
    }
    f = tmp_path / "lanes.json"
    f.write_text(json.dumps(cfg), encoding="utf-8")
    return f

class _FakeBox:
    def __init__(self, xyxy, cls_id, track_id):
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.cls  = [np.array([cls_id], dtype=float)]
        self.id   = [np.array([track_id], dtype=float)]

class _FakeBoxes:
    def __init__(self, boxes): self._boxes = boxes
    def __iter__(self): return iter(self._boxes)

class _FakeResult:
    def __init__(self, boxes): self.boxes = _FakeBoxes(boxes)

@pytest.fixture
def make_result():
   
    def _make(frame_boxes):
        boxes = [_FakeBox(b["xyxy"], b["cls"], b["id"]) for b in frame_boxes]
        return [_FakeResult(boxes)]
    return _make
