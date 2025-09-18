from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple, Optional
import time
from src.utils.geometry import segments_intersect, center
from src.utils.io import load_json

class LineCounter:
  
    def __init__(self, lanes_json: str):
        cfg = load_json(lanes_json)

        self.lines: List[Dict[str, Any]] = []
        for ap in cfg.get("approaches", []):
            ap_name = ap.get("name", "AP")
            for ln in ap.get("count_lines", []):
                self.lines.append({
                    "p1": (ln["p1"][0], ln["p1"][1]),
                    "p2": (ln["p2"][0], ln["p2"][1]),
                    "approach": ap_name,
                    "dir": ln.get("dir")  
                })

        self.rate_window_s: int = int(cfg.get("rate_window_s", 30))

        self.memory: Dict[int, Tuple[float, float]] = {}     
        self.totals: Dict[str, int] = defaultdict(int)       
        self.by_approach: Dict[str, int] = defaultdict(int)  

        self._seen_per_line: List[set] = [set() for _ in self.lines]

        self.live_counts: Dict[str, int] = {}

        self._ap_class_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        self._ap_cross_times: Dict[str, deque] = defaultdict(lambda: deque())

    @staticmethod
    def _dir_ok(dir_flag: Optional[str],
                last_pt: Tuple[float, float],
                new_pt: Tuple[float, float]) -> bool:
        if not dir_flag:
            return True
        dx = new_pt[0] - last_pt[0]
        dy = new_pt[1] - last_pt[1]
        if dir_flag == "up":
            return dy < 0
        if dir_flag == "down":
            return dy > 0
        if dir_flag == "left":
            return dx < 0
        if dir_flag == "right":
            return dx > 0
        return True

    def _purge_old(self, dq: deque, now: float):
        win = self.rate_window_s
        while dq and (now - dq[0]) > win:
            dq.popleft()

    def update(self, results, class_names=None) -> Dict[str, Any]:
        live = defaultdict(int)
        now = time.time()

        for r in results:
            for b in r.boxes:
                raw_cls = b.cls[0]
                cls_scalar = raw_cls.item() if hasattr(raw_cls, "item") else raw_cls
                cls_idx = int(cls_scalar)
                name = class_names[cls_idx] if class_names else str(cls_idx)
                live[name] += 1

                if b.id is not None:
                    raw_id = b.id[0]
                    id_scalar = raw_id.item() if hasattr(raw_id, "item") else raw_id
                    track_id = int(id_scalar)
                    pt = center(b.xyxy[0].tolist())
                    last = self.memory.get(track_id)

                    if last:
                        for i, L in enumerate(self.lines):
                            if track_id in self._seen_per_line[i]:
                                continue 

                            if segments_intersect(last, pt, L["p1"], L["p2"]) and self._dir_ok(L.get("dir"), last, pt):
                                self.totals[name] += 1
                                self.by_approach[L["approach"]] += 1
                                self._ap_class_totals[L["approach"]][name] += 1
                                self._ap_cross_times[L["approach"]].append(now)
                                self._purge_old(self._ap_cross_times[L["approach"]], now)
                                self._seen_per_line[i].add(track_id)

                    self.memory[track_id] = pt

        veh_per_min = {}
        for ap, dq in self._ap_cross_times.items():
            self._purge_old(dq, now)
            veh_per_min[ap] = (len(dq) * 60.0 / max(1.0, self.rate_window_s))

        by_approach_extended: Dict[str, Any] = {}
        all_approaches = {L["approach"] for L in self.lines} | set(self._ap_class_totals.keys()) | set(veh_per_min.keys())
        for ap in all_approaches:
            by_approach_extended[ap] = {
                "total": self.by_approach.get(ap, 0),
                "classes": dict(self._ap_class_totals.get(ap, {})),
                "veh_per_min": veh_per_min.get(ap, 0.0),
            }

        self.live_counts = dict(live)

        return {
            "live_counts": self.live_counts,     
            "counts": dict(self.totals),         
            "by_approach": by_approach_extended,
            "veh_per_min": veh_per_min
        }

    def lines_for_draw(self):
        return [(L["p1"], L["p2"]) for L in self.lines]

    def reset(self):
        self.memory.clear()
        self.totals.clear()
        self.by_approach.clear()
        for s in self._seen_per_line:
            s.clear()
        self.live_counts = {}
        self._ap_class_totals.clear()
        self._ap_cross_times.clear()


