from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class ControlParams:
    min_green: int = 12
    max_green: int = 60
    green_extension: int = 3
    clearance: int = 4
    thresholds: Optional[Dict[str, float]] = None 
   
    class_weights: Dict[str, float] = field(default_factory=lambda: {
        "car": 1.0,
        "motorbike": 0.6,
        "bus": 2.5,
        "truck": 2.0,
        "van": 1.2,
        "bicycle": 0.3,
        "rickshaw": 0.8,
    })
   
    phase_change_penalty: float = 0.0  

class cont_heuristico:
   
    def __init__(self, params: ControlParams):
        self.p = params
        if self.p.thresholds is None:
            self.p.thresholds = {"low": 6, "medium": 12, "high": 20}
        self._last_phase: Optional[str] = None

    def _weighted_rate(self,
                       ap: str,
                       rates_vpm: Dict[str, float],
                       detail_by_approach: Optional[Dict[str, Dict]] = None,
                       live_counts: Optional[Dict[str, int]] = None) -> float:

        base = 0.0
        if detail_by_approach and ap in detail_by_approach:
            cls_counts = detail_by_approach[ap].get("classes", {}) or {}
            for cls_name, count in cls_counts.items():
                w = self.p.class_weights.get(cls_name, 1.0)
                base += w * float(count)
            vpm = detail_by_approach[ap].get("veh_por_min", rates_vpm.get(ap, 0.0))
            base = 0.5 * vpm + 0.5 * base
        else:
            base = rates_vpm.get(ap, 0.0)
        if live_counts:
            occ_term = 0.0
            for cls_name, occ in live_counts.items():
                w = self.p.class_weights.get(cls_name, 1.0)
                occ_term += w * float(occ)
            base += 0.1 * occ_term  

        if self.p.phase_change_penalty > 0 and self._last_phase and ap != self._last_phase:
            base -= float(self.p.phase_change_penalty)

        return max(0.0, base)

    def _suggest_green(self, demand_weighted: float) -> int:
        t = self.p.min_green
        th = self.p.thresholds
        if demand_weighted >= th["high"]:
            t = self.p.max_green
        elif demand_weighted >= th["medium"]:
            t = max(self.p.min_green + self.p.green_extension, int(self.p.min_green * 1.8))
        elif demand_weighted >= th["low"]:
            t = int(self.p.min_green * 1.3)
        return max(self.p.min_green, min(t, self.p.max_green))

    def decide(self,
               rates_vpm: Dict[str, float], *,
               detail_by_approach: Optional[Dict[str, Dict]] = None,
               live_counts: Optional[Dict[str, int]] = None) -> Dict[str, object]:
        if not rates_vpm:
            return {"next_phase": None, "greens": {}, "clearance": self.p.clearance}

        weighted = {ap: self._weighted_rate(ap, rates_vpm, detail_by_approach, live_counts)
                    for ap in rates_vpm.keys()}

        next_phase = max(weighted, key=weighted.get)
        greens = {ap: self._suggest_green(d) for ap, d in weighted.items()}

        self._last_phase = next_phase
        return {"next_phase": next_phase, "greens": greens, "clearance": self.p.clearance}
