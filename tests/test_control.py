from src.control.cont_heuristico import cont_heuristico, ControlParams

def test_controller_bounds_and_clearance():
    params = ControlParams(min_green=12, max_green=60, green_extension=3, clearance=4)
    ctl = cont_heuristico(params)
    plan = ctl.decide({"Principal": 25})
    assert params.min_green <= plan["greens"]["Principal"] <= params.max_green
    assert plan["clearance"] == params.clearance

def test_controller_phase_selection_changes_with_load():
    params = ControlParams()
    ctl = cont_heuristico(params)
    p1 = ctl.decide({"Norte": 2, "Sur": 20})
    p2 = ctl.decide({"Norte": 30, "Sur": 5})
    assert p1["next_phase"] != p2["next_phase"]
