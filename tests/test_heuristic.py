from src.control.cont_heuristico import cont_heuristico, ControlParams

def test_controller_respects_bounds():
    params = ControlParams(min_green=12, max_green=60, green_extension=3)
    ctl = cont_heuristico(params)

    plan = ctl.decide({"Principal": 25})
    assert plan["greens"]["Principal"] <= params.max_green
    assert plan["greens"]["Principal"] >= params.min_green
    assert plan["clearance"] == params.clearance

def test_controller_phase_selection_changes_with_load():
    params = ControlParams()
    ctl = cont_heuristico(params)

    plan1 = ctl.decide({"Norte": 2, "Sur": 20})
    plan2 = ctl.decide({"Norte": 30, "Sur": 5})
    assert plan1["next_phase"] != plan2["next_phase"]
