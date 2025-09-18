from src.vision.counter import LineCounter

def test_line_counter_counts_crossing(lanes_file, make_result):
  
    lc = LineCounter(str(lanes_file))

    res1 = make_result([{"xyxy":[100,10,140,30], "cls":0, "id":1}])
    info1 = lc.update(res1)
    assert info1["counts"] == {}

    res2 = make_result([{"xyxy":[100,60,140,80], "cls":0, "id":1}])
    info2 = lc.update(res2)

    assert "0" in info2["counts"]
    assert info2["counts"]["0"] == 1

    assert "Principal" in info2["by_approach"]
    assert info2["by_approach"]["Principal"]["total"] >= 1
    assert info2["veh_per_min"]["Principal"] >= 0.0

    assert info2["live_counts"].get("0", 0) >= 1



