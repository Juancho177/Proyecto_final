from src.utils.geometry import segments_intersect, center

def test_segments_intersect_true():
    A,B=(0,0),(10,0); C,D=(5,-5),(5,5)
    assert segments_intersect(A,B,C,D)

def test_segments_intersect_false():
    A,B=(0,0),(10,0); C,D=(0,5),(10,5)
    assert not segments_intersect(A,B,C,D)

def test_center():
    xyxy = [10, 20, 30, 60]
    cx, cy = center(xyxy)
    assert cx == 20 and cy == 40

