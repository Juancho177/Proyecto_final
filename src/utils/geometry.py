from typing import Tuple

def ccw(A,B,C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def segments_intersect(A,B,C,D) -> bool:
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def center(xyxy):
    x1,y1,x2,y2 = xyxy
    return ((x1+x2)/2, (y1+y2)/2)


