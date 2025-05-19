# utils.py

import numpy as np

def calc_distance(p1, p2):
    """
    두 점 (x1, y1), (x2, y2) 사이의 유클리드 거리 계산
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def select_best_shadow_point(contour, finger_pos, x_offset=0, y_offset=0):
    """
    주어진 손가락 위치와 그림자 컨투어(contour)를 비교하여
    수직선 기준으로 가장 적합한 그림자 점을 선택

    기준:
    - 손가락 아래쪽에 위치 (py_global > fy)
    - 수직선으로 가까움 (|px_global - fx| < 10)
    - 가장 아래에 있는 점 (py_global 최대)
    """
    fx, fy = finger_pos
    best_val = -1
    best_point = None
    for pt in contour:
        px, py = pt[0]
        px_global = px + x_offset
        py_global = py + y_offset
        if abs(px_global - fx) < 10 and py_global > fy:
            if py_global > best_val:
                best_val = py_global
                best_point = (px_global, py_global)
    return best_point


def is_shadow_touch(finger_pos, shadow_pos, threshold=20):
    """
    손가락과 그림자 사이 거리가 특정 임계값보다 작으면 터치된 것으로 판정
    """
    if not shadow_pos:
        return False
    return calc_distance(finger_pos, shadow_pos) < threshold
