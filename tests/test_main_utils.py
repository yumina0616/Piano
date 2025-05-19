# tests/test_main_utils.py

import unittest
import numpy as np
from utils import calc_distance, select_best_shadow_point, is_shadow_touch

class TestUtilsFunctions(unittest.TestCase):

    def test_calc_distance(self):
        p1 = (0, 0)
        p2 = (3, 4)
        result = calc_distance(p1, p2)
        self.assertAlmostEqual(result, 5.0)

    def test_select_best_shadow_point(self):
        contour = np.array([
            [[5, 10]],
            [[6, 20]],
            [[7, 30]]
        ])
        finger_pos = (6, 15)
        best_point = select_best_shadow_point(contour, finger_pos, x_offset=0, y_offset=0)
        self.assertEqual(best_point, (7, 30))  # 가장 아래에 있는 점을 반환

    def test_select_best_shadow_point_none(self):
        contour = np.array([
            [[5, 5]],
            [[6, 6]]
        ])
        finger_pos = (50, 10)  # x 좌표 범위 안 맞게 설정
        result = select_best_shadow_point(contour, finger_pos)
        self.assertIsNone(result)

    def test_is_shadow_touch_true(self):
        finger = (10, 10)
        shadow = (12, 12)
        self.assertTrue(is_shadow_touch(finger, shadow, threshold=5))

    def test_is_shadow_touch_false(self):
        finger = (10, 10)
        shadow = (50, 50)
        self.assertFalse(is_shadow_touch(finger, shadow, threshold=5))

    def test_is_shadow_touch_none(self):
        finger = (10, 10)
        shadow = None
        self.assertFalse(is_shadow_touch(finger, shadow))

if __name__ == '__main__':
    unittest.main()
