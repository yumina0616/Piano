# tests/test_istouch_mock.py

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

class TestIstouchMock(unittest.TestCase):

    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    @patch('cv2.waitKey', return_value=27)  # ESC 입력 시뮬레이션
    @patch('cv2.destroyAllWindows')
    def test_run_touch_detection_once(self, mock_destroy, mock_waitkey, mock_imshow, mock_videocap):

        # Step 1: Mock VideoCapture 인스턴스 생성
        mock_cap = MagicMock()
        mock_cap.isOpened.side_effect = [True, False]  # while 루프 1회만
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))  # 검은 화면 프레임
        mock_videocap.return_value = mock_cap

        # Step 2: istouch 모듈에서 함수 실행
        import istouch
        istouch.run_touch_detection()

        # Step 3: 호출 여부 확인
        mock_cap.read.assert_called()
        mock_imshow.assert_called()
        mock_waitkey.assert_called_once()
        mock_destroy.assert_called_once()
