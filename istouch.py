# istouch.py

import cv2
import mediapipe as mp
import numpy as np

def run_touch_detection():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            height, width, _ = image.shape

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    fingertip_indices = [4, 8, 12, 16, 20]
                    padding_x = 20
                    padding_y = 40
                    threshold_distance = 20

                    for i, idx in enumerate(fingertip_indices):
                        fx = int(hand_landmarks.landmark[idx].x * width)
                        fy = int(hand_landmarks.landmark[idx].y * height)

                        x1 = max(fx - padding_x, 0)
                        x2 = min(fx + padding_x, width)
                        y1 = fy
                        y2 = min(fy + padding_y, height)

                        roi = image[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue

                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)

                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        mean_brightness = np.mean(gray_roi)
                        threshold = mean_brightness - 15
                        roi_mask = cv2.inRange(gray_roi, 0, threshold)

                        shadow_mask = np.zeros((height, width), dtype=np.uint8)
                        shadow_mask[y1:y2, x1:x2] = roi_mask

                        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours:
                            continue

                        largest_contour = max(contours, key=cv2.contourArea)

                        best_shadow_point = None
                        best_val = -1

                        for point in largest_contour:
                            px, py = point[0]
                            px_global = px + x1
                            py_global = py + y1

                            if abs(px_global - fx) < 10 and py_global > fy:
                                if py_global > best_val:
                                    best_val = py_global
                                    best_shadow_point = (px_global, py_global)

                        cv2.circle(image, (fx, fy), 5, (255, 0, 0), -1)
                        finger_number = str(i + 1)
                        cv2.putText(image, finger_number, (fx - 10, fy - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        if best_shadow_point:
                            cv2.circle(image, best_shadow_point, 5, (255, 0, 255), -1)
                            dist = np.linalg.norm(np.array([fx, fy]) - np.array(best_shadow_point))
                            if dist < threshold_distance:
                                cv2.putText(image, "Touch", (fx - 10, fy + 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        largest_contour_shifted = largest_contour + [x1, y1]
                        cv2.drawContours(image, [largest_contour_shifted], -1, (0, 255, 255), 1)

            cv2.imshow("Phase1 - Finger Touch Detection", image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_touch_detection()
