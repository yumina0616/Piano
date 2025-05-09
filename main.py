import cv2
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import pygame

pygame.mixer.init()
note_names = ["c1","d1","e1","f1","g1","a1","b1","c2"]
notes = [pygame.mixer.Sound(f"assets/sounds/{name}.wav") for name in note_names]
font_path = "C:/Windows/Fonts/malgun.ttf"


def draw_korean_text(image, text, position, font_path, font_size=30, color=(255, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


def draw_virtual_keys(image, origin_x, origin_y, key_width=40, key_height=100):
    key_rects = []
    for i in range(8):
        top_left = (origin_x + i * key_width, origin_y)
        bottom_right = (origin_x + (i + 1) * key_width, origin_y + key_height)
        key_rects.append((top_left, bottom_right))
        cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 2)
    return image, key_rects


def find_nearest_shadow(finger_pos, contours):
    min_distance = float('inf')
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist = np.linalg.norm(np.array([cx, cy]) - np.array(finger_pos))
            if dist < min_distance:
                min_distance = dist
    return min_distance


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

calibrated = False
prev_note_index = -1
calibration_x, calibration_y = None, None
stable_count = 0
required_stable_frames = 30
stability_threshold = 5

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        height, width, _ = image.shape
        message = ""

        # 그림자 마스크 및 윤곽선 추출
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[4]
                x = int(thumb_tip.x * width)
                y = int(thumb_tip.y * height)
                cv2.circle(image, (x, y), 10, (0, 255, 0), -1)

                if not calibrated:
                    if calibration_x is None:
                        calibration_x, calibration_y = x, y
                        stable_count = 1
                    elif abs(x - calibration_x) < stability_threshold and abs(y - calibration_y) < stability_threshold:
                        stable_count += 1
                        if stable_count >= required_stable_frames:
                            calibrated = True
                            message = "건반이 준비되었습니다"
                    else:
                        calibration_x, calibration_y = x, y
                        stable_count = 1
                    message = "손을 건반 위치에 고정해주세요"

                else:
                    message = f"캘리브레이션 좌표: ({calibration_x}, {calibration_y})"
                    cv2.circle(image, (calibration_x, calibration_y), 15, (0, 0, 255), 2)

                    image, key_rects = draw_virtual_keys(image, calibration_x, calibration_y + 20)

                    finger_tip = hand_landmarks.landmark[8]
                    fx = int(finger_tip.x * width)
                    fy = int(finger_tip.y * height)
                    note_index = -1

                    for i, (tl, br) in enumerate(key_rects):
                        if tl[0] <= fx <= br[0] and tl[1] <= fy <= br[1]:
                            dist_to_shadow = find_nearest_shadow((fx, fy), contours)
                            if dist_to_shadow > 80:  # 그림자가 가까이 없을 때 = 손이 책상에 닿음
                                note_index = i
                            break

                    if note_index != -1 and note_index != prev_note_index:
                        notes[note_index].play()
                        prev_note_index = note_index

        if message:
            image = draw_korean_text(image, message, (30, 50), font_path, font_size=30, color=(255, 255, 0))

        cv2.imshow("Shadow-based Key Press", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
