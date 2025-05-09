import cv2
import mediapipe as mp
from PIL import ImageFont, ImageDraw, Image
import numpy as np

font_path = "C:/Windows/Fonts/malgun.ttf"

def draw_korean_text(image, text, position, font_path, font_size=30, color=(255, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


def find_nearest_shadow(finger_pos, contours):
    min_distance = float('inf')
    closest_center = None
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
                closest_center = (cx, cy)
    return closest_center, min_distance

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

        # 그림자 마스크 및 윤곽선 추출
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                for idx in [8]:  # 검지 끝만 우선 테스트
                    landmark = hand_landmarks.landmark[idx]
                    fx = int(landmark.x * width)
                    fy = int(landmark.y * height)
                    cv2.circle(image, (fx, fy), 6, (0, 255, 0), -1)

                    shadow_center, dist = find_nearest_shadow((fx, fy), contours)
                    if shadow_center:
                        cx, cy = shadow_center
                        cv2.circle(image, (cx, cy), 8, (255, 0, 0), -1)
                        cv2.line(image, (fx, fy), (cx, cy), (255, 0, 0), 1)
                        text = f"거리: {int(dist)}"
                        image = draw_korean_text(image, text, (fx + 10, fy - 10), font_path, 20, (255, 255, 0))

        cv2.imshow("손가락과 가장 가까운 그림자 시각화", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
