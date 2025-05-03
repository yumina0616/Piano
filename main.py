import cv2
import mediapipe as mp
import pygame

# PyGame 오디오 초기화
pygame.mixer.init()

# 피아노 음 로딩
note_names = ["a1", "b1", "c1", "d1", "e1", "f1", "g1", "g1s"]
notes = [pygame.mixer.Sound(f"assets/sounds/{name}.wav") for name in note_names]

# Mediapipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 상태 변수
prev_note_index = -1
is_pressed = False
z_threshold = -0.15  # 손이 가까워졌다고 판단할 기준값 (조정 가능)

with mp_hands.Hands(
    max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)
        height, width, _ = image.shape

        # 가상 건반 시각화
        key_width = width // 8
        for i in range(8):
            x1 = i * key_width
            x2 = (i + 1) * key_width
            cv2.rectangle(image, (x1, 0), (x2, height), (50, 50, 50), 1)
            cv2.putText(
                image,
                note_names[i],
                (x1 + 10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (100, 100, 255),
                2,
            )

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # 검지 끝 좌표
                x = int(hand_landmarks.landmark[8].x * width)
                y = int(hand_landmarks.landmark[8].y * height)
                z = hand_landmarks.landmark[8].z  # 깊이 정보

                cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
                cv2.putText(
                    image,
                    f"Z: {z:.3f}",
                    (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

                # 건반 판정
                note_index = x // key_width

                if not is_pressed and z < z_threshold and note_index < len(notes):
                    notes[note_index].play()
                    is_pressed = True
                    prev_note_index = note_index

                # 손이 멀어지면 다시 눌 수 있게 초기화
                if z > z_threshold + 0.05:
                    is_pressed = False

        cv2.imshow("Air Piano (Z-based trigger)", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
