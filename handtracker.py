import cv2
import mediapipe as mp
import numpy as np
import time

NOTE_NAMES = ["A","E","B","Gb","Db","Ab","Eb","Bb","F","C","G","D"]
SHAPE_NAMES = ["maj","min","maj7","m7","dom7","dim7"]

class Handtracker:
    def __init__(self) -> None:
        self.mp_hands = mp.solutions.hands # type: ignore
        self.mp_draw = mp.solutions.drawing_utils # type: ignore

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def read_frame(self):
        att = 0
        success, img = self.cap.read()
        while not success and att < 5:
            time.sleep(0.2)
            success, img = self.cap.read()
            att += 1
        if success:
            img = cv2.flip(img, 1)
        return success, img

    def process(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.hands.process(rgb)

    def draw_landmarks(self, img, result):
        h, w, _ = img.shape
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
            
                finger_tips = {
                    # "Thumb": hand_landmarks.landmark[4],
                    "Index": hand_landmarks.landmark[8],
                    # "Middle": hand_landmarks.landmark[12],
                    # "Ring": hand_landmarks.landmark[16],
                    # "Pinky": hand_landmarks.landmark[20],
                }

                for name, landmark in finger_tips.items():
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.putText(
                        img, name, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        (255, 255, 255), 1
                    )
                    cv2.circle(
                        img, (x, y), 10, (0, 255, 0), 2
                    )

    def get_chord_selection(self, img, result):
        return self.draw_overlay(img, result)

    def show_img(self, name: str, img):
        cv2.imshow(name, img)

    def is_quit_pressed(self, key: str):
        return cv2.waitKey(1) & 0xFF == ord(key)

    def release(self):
        self.hands.close()
        self.cap.release()
        cv2.destroyAllWindows()

    def zone_from(self, hx, hy, wx, wy, wr, sr, labels) -> str | None:
        angle = (2 * np.pi) / len(labels)
        for i, label in enumerate(labels):
            a = angle * i
            nx = int(np.cos(a) * wr + wx)
            ny = int(np.sin(a) * wr + wy)
            if (hx - nx)**2 + (hy - ny)**2 <= sr**2:
                return label
        return None

    def draw_zone_wheel(self, img, x, y, r, sr, labels, active_label, color):
        n = len(labels)
        angle = (2 * np.pi) / n
        for i, label in enumerate(labels):
            a = angle * i
            nx = int(np.cos(a) * r + x)
            ny = int(np.sin(a) * r + y)

            is_active = (label == active_label)
            if is_active:
                overlay = img.copy()
                cv2.circle(overlay, (nx, ny), sr, color, -1)
                img[:] = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
            
            cv2.circle(img, (nx, ny), sr, (80, 80, 80), 1)
            text_color = (255, 255, 255) if is_active else (10, 10, 10)
            cv2.putText(img, label, (nx - 10, ny),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 3 if is_active else 2)
        
        return img

    def draw_overlay(self, img, result):
        h, w, _ = img.shape
        root_label, shape_label = None, None

        l_r = int(h * 0.25)
        sl_r = int(l_r * 0.25)
        img = self.draw_zone_wheel(img, w // 4, h // 2, l_r, sl_r, NOTE_NAMES, None, (0, 200, 255))
        r_r = int(h * 0.25)
        sr_r = int(r_r * 0.42)
        img = self.draw_zone_wheel(img, 3 * w // 4, h // 2, r_r, sr_r, SHAPE_NAMES, None, (0, 255, 120))

        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"
                tip = hand_landmarks.landmark[8]  # index fingertip, normalized 0..1
                cx, cy = int(tip.x * w), int(tip.y * h)

                if label == "Left":
                    root_label = self.zone_from(cx, cy, w // 4, h // 2, h // 4, sl_r, NOTE_NAMES)
                    cv2.circle(img, (cx, cy), 12, (0, 200, 255), -1)
                else:
                    shape_label = self.zone_from(cx, cy, 3 * w // 4, h // 2, h // 4, sr_r, SHAPE_NAMES)
                    cv2.circle(img, (cx, cy), 12, (0, 255, 120), -1)

        img = self.draw_zone_wheel(img, w // 4, h // 2, l_r, sl_r, NOTE_NAMES, root_label, (0, 200, 255))
        img = self.draw_zone_wheel(img, 3 * w // 4, h // 2, r_r, sr_r, SHAPE_NAMES, shape_label, (0, 255, 120))
    
        chord_text = f"{root_label or '--'} {shape_label or '--'}"
        cv2.putText(img, chord_text, (w // 2 - 80, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        return img, root_label, shape_label
