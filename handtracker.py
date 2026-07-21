import cv2
import mediapipe as mp
import time
import numpy as np
from synthesizer import NOTES, CHORD_SHAPES, ChordPlayer

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# NOTE_NAMES = list(NOTES.keys())          # ["A","Bb","B","C","Db","D","Eb","E","F","Gb","G","Ab"]
NOTE_NAMES = ["A","E","B","Gb","Db","Ab","Eb","Bb","F","C","G","D"]
SHAPE_NAMES = list(CHORD_SHAPES.keys())   # ["maj","min","maj7","m7","dom7","dim7"]

def zone_from(hx, hy, wx, wy, wr, sr, labels) -> str | None:
    angle = (2 * np.pi) / len(labels)
    for i, label in enumerate(labels):
        a = angle * i
        nx = int(np.cos(a) * wr + wx)
        ny = int(np.sin(a) * wr + wy)
        if (hx - nx)**2 + (hy - ny)**2 <= sr**2:
            return label
    return None

def draw_zone_wheel(img, x, y, r, sr, labels, active_label, color):
    h, w, _ = img.shape
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

def draw_overlay(img, result):
    h, w, _ = img.shape
    root_label, shape_label = None, None

    # draw the two zone bars: notes along on the left, shapes on the right
    l_r = int(h * 0.25)
    sl_r = int(l_r * 0.25)
    img = draw_zone_wheel(img, w // 4, h // 2, l_r, sl_r, NOTE_NAMES, None, (0, 200, 255))
    r_r = int(h * 0.25)
    sr_r = int(r_r * 0.5)
    img = draw_zone_wheel(img, 3 * w // 4, h // 2, r_r, sr_r, SHAPE_NAMES, None, (0, 255, 120))

    if result.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            label = handedness.classification[0].label  # "Left" or "Right"
            tip = hand_landmarks.landmark[8]  # index fingertip, normalized 0..1
            cx, cy = int(tip.x * w), int(tip.y * h)

            if label == "Left":
                root_label = zone_from(cx, cy, w // 4, h // 2, h // 4, sl_r, NOTE_NAMES)
                cv2.circle(img, (cx, cy), 12, (0, 200, 255), -1)
            else:
                shape_label = zone_from(cx, cy, 3 * w // 4, h // 2, h // 4, sr_r, SHAPE_NAMES)
                cv2.circle(img, (cx, cy), 12, (0, 255, 120), -1)

    # redraw zone bars now that we know the active labels, so they light up
    img = draw_zone_wheel(img, w // 4, h // 2, l_r, sl_r, NOTE_NAMES, root_label, (0, 200, 255))
    img = draw_zone_wheel(img, 3 * w // 4, h // 2, r_r, sr_r, SHAPE_NAMES, shape_label, (0, 255, 120))
 
    chord_text = f"{root_label or '--'} {shape_label or '--'}"
    cv2.putText(img, chord_text, (w // 2 - 80, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    return img, root_label, shape_label

def main():
    chord_player = ChordPlayer()

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        while True:
            att = 0
            success, img = cap.read()
            while not success and att < 5:
                time.sleep(0.2)
                success, img = cap.read()
                att += 1
            if not success:
                print("Failed to open camera")
                break

            img = cv2.flip(img, 1)

            h, w, _ = img.shape
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
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
                            img,
                            name,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1
                        )
                        cv2.circle(
                            img,
                            (x, y),
                            10,
                            (0, 255, 0),
                            2
                        )

            img, root_label, shape_label = draw_overlay(img, result)
            if root_label is not None and shape_label is not None:
                new_chord = (root_label, shape_label)
                if new_chord != chord_player.curr_chord:
                    chord_player.play_chord(root_label, shape_label)
            else:
                chord_player.stop()

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()