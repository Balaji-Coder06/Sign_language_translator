import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Gesture tracking variables
gesture_text = ""
x_positions = []
wave_detected = False

# Function: Detect if hand is a fist
def is_fist(landmarks):
    tips = [8, 12, 16, 20]
    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:  # finger is not folded
            return False
    return True

# Function: Detect if hand is open palm
def is_palm(landmarks):
    tips = [8, 12, 16, 20]
    for tip in tips:
        if landmarks[tip].y > landmarks[tip - 2].y:  # finger is bent
            return False
    return True

# Function: Detect wave motion
def detect_wave(x_list):
    if len(x_list) < 6:
        return False
    direction_changes = 0
    prev_diff = 0
    for i in range(1, len(x_list)):
        diff = x_list[i] - x_list[i - 1]
        if diff * prev_diff < 0:
            direction_changes += 1
        if diff != 0:
            prev_diff = diff
    return direction_changes >= 2  # wave if direction changes 2+ times

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            lm = handLms.landmark

            # Track fingertip movement
            index_x = lm[8].x
            x_positions.append(index_x)
            if len(x_positions) > 15:
                x_positions.pop(0)

            # Gesture logic
            if is_palm(lm):
                if detect_wave(x_positions) and not wave_detected:
                    gesture_text = "Hello 👋"
                    wave_detected = True
                elif not detect_wave(x_positions):
                    gesture_text = "Hi"
                    wave_detected = False
            elif is_fist(lm):
                gesture_text = "Yes"
                wave_detected = False

    # Display gesture text
    cv2.putText(frame, f"Gesture: {gesture_text}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Real-Time Sign Translator", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
