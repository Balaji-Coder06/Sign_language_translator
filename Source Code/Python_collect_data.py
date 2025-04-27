import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)

# Start webcam
cap = cv2.VideoCapture(0)  # or use 1 if 0 doesn't work

label = input("Enter the label for this sign: ")
data = []

print("Collecting data... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        
        # ✅ Draw landmarks
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Save landmark data
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        landmarks.append(label)
        data.append(landmarks)

    cv2.imshow("Collecting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Save to CSV
columns = [str(i) for i in range(len(data[0]) - 1)] + ['label']
df = pd.DataFrame(data, columns=columns)

file_name = "sign_data.csv"
if os.path.exists(file_name):
    df.to_csv(file_name, mode='a', header=False, index=False)
else:
    df.to_csv(file_name, index=False)

print("✅ Data saved successfully!")