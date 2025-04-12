import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import time
from collections import deque
from sklearn.preprocessing import LabelEncoder

# Load the model and label encoder
model = joblib.load("D:\\My_Projects\\Working On\\Sign_language_translator\\model.pkl")
label_encoder = joblib.load("D:\\My_Projects\Working On\\Sign_language_translator\\label_encoder.pkl")  # This will decode labels

# TTS setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

spoken = ""
last_spoken_time = time.time()
prediction_history = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) == model.n_features_in_:
            prediction = model.predict([landmarks])[0]
            label = label_encoder.inverse_transform([prediction])[0]

            prediction_history.append(label)
            most_common = max(set(prediction_history), key=prediction_history.count)

            # Display on screen
            cv2.putText(frame, f'Sign: {most_common}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # Speak only if new and time passed
            current_time = time.time()
            if most_common != spoken and (current_time - last_spoken_time) > 1:
                engine.say(most_common)
                engine.runAndWait()
                spoken = most_common
                last_spoken_time = current_time

    cv2.imshow("Sign Language Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
