import cv2
import mediapipe as mp
import numpy as np
import joblib
import threading
import time
import os
from collections import deque
from gtts import gTTS
from playsound import playsound
import pyttsx3

# ===== Load Model and Label Encoder =====
model = joblib.load("D:\\My_Projects\\Working On\\Sign_language_translator\\model.pkl")
label_encoder = joblib.load("D:\\My_Projects\\Working On\\Sign_language_translator\\label_encoder.pkl")

# Text-to-Speech engine setup
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # 1 is usually the female voice
engine.setProperty('rate', 150)

# ===== Text-to-Speech Functions =====
def speak_english(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

def speak_tamil(text):
    tts = gTTS(text=text, lang='ta')
    filename = "temp_tamil.mp3"
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

def speak_hindi(text):
    tts = gTTS(text=text, lang='hi')
    filename = "temp_hindi.mp3"
    tts.save(filename)
    playsound(filename)
    os.remove(filename)

selected_language = "English"
language_options = {'1': 'English', '2': 'Tamil', '3': 'Hindi'}

# ===== Label to Translation Mapping =====
label_translations = {
    "Hello": {"English": "Hello", "Tamil": "வணக்கம்", "Hindi": "नमस्ते"},
    "Yes": {"English": "Yes", "Tamil": "ஆம்", "Hindi": "हाँ"},
    "No": {"English": "No", "Tamil": "இல்லை", "Hindi": "नहीं"},
    "Stop": {"English": "Stop", "Tamil": "நிறுத்து", "Hindi": "रुको"},
    "Peace": {"English": "Peace", "Tamil": "அமைதி", "Hindi": "शांति"},
    "I Love You": {"English": "I Love You", "Tamil": "நான் உன்னை நேசிக்கிறேன்", "Hindi": "मैं तुमसे प्यार करता हूँ"}
}

# ===== MediaPipe Hands Setup =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ===== Camera Setup =====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ===== Prediction Buffer and State Variables =====
prediction_history = deque(maxlen=10)
spoken = ""
last_spoken_time = time.time()
is_speaking = False

# ===== Main Loop =====
prev_time = time.time()
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (640, 480))
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box coordinates
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            h, w, _ = img.shape
            x1, y1 = int(x_min * w), int(y_min * h)
            x2, y2 = int(x_max * w), int(y_max * h)

            # Draw rectangle around hand
            cv2.rectangle(img, (x1 - 20, y1 - 40), (x2 + 20, y2 + 20), (0, 0, 255), 2)

            # Extract and normalize landmarks
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
            wrist = landmarks[0]
            landmarks[:, 0] -= wrist[0]
            landmarks[:, 1] -= wrist[1]
            scale = np.sqrt(landmarks[12][0]**2 + landmarks[12][1]**2)
            landmarks[:, 0] /= scale
            landmarks[:, 1] /= scale

            features = landmarks.flatten().reshape(1, -1)

            if features.shape[1] == model.n_features_in_:
                prediction = model.predict(features)[0]
                label = label_encoder.inverse_transform([prediction])[0]

                # Add to prediction buffer
                prediction_history.append(label)
                most_common = max(set(prediction_history), key=prediction_history.count)

                # Get translation
                translation = label_translations.get(most_common, {}).get(selected_language, most_common)

                # Display label above hand
                cv2.rectangle(img, (x1, y1 - 65), (x1 + 200, y1 - 35), (0, 0, 0), -1)
                cv2.putText(img, most_common, (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Speak if not already spoken
                curr_time = time.time()
                if most_common != spoken and (curr_time - last_spoken_time) > 1 and not is_speaking:
                    spoken = most_common
                    last_spoken_time = curr_time
                    is_speaking = True

                    def speak():
                        if selected_language == "English":
                            speak_english(translation)
                        elif selected_language == "Tamil":
                            speak_tamil(translation)
                        elif selected_language == "Hindi":
                            speak_hindi(translation)
                        global is_speaking
                        is_speaking = False


                    threading.Thread(target=speak, daemon=True).start()

    else:
        prediction_history.clear()

    # ===== FPS Display =====
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if curr_time != prev_time else 0
    prev_time = curr_time
    cv2.putText(img, f"FPS: {fps}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    key = cv2.waitKey(1) & 0xFF

    # Language change keys
    if key == ord('1'):
        selected_language = 'English'
        print("Language changed to English")
    elif key == ord('2'):
        selected_language = 'Tamil'
        print("Language changed to Tamil")
    elif key == ord('3'):
        selected_language = 'Hindi'
        print("Language changed to Hindi")
    elif key == ord('q'):
        break

    # Display selected language at top-right corner
    lang_text = f"Language: {selected_language}"
    text_size, _ = cv2.getTextSize(lang_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    text_x = img.shape[1] - text_size[0] - 10  # 10 pixels from right edge
    text_y = 30  # 30 pixels from top
    cv2.putText(img, lang_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show video feed
    cv2.imshow("Sign Language Translator", img)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()