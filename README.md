✋ Sign Language Translator
===========================

A real-time sign language to speech translator using OpenCV, MediaPipe, and a trained machine learning model. This system recognizes basic hand gestures and speaks out the detected sign in English, Tamil, Hindi, or Japanese.

🎥 Demo Preview
> _Add a demo GIF or YouTube link here if available_

📌 Features
-----------
- 🤚 Real-time hand gesture detection using MediaPipe
- 🧠 Pre-trained ML model to predict signs from hand landmarks
- 🗣️ Multilingual speech output: English, Tamil, Hindi, and Japanese
- 🔄 Dynamic language switching (press keys 1–4)
- 📊 FPS counter and gesture overlay on video
- 🧠 Smooth prediction with a rolling window (debouncing false triggers)

🧑‍💻 Tech Stack
---------------
| Component     | Description                          |
|---------------|--------------------------------------|
| OpenCV        | Real-time video processing           |
| MediaPipe     | Hand landmark detection              |
| scikit-learn  | ML model for gesture classification  |
| pyttsx3       | Offline English speech synthesis     |
| gTTS          | Online multilingual speech synthesis |
| playsound     | Audio playback                       |
| joblib        | Model and label encoder loading      |

📦 Requirements
---------------
Install the dependencies using:

    pip install opencv-python mediapipe numpy scikit-learn gTTS playsound pyttsx3 joblib

🧠 Pre-trained Model
---------------------
Ensure the following files are present in your project folder:

- model.pkl — Trained ML model
- label_encoder.pkl — Encodes/decodes label outputs

🚀 How to Run
-------------
    python main.py

- Press 1 for English  
- Press 2 for Tamil  
- Press 3 for Hindi  
- Press 4 for Japanese  
- Press q to Quit

🖐️ Supported Gestures
----------------------
| Gesture        | Tamil               | Hindi            | Japanese     |
|----------------|---------------------|------------------|--------------|
| Hello          | வணக்கம்              | नमस्ते           | こんにちは     |
| Yes            | ஆம்                 | हाँ              | はい          |
| No             | இல்லை              | नहीं             | いいえ         |
| Stop           | நிறுத்து             | रुको             | 止まる         |
| Peace          | அமைதி               | शांति            | 平和           |
| I Love You     | நான் உன்னை நேசிக்கிறேன் | मैं तुमसे प्यार करता हूँ | 愛してる    |

🎮 Controls
-----------
| Key | Action              |
|-----|---------------------|
| 1   | Speak in English    |
| 2   | Speak in Tamil      |
| 3   | Speak in Hindi      |
| 4   | Speak in Japanese   |
| q   | Quit the program    |

📂 Project Structure
--------------------
Sign_Language_Translator/
├── main.py
├── model.pkl
├── label_encoder.pkl
└── README.md

🙋‍♂️ Author
-----------
S Balaji  
Aspiring Software Engineer | C & Python Developer  
🔗 GitHub: https://github.com/Balaji-Coder06  
🔗 LinkedIn: https://www.linkedin.com/in/s-balaji06/

🛡 Disclaimer
-------------
This project recognizes limited signs and is meant for educational and prototype purposes only. Full sign language interpretation requires larger datasets and robust models.
