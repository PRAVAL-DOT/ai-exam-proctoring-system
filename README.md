# 🚀 AI-Based Exam Proctoring System

An intelligent real-time proctoring system built using Computer Vision to monitor user behavior during online examinations.

---

## 🧠 Overview
This system uses OpenCV's deep learning-based face detection model to track and analyze candidate activity via webcam.

It detects suspicious behavior such as:
- Absence from screen
- Presence of multiple people
- Excessive head movement

---

## ⚙️ Features
- 🎯 Real-time face detection using DNN (Caffe model)
- 🚫 Alert when no face is detected
- 👥 Detection of multiple faces with warning system
- 🔄 Movement tracking using face center displacement
- 🔊 Audio alerts using system beeps
- ⛔ Auto-stop on repeated violations

---

## 🛠️ Tech Stack
- Python
- OpenCV (cv2)
- NumPy

---

## 📂 Project Structure
│── webcamp.py
│── README.md
│── requirements.txt

---

## ▶️ How to Run

1. Install dependencies:
    pip install -r requirements.txt

2. Run the program:



---

## ⚠️ Note
Pre-trained DNN model files are not included due to size limitations.

Download the following files and place them in the project directory:
- `res10_300x300_ssd_iter_140000.caffemodel`
- `deploy.prototxt`

---

## 💡 Key Learnings
- Real-time video processing using OpenCV
- Deep learning-based face detection (DNN)
- Behavioral monitoring using rule-based logic
- Handling edge cases in live camera systems

---

## 🔮 Future Improvements
- Eye tracking (gaze detection)
- Phone detection
- Head pose estimation
- Violation logging and analytics dashboard

---
The system actively monitors and flags suspicious behavior in real-time as shown below:

## 📸 Demo
![Demo](demo.png)
