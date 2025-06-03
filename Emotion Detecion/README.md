# 🎭 Real-Time Emotion Recognition with DeepFace, MediaPipe & OpenCV

This project brings together computer vision and AI to detect human emotions in real-time using your webcam. Built with Python, it combines the power of DeepFace, MediaPipe, and OpenCV.

---

## 📌 Features

- Real-time webcam-based emotion detection  
- Displays the dominant emotion on the video feed  
- Lightweight and runs locally  
- Can be extended with more features (e.g. emotion logging, charts)

---

## 🛠️ Installation

Install the required packages using pip:

```bash
pip install opencv-python mediapipe deepface
```

(Optional for better performance):

```bash
pip install tensorflow
```

---

## 🚀 Getting Started

1. Clone this repository.
2. Run the main script:

```bash
python emotion_detector.py
```

> Press `Q` to quit the video stream.

---

## 📂 Project Structure

```
emotion-recognition/
├── emotion_detector.py       # Main script
├── README.md                 # This file
└── demo.png / demo.gif       # Optional: Preview images or video
```

---

## 🧠 How It Works

1. Captures webcam input using OpenCV  
2. Uses DeepFace to analyze facial expressions in real time  
3. Identifies the **dominant emotion** for each frame  
4. Displays the emotion as text overlay on the live feed

---

## 📸 Example Output

```
Emotion: happy 😄
```

Detected emotions may include:
- happy
- sad
- angry
- surprise
- fear
- disgust
- neutral

---

## 🌱 Future Improvements

- Add MediaPipe face mesh visualization  
- Save emotion logs over time  
- Display probability bar chart  
- Deploy as a web app using Streamlit or Gradio

---

## 🧾 Requirements Summary

- Python 3.7+
- OpenCV (`opencv-python`)
- MediaPipe
- DeepFace
- TensorFlow (optional but recommended)

---

## 🙌 Credits

This project uses:
- [DeepFace](https://github.com/serengil/deepface)
- [MediaPipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)

---

## 📄 License

This project is licensed under the **MIT License**.

---

Made with ❤️ using Python and AI.
