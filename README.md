# Exam-Proctoring-System


# 🕵️‍♂️ AI Proctor

A modern, AI-powered online exam proctoring system with real-time object detection, proctor dashboard, and anti-cheating measures.

---

## 📋 Table of Contents

- [🚀 Features](#-features)
- [🛠 Tech Stack](#-tech-stack)
- [⚙ Installation](#-installation)
- [🔧 Configuration](#-configuration)
- [📁 Project Structure](#-project-structure)
- [📚 Usage Guide](#-usage-guide)
- [🚀 Deployment](#-deployment)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)
- [📞 Support](#-support)

---

## 🚀 Features

- 👤 **User Authentication** (Student login with roll number)
- 📝 **Online Exam Interface** (MCQs, timer, auto-submit)
- 🖥 **Fullscreen Enforcement** (Auto-submit on exit)
- 🎥 **Webcam Proctoring** (Face registration, live face verification)
- 🤳 **Object Detection** (Detects phones, laptops, etc. and auto-submits)
- 📊 **Proctor Dashboard** (Live alerts, analytics, and charts)
- 🛡 **Anti-cheating Measures** (Logs, auto-submission, and alerts)

---

## 🛠 Tech Stack

- **Frontend:** React, Bootstrap, React Webcam, Recharts
- **Backend:** Flask, Flask-CORS, OpenCV, YOLOv5 (PyTorch)
- **AI/ML:** YOLOv5 (object detection), OpenCV
- **Database:** PostgreSQL
- **Other:** Python, JavaScript, HTML/CSS

---

## ⚙ Installation

1. **Clone the repository:**

   cd Ai-proctor
   

2. **Frontend Setup:**
   ```sh
   cd ai-proctor
   npm install
   ```

3. **Backend Setup:**
   ```sh
   cd backend
   python -m venv venv
   venv/Scripts/activate  # On Windows
   # or
   source venv/bin/activate  # On Mac/Linux

   pip install -r requirements.txt
   ```

---

## 🔧 Configuration

- **Frontend:**  
  - Update API URLs in React to point to your deployed backend.
- **Backend:**  
  - YOLOv5 model weights (`yolov5m.pt`) are downloaded from Google Drive on first run.
  - Set CORS as needed for your deployment.
  - Configure PostgreSQL connection in `app.py` or via environment variables.

---

## 📁 Project Structure

```
ai-proctor/
├── ai-proctor/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.css
│   │   ├── App.js
│   │   ├── index.js
│   │   ├── components/
│   │   │   ├── Exam.js
│   │   │   ├── Instruction.js
│   │   │   ├── Login.js
│   │   │   └── NotFound.js
│   │   └── pages/
│   │       └── ProctorDashboard.js
│   ├── package.json
│   └── ...
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   └── (yolov5m.pt downloaded at runtime)
├── .gitignore
└── README.md
```

---

## 📚 Usage Guide

1. **Start the backend:**
   ```sh
   cd backend
   venv/Scripts/activate  # or source venv/bin/activate
   python app.py
   ```

2. **Start the frontend:**
   ```sh
   cd ai-proctor
   npm start
   ```

3. **Workflow:**
   - Student logs in with roll number.
   - Reads instructions and acknowledges.
   - Registers face before exam starts.
   - Exam runs in fullscreen; webcam monitors face and objects.
   - Alerts and logs are sent to the proctor dashboard.
   - Proctor can view live analytics and logs.

---

## 🚀 Deployment

- **Frontend:** Deploy as a static site (e.g., Render Static Site).
- **Backend:** Deploy as a Python web service (e.g., Render Web Service).
- **Database:** Use Render PostgreSQL or your own PostgreSQL instance.
- **Environment Variables:** Set database connection strings and API URLs as needed.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)
- [React](https://reactjs.org/)
- [Bootstrap](https://getbootstrap.com/)
- [Recharts](https://recharts.org/)
- Open source contributors and the AI/ML community
AI/CV: OpenCV, MediaPipe, dlib
Audio: Web Audio API + custom voice detection
Realtime: WebSocket

License
MIT License © 2025 Sonali Gupta
