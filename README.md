# Exam-Proctoring-System


**Real-time cheating detection** - Gaze | Multiple faces | Tab switch | Phone | Voice | Auto-screenshot | Scorecard

[![Stars](https://img.shields.io/github/stars/sonaleevgupta/Exam-Proctoring-System?style=social)](https://github.com/sonaleevgupta/Exam-Proctoring-System)

## Features
- Real-time face & eye tracking (MediaPipe)
- Detects multiple faces, tab switching, phone usage
- Detects multiple voices & background noise
- Auto-screenshot on violation
- Gaze estimation & head pose detection
- Live proctor dashboard with WebSocket
- Screen + webcam recording
- Auto-generates scorecard after exam

## Architecture
Exam-Proctoring-System/
├── frontend/          ← React + Vite + Tailwind (submodule)
├── backend/           ← Flask + OpenCV + SocketIO
├── README.md
└── .gitmodules

## Quick Start

```bash
# Clone the full project
git clone https://github.com/sonaleevgupta/Exam-Proctoring-System
cd Exam-Proctoring-System

# Initialize frontend submodule
git submodule update --init --recursive

# Start Backend
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python app.py

# Start Frontend (open new terminal)
cd ../frontend
npm install
npm run dev

Open http://localhost:5173 → Exam starts with full AI proctoring!

##Tech Stack

Frontend: React 18, Vite, Tailwind CSS
Backend: Python Flask, Flask-SocketIO
AI/CV: OpenCV, MediaPipe, dlib
Audio: Web Audio API + custom voice detection
Realtime: WebSocket

License
MIT License © 2025 Sonali Gupta
