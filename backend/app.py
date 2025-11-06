from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import torch
import mysql.connector
import os
from datetime import datetime
import json
import gdown

# -------------------------
# Flask & CORS
# -------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -------------------------
# MySQL Database Configuration
# -------------------------
DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,  # MySQL default port
    "database": "proctor",
    "user": "root",      # change if needed
    "password": ""       # change if needed
}

def get_db_connection():
    """Create and return a database connection."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

def init_database():
    """
    Ensure database connectivity and create required tables if they do not exist.
    """
    conn = get_db_connection()
    if not conn:
        print("Failed to connect to database. Please check your database configuration.")
        return

    try:
        cursor = conn.cursor()

        # Create student table if missing
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS student (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100),
                roll_number VARCHAR(50) UNIQUE,
                password VARCHAR(255),
                score INT
            )
        """)

        # Create alerts table if missing
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INT AUTO_INCREMENT PRIMARY KEY,
                student_id INT,
                direction VARCHAR(100),
                alert_time DATETIME,
                details JSON,
                FOREIGN KEY (student_id) REFERENCES student(id)
            )
        """)

        conn.commit()

        # Small checks
        cursor.execute("SELECT COUNT(*) FROM student")
        student_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM alerts")
        alerts_count = cursor.fetchone()[0]

        print("Database connection successful!")
        print(f"Student table has {student_count} records")
        print(f"Alerts table has {alerts_count} records")
        print("Using existing database tables and data.")

    except Exception as e:
        print(f"Database check/init error: {e}")
        print("Please ensure your database user has permission to create tables.")
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass

# -------------------------
# Face utils (MediaPipe)
# -------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# In-memory storage for "registered" roll numbers (demo only)
registered_faces = set()

# 3D model points & thresholds for head pose
model_points = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float64)

landmark_ids = [1, 152, 263, 33, 287, 57]
YAW_THRESHOLD, PITCH_THRESHOLD, ROLL_THRESHOLD = 30, 20, 30

# -------------------------
# YOLO (Ultralytics) setup
# -------------------------
from ultralytics import YOLO

MODEL_PATH = "yolov5mu.pt"  # your custom path
GDRIVE_ID = "1cfF-h42hdcfYdqzUxyE38qXryUddtT-n"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"

def load_yolo_model():
    """
    Try to load your custom yolov5mu.pt (download if missing),
    else gracefully fall back to yolov8n.pt (official tiny model).
    """
    # If custom file exists, load it
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Loading YOLO model from {MODEL_PATH} ...")
            return YOLO(MODEL_PATH)
        except Exception as e:
            print(f"Failed to load {MODEL_PATH}: {e}. Falling back to yolov8n.pt")

    # Try downloading custom file if missing
    if not os.path.exists(MODEL_PATH):
        try:
            print("Downloading YOLO model from Google Drive ...")
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
            if os.path.exists(MODEL_PATH):
                print(f"Downloaded to {MODEL_PATH}, loading ...")
                return YOLO(MODEL_PATH)
        except Exception as e:
            print(f"Download failed or not accessible: {e}. Falling back to yolov8n.pt")

    # Fallback to yolov8n
    print("Loading fallback model yolov8n.pt ...")
    return YOLO("yolov8n.pt")

model = load_yolo_model()

# -------------------------
# Routes
# -------------------------

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json(silent=True) or {}
    username = data.get('username')
    roll_number = data.get('rollNumber')
    password = data.get('password')

    if not username or not roll_number or not password:
        return jsonify({'message': 'Missing fields'}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({'message': 'Database connection failed'}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM student WHERE roll_number = %s", (roll_number,))
        if cursor.fetchone():
            return jsonify({'message': 'Roll number already registered'}), 409

        cursor.execute(
            "INSERT INTO student (username, roll_number, password) VALUES (%s, %s, %s)",
            (username, roll_number, password)
        )
        conn.commit()
        return jsonify({'message': 'Registration successful'}), 201
    except Exception as e:
        print(f"Registration error: {e}")
        return jsonify({'message': 'Database error'}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json(silent=True) or {}
    roll_number = data.get('rollNumber')
    password = data.get('password')

    if not roll_number or not password:
        return jsonify({'message': 'Missing fields'}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({'message': 'Database connection failed'}), 500

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM student WHERE roll_number = %s AND password = %s",
            (roll_number, password)
        )
        student = cursor.fetchone()
        if student:
            return jsonify({'message': 'Login successful'})
        else:
            return jsonify({'message': 'Invalid credentials'}), 401
    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'message': 'Database error'}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass

@app.route('/submit-score', methods=['POST'])
def submit_score():
    data = request.get_json(silent=True) or {}
    roll_number = data.get('rollNumber')
    score = data.get('score')

    if not roll_number or score is None:
        return jsonify({'message': 'Missing rollNumber or score'}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({'message': 'Database connection failed'}), 500

    try:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE student SET score = %s WHERE roll_number = %s",
            (score, roll_number)
        )
        if cursor.rowcount == 0:
            return jsonify({'message': 'Student not found'}), 404
        conn.commit()
        return jsonify({'message': 'Score submitted successfully'}), 200
    except Exception as e:
        print(f"Score submission error: {e}")
        return jsonify({'message': 'Database error'}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass
        

@app.route('/score/<roll_number>', methods=['GET'])
def get_score(roll_number):
    conn = get_db_connection()
    if not conn:
        return jsonify({'message': 'Database connection failed'}), 500

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT score FROM student WHERE roll_number = %s",
            (roll_number,)
        )
        result = cursor.fetchone()
        if result and result[0] is not None:
            return jsonify({'score': result[0], 'total': 10}), 200
        else:
            return jsonify({'message': 'Score not found or student not registered'}), 404
    except Exception as e:
        print(f"Score retrieval error: {e}")
        return jsonify({'message': 'Database error'}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass

@app.route('/get-student-id/<roll_number>', methods=['GET'])
def get_student_id(roll_number):
    conn = get_db_connection()
    if not conn:
        return jsonify({'message': 'Database connection failed'}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM student WHERE roll_number = %s", (roll_number,))
        result = cursor.fetchone()
        if result:
            return jsonify({'student_id': result[0]}), 200
        else:
            return jsonify({'message': 'Student not found'}), 404
    except Exception as e:
        print(f"Error fetching student ID: {e}")
        return jsonify({'message': 'Database error'}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass

@app.route('/scores', methods=['GET'])
def get_all_scores():
    conn = get_db_connection()
    if not conn:
        return jsonify({'message': 'Database connection failed'}), 500

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT username, roll_number, score FROM student WHERE score IS NOT NULL"
        )
        results = cursor.fetchall()
        scores = [
            {'username': row[0], 'roll_number': row[1], 'score': row[2]}
            for row in results
        ]
        return jsonify({'scores': scores}), 200
    except Exception as e:
        print(f"All scores retrieval error: {e}")
        return jsonify({'message': 'Database error'}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass

@app.route('/log-alert', methods=['POST'])
def log_alert():
    data = request.get_json(silent=True) or {}
    student_id = data.get('student_id')
    direction = data.get('direction')
    t = data.get('time')  # provided by client

    if not (student_id and direction):
        return jsonify({'status': 'error', 'message': 'Missing student_id or direction'}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({'status': 'error', 'message': 'Database connection failed'}), 500

    try:
        cursor = conn.cursor()
        # Store the entire data object as JSON in details column
        cursor.execute("""
            INSERT INTO alerts (student_id, direction, alert_time, details)
            VALUES (%s, %s, %s, %s)
        """, (student_id, direction, datetime.now(), json.dumps(data)))
        conn.commit()
        print(f"Alert logged successfully: student_id={student_id}, direction={direction}")
        return jsonify({'status': 'ok'})
    except Exception as e:
        print(f"Database logging error: {e}")
        return jsonify({'status': 'error', 'message': 'Database error'}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass

@app.route('/alerts', methods=['GET'])
def get_alerts():
    conn = get_db_connection()
    if not conn:
        return jsonify({'message': 'Database connection failed'}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.student_id, s.username, s.roll_number, a.direction, a.alert_time, a.details
            FROM alerts a
            JOIN student s ON a.student_id = s.id
            ORDER BY a.alert_time DESC
        """)
        results = cursor.fetchall()
        alerts = []
        for row in results:
            student_id, username, roll_number, direction, alert_time, details_raw = row
            details_obj = {}
            if details_raw:
                if isinstance(details_raw, (dict, list)):
                    details_obj = details_raw
                else:
                    try:
                        details_obj = json.loads(details_raw)
                    except Exception:
                        details_obj = {"raw": str(details_raw)}

            alerts.append({
                'student_id': student_id,
                'username': username,
                'roll_number': roll_number,
                'direction': direction,
                'time': alert_time.isoformat() if hasattr(alert_time, "isoformat") else str(alert_time),
                'details': details_obj
            })
        return jsonify(alerts), 200
    except Exception as e:
        print(f"Alerts retrieval error: {e}")
        return jsonify({'message': 'Database error'}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass

@app.route('/submit-exam', methods=['POST'])
def submit_exam():
    data = request.json
    student_id = data.get('student_id')
    score = data.get('score')  # score calculated from the exam

    if student_id is None or score is None:
        return jsonify({'message': 'Invalid data'}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({'message': 'Database connection failed'}), 500

    try:
        cursor = conn.cursor()
        # Update the student's score in the table
        cursor.execute(
            "UPDATE student SET score = %s WHERE id = %s",
            (score, student_id)
        )
        conn.commit()
        return jsonify({'message': 'Score submitted successfully'}), 200
    except Exception as e:
        print(f"Error updating score: {e}")
        return jsonify({'message': 'Failed to submit score'}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass

@app.route('/detect-head', methods=['POST'])
def detect_head():
    if 'image' not in request.files:
        return jsonify({'message': 'No image file provided'}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'message': 'Invalid image'}), 400

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    direction, yaw, pitch, roll = "No face detected", 0.0, 0.0, 0.0

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        image_points = []
        for idx in landmark_ids:
            pt = face_landmarks.landmark[idx]
            x, y = int(pt.x * w), int(pt.y * h)
            image_points.append((x, y))

        image_points = np.array(image_points, dtype=np.float64)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success:
            rmat, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch, yaw, roll = [float(a) for a in angles]

            direction = "Looking Forward"
            if yaw > YAW_THRESHOLD:
                direction = "ALERT: Looking Right"
            elif yaw < -YAW_THRESHOLD:
                direction = "ALERT: Looking Left"
            elif pitch > PITCH_THRESHOLD:
                direction = "ALERT: Looking Down"
            elif pitch < -PITCH_THRESHOLD:
                direction = "ALERT: Looking Up"
            elif abs(roll) > ROLL_THRESHOLD:
                direction = "ALERT: Tilting Head"

    return jsonify({'direction': direction, 'yaw': yaw, 'pitch': pitch, 'roll': roll})

@app.route('/scores-with-alerts', methods=['GET'])
def get_scores_with_alerts():
    conn = get_db_connection()
    if not conn:
        return jsonify({'message': 'Database connection failed'}), 500

    try:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT s.username, s.roll_number, s.score,
           (SELECT COUNT(*) FROM alerts a WHERE a.student_id = s.id) as alert_count
        FROM student s
         WHERE s.score IS NOT NULL
        """)
        results = cursor.fetchall()
        data = [
            {'username': row[0], 'roll_number': row[1], 'score': row[2], 'alert_count': row[3]}
            for row in results
        ]
        return jsonify({'scores': data}), 200
    except Exception as e:
        print(f"Scores with alerts retrieval error: {e}")
        return jsonify({'message': 'Database error'}), 500
    finally:
        try:
            cursor.close()
            conn.close()
        except:
            pass

@app.route('/register-face', methods=['POST'])
def register_face():
    roll_number = request.form.get('roll_number')
    if not roll_number:
        return jsonify({'status': 'missing_roll_number'}), 400

    if 'image' not in request.files:
        return jsonify({'status': 'no_face'}), 400

    npimg = np.frombuffer(request.files['image'].read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'status': 'no_face'})
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if rgb is None or rgb.size == 0:
        return jsonify({'status': 'no_face'})

    results = face_detection.process(rgb)
    if results.detections:
        if len(results.detections) == 1:
            registered_faces.add(roll_number)
            return jsonify({'status': 'registered'})
        else:
            return jsonify({'status': 'multiple_faces'})
    else:
        return jsonify({'status': 'no_face'})

@app.route('/verify-face', methods=['POST'])
def verify_face():
    roll_number = request.form.get('roll_number')
    if not roll_number:
        return jsonify({'status': 'missing_roll_number'}), 400

    if 'image' not in request.files:
        return jsonify({'status': 'no_face'}), 400

    npimg = np.frombuffer(request.files['image'].read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'status': 'no_face'})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if not results.detections:
        return jsonify({'status': 'no_face'})
    elif len(results.detections) > 1:
        return jsonify({'status': 'multiple_faces'})
    else:
        if roll_number in registered_faces:
            return jsonify({'status': 'match'})
        else:
            return jsonify({'status': 'mismatch'})

@app.route('/detect-object', methods=['POST'])
def detect_object():
    if 'image' not in request.files:
        return jsonify({'message': 'No image file provided'}), 400

    npimg = np.frombuffer(request.files['image'].read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'message': 'Invalid image'}), 400

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        results_list = model(rgb)  
        results = results_list[0]  

    # extract boxes and class names
    boxes = getattr(results, "boxes", [])
    names = getattr(results, "names", {})

    data = []
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            name = names.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            data.append({
                'name': name,
                'confidence': conf,
                'xmin': x1,
                'ymin': y1,
                'xmax': x2,
                'ymax': y2
            })

    # confidence filter
    data = [d for d in data if d['confidence'] > 0.5]
    labels = [d['name'] for d in data]

    # Check for forbidden objects
    forbidden = {'cell phone', 'laptop'}
    detected = [label for label in labels if label in forbidden]

    if detected:
        return jsonify({'status': 'forbidden_object', 'objects': detected})
    else:
        return jsonify({'status': 'clear'})

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    init_database()
    # For local dev: http://127.0.0.1:5000
    app.run(host='0.0.0.0', port=5000, debug=True)