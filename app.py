import os
import requests
from datetime import datetime, date
import shutil
import sqlite3
import threading
import time
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(BASE_DIR, "runs", "detect", "chicken_model-3", "weights", "best.pt")
COCO_PATH        = os.path.join(BASE_DIR, "yolov8n.pt")
UPLOAD_FOLDER    = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER    = os.path.join(BASE_DIR, "static", "results")
DB_PATH          = os.path.join(BASE_DIR, "detections.db")
ALLOWED_EXT      = {"jpg", "jpeg", "png"}
CONF_THRESHOLD   = 0.30
COCO_CONF        = 0.45
CAPTURE_INTERVAL = 5      # seconds between auto captures
CAMERA_INDEX     = 1      # C270 Logitech
TOTAL_CHICKENS   = 30     # how many chickens owner has

# Classes to reject from COCO
REJECTED_CLASSES = {
    0: "person", 15: "cat", 16: "dog", 17: "horse",
    18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe",
}

app = Flask(__name__)
app.secret_key = "smartcoop_secret_2024"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ──────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────
camera             = None
cam_lock           = threading.Lock()
auto_capture_running = False
auto_capture_thread  = None

# Door state
door_state = {
    "status":          "CLOSE",   # current door status
    "manual_override": False,     # True = owner controlling manually
    "override_action": None,      # "OPEN" or "CLOSE"
    "is_daytime":      True,      # LDR reading
    "ldr_value":       0,         # raw LDR value from ESP32
    "chicken_count":   0,         # chickens counted in last frame
    "all_inside":      False,     # True when count >= TOTAL_CHICKENS
}


# ── Sunrise/Sunset for Casablanca ─────────────
sun_times = {
    "sunrise": "06:00",
    "sunset":  "19:30",
    "date":    None,
}

def fetch_sun_times():
    """Fetch today's sunrise/sunset for Casablanca from free API."""
    global sun_times
    try:
        # Casablanca coordinates: 33.5731° N, 7.5898° W
        url = "https://api.sunrise-sunset.org/json?lat=33.5731&lng=-7.5898&formatted=0"
        r   = requests.get(url, timeout=5)
        data = r.json()
        if data["status"] == "OK":
            # API returns UTC — Casablanca is UTC+1 (UTC+0 in winter)
            from datetime import datetime, timezone, timedelta
            tz_offset = timedelta(hours=1)   # Morocco is UTC+1
            sunrise_utc = datetime.fromisoformat(data["results"]["sunrise"])
            sunset_utc  = datetime.fromisoformat(data["results"]["sunset"])
            sunrise_local = (sunrise_utc + tz_offset).strftime("%H:%M")
            sunset_local  = (sunset_utc  + tz_offset).strftime("%H:%M")
            sun_times["sunrise"] = sunrise_local
            sun_times["sunset"]  = sunset_local
            sun_times["date"]    = str(date.today())
            print(f"  ☀️  Sunrise: {sunrise_local} | Sunset: {sunset_local}")
    except Exception as e:
        print(f"[SUN] Could not fetch — using defaults: {e}")

def is_daytime_now():
    fetch_sun_times()

    sunrise = sun_times["sunrise"]
    sunset = sun_times["sunset"]

    now = datetime.now().strftime("%H:%M")

    is_day = sunrise <= now <= sunset

    return is_day, sunrise, sunset
# ──────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────
def get_camera():
    global camera
    with cam_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
    return camera

# ──────────────────────────────────────────────
# DB
# ──────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name     TEXT    NOT NULL,
                detection_time TEXT    NOT NULL,
                detected       INTEGER NOT NULL,
                confidence     REAL    NOT NULL,
                door_status    TEXT    NOT NULL,
                source         TEXT    DEFAULT 'upload',
                chicken_count  INTEGER DEFAULT 0,
                is_daytime     INTEGER DEFAULT 1,
                manual_override INTEGER DEFAULT 0
            )
        """)
        conn.commit()

# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}.")
    return YOLO(MODEL_PATH)

try:
    model        = load_model()
    coco_model   = YOLO(COCO_PATH) if os.path.exists(COCO_PATH) else None
    MODEL_LOADED = True
    print(f"  COCO filter  : {'ON' if coco_model else 'OFF'}")
except Exception as e:
    model = None
    coco_model = None
    MODEL_LOADED = False
    print(f"[WARNING] {e}")

# ──────────────────────────────────────────────
# Inference — counts chickens too
# ──────────────────────────────────────────────
def run_inference(image_path: str, result_filename: str):
    """
    Returns (detected, confidence, chicken_count, out_path, rejected_as)
    chicken_count = number of chickens found in frame
    """
    if not MODEL_LOADED:
        raise RuntimeError("Model is not loaded.")

    # Stage 1 — COCO filter
    if coco_model is not None:
        coco_results = coco_model(image_path, conf=COCO_CONF)
        coco_boxes   = coco_results[0].boxes
        if coco_boxes is not None and len(coco_boxes) > 0:
            detected_ids = coco_boxes.cls.cpu().numpy().astype(int)
            for cls_id in detected_ids:
                if cls_id in REJECTED_CLASSES:
                    out_path = os.path.join(RESULT_FOLDER, result_filename)
                    shutil.copy(image_path, out_path)
                    return False, 0.0, 0, out_path, REJECTED_CLASSES[cls_id]

    # Stage 2 — Chicken model
    results = model(image_path, conf=CONF_THRESHOLD, imgsz=1024)
    result  = results[0]

    detected       = False
    confidence     = 0.0
    chicken_count  = 0

    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id     = int(box.cls[0])
            class_name = model.names[cls_id]
            conf       = float(box.conf[0])
            print(f"[DEBUG] {class_name} ({conf:.2f})")
            if class_name.lower() == "chicken" and conf > 0.40:
                chicken_count += 1
                confidence = max(confidence, conf)

        detected = chicken_count > 0

    annotated_bgr = result.plot()
    out_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(out_path, annotated_bgr)

    return detected, confidence, chicken_count, out_path, None

def decide_door(detected, chicken_count, is_daytime):
    """
    Core door logic:
    - Manual override → always wins
    - Nighttime → always CLOSE
    - All chickens inside → CLOSE
    - Chicken detected + daytime → OPEN
    - Otherwise → CLOSE
    """
    # Manual override always wins
    if door_state["manual_override"]:
        return door_state["override_action"]

    # Nighttime → always close
    if not is_daytime:
        return "CLOSE"

    # All chickens accounted for → close
    if chicken_count >= TOTAL_CHICKENS:
        return "CLOSE"

    # Chicken detected + daytime → open
    if detected and is_daytime:
        return "OPEN"

    return "CLOSE"

def save_detection(image_name, detected, confidence, door_action,
                   chicken_count=0, is_daytime=True, manual=False, source="auto"):
    detection_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    db = get_db()
    db.execute(
        "INSERT INTO detections "
        "(image_name, detection_time, detected, confidence, door_status, "
        "source, chicken_count, is_daytime, manual_override) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (image_name, detection_dt, int(detected), round(confidence, 4),
         door_action, source, chicken_count, int(is_daytime), int(manual))
    )
    db.commit()
    db.close()
    return detection_dt

# ──────────────────────────────────────────────
# Auto-capture background thread
# ──────────────────────────────────────────────
def auto_capture_loop():
    global auto_capture_running
    print(f"[AUTO] Started — every {CAPTURE_INTERVAL}s")

    while auto_capture_running:
        try:
            cap = get_camera()
            with cam_lock:
                ret, frame = cap.read()

            if ret:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_name   = f"auto_{timestamp_str}.jpg"
                upload_path   = os.path.join(UPLOAD_FOLDER, unique_name)
                cv2.imwrite(upload_path, frame)

                result_name = f"result_{unique_name}"
                detected, confidence, chicken_count, _, rejected_as = run_inference(
                    upload_path, result_name
                )

                is_daytime, sunrise, sunset = is_daytime_now()
                door_state["is_daytime"] = is_daytime
                door_action = decide_door(detected, chicken_count, is_daytime)

                # Update global door state
                door_state["status"]        = door_action
                door_state["chicken_count"] = chicken_count
                door_state["all_inside"]    = chicken_count >= TOTAL_CHICKENS

                save_detection(
                    unique_name, detected, confidence, door_action,
                    chicken_count, is_daytime, False, "auto"
                )

                status_icon = "🐔" if detected else "❌"
                day_icon    = "☀️" if is_daytime else "🌙"
                print(f"[AUTO] {datetime.now().strftime('%H:%M:%S')} "
                      f"{status_icon} {chicken_count}/{TOTAL_CHICKENS} chickens "
                      f"{day_icon} → Door {door_action}")
            else:
                print("[AUTO] Could not read frame")

        except Exception as e:
            print(f"[AUTO ERROR] {e}")

        time.sleep(CAPTURE_INTERVAL)

    print("[AUTO] Stopped")

def start_auto_capture():
    global auto_capture_running, auto_capture_thread
    if not auto_capture_running:
        auto_capture_running = True
        auto_capture_thread  = threading.Thread(target=auto_capture_loop, daemon=True)
        auto_capture_thread.start()

def stop_auto_capture():
    global auto_capture_running
    auto_capture_running = False

# ──────────────────────────────────────────────
# Routes — Dashboard
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return redirect(url_for("dashboard"))

@app.route("/dashboard")
def dashboard():
    db       = get_db()
    total    = db.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    chickens = db.execute("SELECT COUNT(*) FROM detections WHERE detected=1").fetchone()[0]
    last_row = db.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    last_time = last_row["detection_time"] if last_row else "—"
    last_img  = f"results/result_{last_row['image_name']}" if last_row else ""
    db.close()

# Get current day/night and sunrise/sunset
    is_day, sunrise, sunset = is_daytime_now()

    stats = {
    "total":           total,
    "chickens":        chickens,
    "last_time":       last_time,
    "door_status":     door_state["status"],
    "auto_running":    auto_capture_running,
    "interval":        CAPTURE_INTERVAL,
    "last_img":        last_img,
    "manual_override": door_state["manual_override"],

    # NEW
    "is_daytime":      is_day,
    "sunrise":         sunrise,
    "sunset":          sunset,

    "ldr_value":       door_state["ldr_value"],
    "chicken_count":   door_state["chicken_count"],
    "total_chickens":  TOTAL_CHICKENS,
    "all_inside":      door_state["all_inside"],
}
    return render_template("dashboard.html", stats=stats)
# ── Auto capture control ──────────────────────
@app.route("/auto/start", methods=["POST"])
def auto_start():
    start_auto_capture()
    flash(f"Auto detection started — every {CAPTURE_INTERVAL}s.", "success")
    return redirect(url_for("dashboard"))

@app.route("/auto/stop", methods=["POST"])
def auto_stop():
    stop_auto_capture()
    flash("Auto detection stopped.", "warning")
    return redirect(url_for("dashboard"))

# ── Manual door override ──────────────────────
@app.route("/door/open", methods=["POST"])
def door_open():
    door_state["manual_override"] = True
    door_state["override_action"] = "OPEN"
    door_state["status"]          = "OPEN"
    save_detection("manual", False, 0.0, "OPEN", 0,
                   door_state["is_daytime"], True, "manual")
    flash("Door manually OPENED.", "success")
    return redirect(url_for("dashboard"))

@app.route("/door/close", methods=["POST"])
def door_close():
    door_state["manual_override"] = True
    door_state["override_action"] = "CLOSE"
    door_state["status"]          = "CLOSE"
    save_detection("manual", False, 0.0, "CLOSE", 0,
                   door_state["is_daytime"], True, "manual")
    flash("Door manually CLOSED.", "success")
    return redirect(url_for("dashboard"))

@app.route("/door/auto", methods=["POST"])
def door_auto():
    """Release manual override — back to AI control."""
    door_state["manual_override"] = False
    door_state["override_action"] = None
    flash("Door control returned to AI.", "success")
    return redirect(url_for("dashboard"))

# ── ESP32 sensor update ───────────────────────
@app.route("/api/sensor", methods=["POST"])
def api_sensor():
    """ESP32 still sends LDR but day/night is now time-based."""
    data = request.get_json(silent=True) or {}
    ldr  = int(data.get("ldr", 0))
    door_state["ldr_value"] = ldr
    # Day/night now decided by time, not LDR
    is_day, sunrise, sunset = is_daytime_now()
    door_state["is_daytime"] = is_day
    return jsonify({
        "is_daytime": is_day,
        "sunrise":    sunrise,
        "sunset":     sunset,
        "ldr":        ldr,
    })

# ── API status ────────────────────────────────
@app.route("/api/status")
def api_status():
    db       = get_db()
    row      = db.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    total    = db.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    chickens = db.execute("SELECT COUNT(*) FROM detections WHERE detected=1").fetchone()[0]
    recent   = db.execute(
        "SELECT detected, door_status, detection_time, chicken_count, is_daytime, manual_override "
        "FROM detections ORDER BY id DESC LIMIT 5"
    ).fetchall()
    activity = [{
        "detected": bool(r["detected"]),
        "door":     r["door_status"],
        "time":     r["detection_time"],
        "count":    r["chicken_count"],
        "daytime":  bool(r["is_daytime"]),
        "manual":   bool(r["manual_override"]),
    } for r in recent]
    db.close()

    result_img = f"/static/results/result_{row['image_name']}" if row else ""
    return jsonify({
        "door_status":     door_state["status"],
        "detected":        bool(row["detected"]) if row else False,
        "confidence":      round((row["confidence"] * 100), 1) if row else 0,
        "time":            row["detection_time"] if row else "—",
        "result_img":      result_img,
        "total":           total,
        "chickens":        chickens,
        "auto_running":    auto_capture_running,
        "manual_override": door_state["manual_override"],
        "is_daytime":      door_state["is_daytime"],
        "ldr_value":       door_state["ldr_value"],
        "chicken_count":   door_state["chicken_count"],
        "total_chickens":  TOTAL_CHICKENS,
        "all_inside":      door_state["all_inside"],
        "activity":        activity,
    })

@app.route("/api/door/status")
def api_door_status():
    return door_state["status"], 200, {"Content-Type": "text/plain"}

# ──────────────────────────────────────────────
# Routes — History / Cleanup / Delete
# ──────────────────────────────────────────────
@app.route("/history")
def history():
    db   = get_db()
    rows = db.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 100").fetchall()
    db.close()
    return render_template("history.html", detections=rows)

@app.route("/cleanup", methods=["POST"])
def cleanup():
    db = get_db()
    rows = db.execute("SELECT id, image_name FROM detections").fetchall()
    deleted = 0
    for row in rows:
        up = os.path.join(UPLOAD_FOLDER, row["image_name"])
        re = os.path.join(RESULT_FOLDER, f"result_{row['image_name']}")
        if not os.path.exists(up) and not os.path.exists(re):
            db.execute("DELETE FROM detections WHERE id=?", (row["id"],))
            deleted += 1
    db.commit()
    db.close()
    flash(f"Cleaned up {deleted} orphaned record(s).", "success")
    return redirect(url_for("history"))

@app.route("/delete/<int:detection_id>", methods=["POST"])
def delete_detection(detection_id):
    db  = get_db()
    row = db.execute("SELECT * FROM detections WHERE id=?", (detection_id,)).fetchone()
    if row:
        for folder, prefix in [(UPLOAD_FOLDER, ""), (RESULT_FOLDER, "result_")]:
            path = os.path.join(folder, prefix + row["image_name"])
            if os.path.exists(path):
                os.remove(path)
        db.execute("DELETE FROM detections WHERE id=?", (detection_id,))
        db.commit()
        flash("Detection deleted.", "success")
    db.close()
    return redirect(url_for("history"))

# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print("=" * 55)
    print("  🐔  Smart Chicken Coop AI — Dashboard")
    print("=" * 55)
    print(f"  Model        : chicken_model-3 (yolov8m)")
    print(f"  Model loaded : {MODEL_LOADED}")
    print(f"  Camera       : index {CAMERA_INDEX} (C270)")
    print(f"  Total chicks : {TOTAL_CHICKENS}")
    print(f"  Auto capture : every {CAPTURE_INTERVAL}s")
    print(f"  URL          : http://127.0.0.1:5000")
    print("=" * 55)
    if MODEL_LOADED:
        start_auto_capture()
    app.run(debug=False, host="0.0.0.0", port=5000)