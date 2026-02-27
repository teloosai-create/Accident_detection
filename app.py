from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

app = Flask(__name__)
CORS(app)

model = YOLO("yolov8n.pt")

ACCIDENT_SPEED_THRESHOLD = 5
ACCIDENT_FRAME_THRESHOLD = 15
TARGET_CLASSES = [0, 2, 3, 5, 7]

@app.route("/detect", methods=["POST"])
def detect_accident():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["video"]
    
    # Save to a temporary file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    file.save(temp_video.name)

    cap = cv2.VideoCapture(temp_video.name)

    prev_positions = {}
    accident_frame_count = 0
    total_frames = 0
    unique_accident_vehicles = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        results = model.track(frame, persist=True, verbose=False)
        accident_frame = False
        
        if results:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in TARGET_CLASSES:

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2)//2, (y1 + y2)//2

                    obj_id = int(box.id[0]) if box.id is not None else None

                    if cls_id in [2, 3, 5, 7] and obj_id is not None:
                        if obj_id in prev_positions:
                            px, py = prev_positions[obj_id]
                            dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                            if dist < ACCIDENT_SPEED_THRESHOLD:
                                accident_frame = True
                                unique_accident_vehicles.add(obj_id)
                        prev_positions[obj_id] = (cx, cy)

        if accident_frame:
            accident_frame_count += 1

    cap.release()
    os.remove(temp_video.name)

    detected = accident_frame_count >= ACCIDENT_FRAME_THRESHOLD

    return jsonify({
        "accident_detected": detected,
        "total_frames": total_frames,
        "accident_frames": accident_frame_count,
        "vehicles_involved": len(unique_accident_vehicles)
    })