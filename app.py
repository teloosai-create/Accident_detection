from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import cv2
import numpy as np
from ultralytics import YOLO
import uuid
import os

app = FastAPI()

model = YOLO("yolov8n.pt")

ACCIDENT_SPEED_THRESHOLD = 5
ACCIDENT_FRAME_THRESHOLD = 15

@app.post("/detect-accident")
async def detect_accident(video: UploadFile = File(...)):
    # Save uploaded video
    file_id = str(uuid.uuid4())
    input_path = f"input_{file_id}.mp4"
    output_path = f"output_{file_id}.mp4"

    with open(input_path, "wb") as f:
        f.write(await video.read())

    cap = cv2.VideoCapture(input_path)

    prev_positions = {}
    unique_accident_vehicles = set()
    accident_frame_count = 0
    total_frames = 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        accident_detected_in_frame = False

        results = model.track(frame, persist=True)
        boxes = results[0].boxes

        for box in boxes:
            cls = int(box.cls[0])
            if cls in [2,3,5,7]: # vehicles only
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1+x2)//2
                cy = (y1+y2)//2

                obj_id = int(box.id[0]) if box.id is not None else f"{cls}_{cx}_{cy}"

                if obj_id in prev_positions:
                    px, py = prev_positions[obj_id]
                    speed = np.linalg.norm([cx - px, cy - py])

                    if speed < ACCIDENT_SPEED_THRESHOLD:
                        accident_detected_in_frame = True
                        unique_accident_vehicles.add(obj_id)

                prev_positions[obj_id] = (cx, cy)

                color = (0,0,255) if obj_id in unique_accident_vehicles else (0,255,0)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

        if accident_detected_in_frame:
            accident_frame_count += 1
            cv2.putText(frame,"ACCIDENT DETECTED!",(50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,0,255),3)

        out.write(frame)

    cap.release()
    out.release()

    decision = "ACCIDENT DETECTED" if accident_frame_count >= ACCIDENT_FRAME_THRESHOLD else "NO ACCIDENT"

    return JSONResponse({
        "total_frames": total_frames,
        "accident_frames": accident_frame_count,
        "unique_vehicle_accidents": len(unique_accident_vehicles),
        "decision": decision,
        "video_url": f"/download/{file_id}"
    })


@app.get("/download/{file_id}")
async def download_video(file_id: str):
    path = f"output_{file_id}.mp4"
    return FileResponse(path, media_type="video/mp4", filename="annotated_output.mp4")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080)