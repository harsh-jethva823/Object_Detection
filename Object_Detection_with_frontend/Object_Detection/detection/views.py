from django.shortcuts import render

from django.http import StreamingHttpResponse
import cv2
import torch
from ultralytics import YOLO

model = YOLO("yolov9c.pt")  # Ensure this file is in your project directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
cap = cv2.VideoCapture(0)

def index(request):
    return render(request, 'detection/index.html')
def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model(frame)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                label_id = int(box.cls[0].item())
                confidence = box.conf[0].item()
                class_label = model.names[label_id]
                label_text = f"{class_label}: {confidence:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
                                 content_type="multipart/x-mixed-replace; boundary=frame")
