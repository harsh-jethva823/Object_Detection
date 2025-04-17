# Step 1: Import necessary libraries
import cv2
import torch
from ultralytics import YOLO

# Step 2: Load the YOLOv9 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO('yolov9c.pt')
model.to(device)

# Step 3: Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture device")
    exit()

# Step 4: Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Step 5: Use YOLOv9 model to make predictions
    results = model(frame)

    # Step 6: Draw bounding boxes and labels
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            label_id = int(box.cls[0].item())
            confidence = box.conf[0].item()
            class_label = model.names[label_id]
            label_text = f"{class_label}: {confidence:.2f}"

            # Draw rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Draw label
            cv2.putText(frame, label_text, (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Step 7: Display the frame
    cv2.imshow('Real-Time Object Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 8: Release resources
cap.release()
cv2.destroyAllWindows()
