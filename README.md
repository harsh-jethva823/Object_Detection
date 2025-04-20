# Real-Time Object Detection using YOLOv9 with Django Frontend

This project demonstrates real-time object detection using the YOLOv9 model with OpenCV and PyTorch. The script captures video from the webcam, processes each frame using a pre-trained YOLOv9 model, and displays bounding boxes and labels for detected objects. The application uses Django to integrate the backend with a user-friendly frontend built using HTML and CSS.

## Features

- Real-time video capture using OpenCV
- Object detection using the pre-trained YOLOv9
- Bounding boxes and confidence scores displayed on-screen
- Frontend built with HTML and CSS
- Django framework to integrate backend and frontend

---

## Tech Stack

- Backend:- Python, Django, PyTorch
- Frontend:- HTML, CSS
- Libraries: OpenCV, Ultralytics YOLO

---

## Installation

1. Clone the repository or download the code.
2. Navigate to the project directory.
3. Install the required packages:

## Library installation
pip install torch opencv-python-headless ultralytics

## Set up the Django application
- django-admin startproject object_detection_project // in command line
- python manage.py runserver
