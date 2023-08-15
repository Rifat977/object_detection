import os
import cv2
import json
import numpy as np 
from django.conf import settings
from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from channels.generic.websocket import AsyncWebsocketConsumer
from django.http import JsonResponse, HttpResponseServerError
from .models import DetectedObject


def update_object(object_name):
    first_detected_object = object_name
    try:
        detected_object = DetectedObject.objects.first()
    except DetectedObject.DoesNotExist:
        detected_object = None

    if detected_object:
        detected_object.name = first_detected_object
        detected_object.save()
    else:
        DetectedObject.objects.create(name=first_detected_object)

def update_to_null_value():
    try:
        detected_object = DetectedObject.objects.first()
    except DetectedObject.DoesNotExist:
        detected_object = None

    if detected_object:
        detected_object.name = ""
        detected_object.save()
    else:
        DetectedObject.objects.create(name=first_detected_object)

def process_frame(frame, net, classes, layer_names):
    detected_objects = ""
    height, width = frame.shape[:2]

    input_size = (320, 320)
    mean_values = (0, 0, 0)
    scale = 0.00784
    blob = cv2.dnn.blobFromImage(frame, scale, input_size, mean_values, swapRB=True, crop=False)

    net.setInput(blob)
    detections = net.forward(layer_names)
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                object_name = classes[class_id]
                detected_objects = object_name

                update_object(object_name)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, object_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    _, buffer = cv2.imencode('.jpg', frame)
    frame_data = buffer.tobytes()
    return (
        b'--frame\r\n' b'Content-Type: image/jpeg\r\n' b'Object-Name:' + detected_objects.encode() + b'\r\n\r\n' + frame_data + b'\r\n'
    )

def get_frames():
    try:
        yolov3_weights_path = os.path.join(settings.BASE_DIR, "darknet", "yolov3.weights")
        yolov3_cfg_path = os.path.join(settings.BASE_DIR, "darknet", "cfg", "yolov3.cfg")
        net = cv2.dnn.readNet(yolov3_weights_path, yolov3_cfg_path)
        classes = ['None']
        with open(os.path.join(settings.BASE_DIR, "darknet", "data", "coco.names"), "r") as f:
            classes = f.read().strip().split("\n")

        cap = cv2.VideoCapture(0)
        layer_names = net.getUnconnectedOutLayersNames()
        
        # Capture and process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Unable to capture frame.")
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))

            # Process the frame using a separate thread
            result = process_frame(frame, net, classes, layer_names)

            # Yield the processed frame
            yield result

    except Exception as e:
        print("An error occurred:", str(e))
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + b'\r\n')

@gzip.gzip_page
def webcam_stream(request):
    return StreamingHttpResponse(get_frames(), content_type="multipart/x-mixed-replace;boundary=frame")

def index(request):
    update_to_null_value()
    return render(request, 'index.html')

def detected_object(request):
    detected_obj = DetectedObject.objects.first().name
    print(detected_obj)
    return JsonResponse({'data':detected_obj})