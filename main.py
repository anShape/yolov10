from flask import Flask, render_template, jsonify, Response, request
import time
import io
import numpy as np
import cv2
import psutil
from ultralytics import YOLO

#model = YOLOv10.from_pretrained('jameslahm/yolov10n')
model = YOLO('yolov8n.pt')

app = Flask(__name__)

def generate_frames():
    """Generate frames for the video feed."""
    while True:
        
        # results = model(source="0", classes=[2])
        results = model(source="0")
        annotated_frame = results[0].plot()
        
        if annotated_frame is not None:
            annotated_frame = cv2.resize(annotated_frame, (640, 480))
            _, jpeg = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Provide the video feed."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)