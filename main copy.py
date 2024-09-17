from flask import Flask, render_template, jsonify, Response, request
import time
import io
import numpy as np
import cv2
import psutil
from ultralytics import YOLO
import sys

#model = YOLOv10.from_pretrained('jameslahm/yolov10n')
model = YOLO('yolov8n.pt')


app = Flask(__name__)

def plot_bboxes(results):
    img = results[0].orig_img # original image
    names = results[0].names # class names dict
    scores = results[0].boxes.conf.numpy() # probabilities
    classes = results[0].boxes.cls.numpy() # predicted classes
    boxes = results[0].boxes.xyxy.numpy().astype(np.int32) # bboxes
    for score, cls, bbox in zip(scores, classes, boxes): # loop over all bboxes
        class_label = names[cls] # class name
        label = f"{class_label} : {score:0.2f}" # bbox label
        lbl_margin = 3 #label margin
        img = cv2.rectangle(img, (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            color=(0, 0, 255),
                            thickness=1)
        label_size = cv2.getTextSize(label, # labelsize in pixels 
                                     fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                     fontScale=1, thickness=1)
        lbl_w, lbl_h = label_size[0] # label w and h
        lbl_w += 2* lbl_margin # add margins on both sides
        lbl_h += 2*lbl_margin
        img = cv2.rectangle(img, (bbox[0], bbox[1]), # plot label background
                             (bbox[0]+lbl_w, bbox[1]-lbl_h),
                             color=(0, 0, 255), 
                             thickness=-1) # thickness=-1 means filled rectangle
        cv2.putText(img, label, (bbox[0]+ lbl_margin, bbox[1]-lbl_margin), # write label to the image
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=(255, 255, 255 ),
                    thickness=1)
    return img

def generate_frames():
    """Generate frames for the video feed."""
    while True:
        
        # results = model(source="0", classes=[2])
        # source = "rtsp://raspi.local:8554/cam"

        # results = model.predict(source)
        results = model.track(source="0", stream=True)
        # annotated_frame = results[0].plot()
        # for result in results:
        #     boxes = result.boxes  # Boxes object for bounding box outputs
        
        
        # annotated_frame = plot_bboxes(results)
        for result in results:
            annotated_frame = result.plot()
        
        
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