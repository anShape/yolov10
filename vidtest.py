# from ultralytics import YOLOv10
import numpy as np
from ultralytics import YOLO
# model = YOLOv10.from_pretrained('jameslahm/yolov10n')
# ncnn_model = YOLOv10('yolov10n_ncnn_model')
# model = YOLO('yolov8n_ncnn_model')
model = YOLO('yolov8n.pt')

# source = "rtsp://raspi.local:8554/cam"
# source = "camera.streams"

# results = model.predict(source, show=True, stream=True)
results = model.predict(source="0", show=True, stream=True)
# results = model(source="0", show=True, save_txt=True)
# results = model.predict(source="0", stream=True)

# for result in results:
#      result.save_txt("output.txt")

print(results[0].orig_img)

for result in results:
    boxes = result.boxes # Boxes object for bounding box outputs
    print(result.orig_img) # original image
    