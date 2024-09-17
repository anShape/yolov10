# from ultralytics import YOLOv10
from ultralytics import YOLO
# model = YOLOv10.from_pretrained('jameslahm/yolov10n')
# ncnn_model = YOLOv10('yolov10n_ncnn_model')
# model = YOLO('yolov8n_ncnn_model')
model = YOLO('yolov8n.pt')

source = "rtsp://raspi.local:8554/cam"

results = model.predict(source, show=True, stream=True)
# results = model.predict(source="0", show=True, stream=True)
# results = model(source="0", show=True, save_txt=True)
# results = model.predict(source="0", stream=True)

# for result in results:
#      result.save_txt("output.txt")

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen