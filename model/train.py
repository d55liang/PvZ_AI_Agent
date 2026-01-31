from ultralytics import YOLO

model = YOLO("yolov8.yaml")

results = model.train(data="data.yaml", epochs=50)