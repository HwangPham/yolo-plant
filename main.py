from ultralytics import YOLO
model = YOLO("pre-trained/yolov8n-cls.pt")
results = model.train(data="dataset/plantdoc", epochs=100, imgsz=640)