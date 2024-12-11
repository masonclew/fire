from ultralytics import YOLO


model = YOLO('best.pt')
model.predict(source='fire.jpg', imgsz=640, conf=0.6, save=True)
