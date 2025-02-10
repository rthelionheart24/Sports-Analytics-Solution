from ultralytics import YOLO

model = YOLO("models/ball_detection/weights/best.pt")
# model = YOLO("yolo11x.pt")

result = model.predict(source="sinner_trim.mp4", conf=0.15, device="cuda", save=True, exist_ok=True)