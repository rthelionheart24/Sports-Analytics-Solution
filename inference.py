from ultralytics import YOLO

# model = YOLO("models/ball_detection/weights/best.pt")
model = YOLO("yolo11x.pt")

result = model.track(source="sinner.avi", conf=0.25, device="cuda", save=True, project="yolov11x_results", name="track", exist_ok=True)