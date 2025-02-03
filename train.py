from ultralytics import YOLO, settings

model = YOLO("yolo11x.pt")

model.train(data="/home/rwang2000/datasets/ball/ball.yaml", epochs=5, imgsz=640, device="cuda", batch=0.8, project="models", name="ball_detection", exist_ok=True)