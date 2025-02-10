from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(0,[]) for x in ball_positions]
        
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{0: x} for x in df_ball_positions.to_numpy().tolist()]
        
        return ball_positions
        
        
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []
        
        if read_from_stub:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
                return ball_detections
        
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
            
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections
    
    def detect_frame(self, frame):
        results = self.model.track(frame, conf=0.15, device="cuda")[0]
        id_name_dict = results.names
        
        ball_dict = {}
        
        if results.boxes.id is None:
            return ball_dict
        
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[0] = result
        
        return ball_dict
    
    def draw_bboxes(self, frames, ball_dict):
        output_frames = []
        
        for frame, ball_dict in zip(frames, ball_dict):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball {track_id}", (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_frames.append(frame)
                
                
        return output_frames
                