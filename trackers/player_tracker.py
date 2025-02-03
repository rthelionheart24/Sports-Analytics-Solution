from ultralytics import YOLO
import cv2

class PlayerTracker:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        
        
    def detect_frames(self, frames):
        player_detection = []
        
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detection.append(player_dict)
    
    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)
        id_name_dict = results.names
        
        player_dict = {}
        
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict
    
    def draw_bboxes(self, frames, player_dict):
        output_frames = []
        
        for frame, player_dict in zip(frames, player_dict):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player {track_id}", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                output_frames.append(frame)
                
                
        return output_frames
                