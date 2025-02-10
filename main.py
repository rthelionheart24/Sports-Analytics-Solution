from utils import read_video, save_video

from trackers import PlayerTracker, BallTracker
from court_detector import CourtDetector
import cv2

def __main__():
    input_video_path = "./sinner_trim.mp4"
    video_frames = read_video(input_video_path)
    
    #TODO: Add your code here
    player_tracker = PlayerTracker(model_path="yolo11x.pt")
    ball_tracker = BallTracker(model_path="models/ball_detection/weights/best.pt")
    court_detector = CourtDetector("models/tennis_court_keypoints.pth")
    
    player_detecions = player_tracker.detect_frames(video_frames, read_from_stub=True, 
                                                    stub_path="tracker_stubs/player_detections.pkl")
    
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, 
                                                stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    court_detections = court_detector.predict(video_frames[0])
    
    player_detecions = player_tracker.choose_and_filter_players(court_detections, player_detecions)
    
    
    output_frames = player_tracker.draw_bboxes(video_frames, player_detecions)
    output_frames = ball_tracker.draw_bboxes(video_frames, ball_detections)
    output_frames = court_detector.draw_keypoints(output_frames, court_detections)
    

    
    for i, frame in enumerate(output_frames):
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
    
    
    output_video_path = "./output.avi"
    save_video(output_frames, output_video_path)



if __name__ == '__main__':
    __main__()