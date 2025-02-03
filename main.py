from utils import read_video, save_video

from trackers import PlayerTracker

def __main__():
    input_video_path = "sinner.mp4"
    video_frames = read_video(input_video_path)
    
    # @@TODO: Add your code here
    player_tracker = PlayerTracker(model_path="yolo11x.pt")
    player_detecions = player_tracker.detect_frames(video_frames)
    player_detection_output_frames = player_tracker.draw_bboxes(video_frames, player_detecions)
    
    
    output_video_path = "./output.avi"
    save_video(player_detection_output_frames, output_video_path)



if __name__ == '__main__':
    __main__()