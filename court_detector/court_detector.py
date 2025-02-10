import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2

class CourtDetector:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda")))
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def predict(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(img_rgb).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            
        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = img_rgb.shape[:2]

        keypoints[::2] *= original_w/224.0
        keypoints[1::2] *= original_h/224.0
        
        return keypoints
    
    def draw_keypoint(self, image, keypoints):
        
        for i in range(0, len(keypoints), 2):
            x, y = int(keypoints[i]), int(keypoints[i+1])
            image = cv2.putText(image, str(i//2), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
            image = cv2.circle(image, (x, y), 5, (255, 255, 0), -1)
            
        return image
    
    def draw_keypoints(self, video_frames, keypoints):
        output_frames = []
        
        for frame in video_frames:
            frame = self.draw_keypoint(frame, keypoints)
            output_frames.append(frame)
        return output_frames
        