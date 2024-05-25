import cv2
import mediapipe as mp
import numpy as np



class PoseEstimation:
    def __init__(self,video_source=0):
        self.cap = cv2.VideoCapture(video_source)
        self.mp_pose = mp.solutions.pose  # Make mp_pose an instance attribute
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame
    
    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    
    def release(self):
        self.cap.release()
        self.pose.close()
        
    def draw_landmarks(self, image, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        return image