import math
import cv2
import torch

import numpy as np
import matplotlib.pyplot as plt

from kp import MMPose
from yolo import YOLOv8


class SmokingDetection():
    def __init__(self, pd_model_path: str, pose_model_path: str):
        # Initialize models
        self.pose = MMPose(pose_model_path)
        self.yolo = YOLOv8(pd_model_path)
    
    def detect(self, image: np.array) -> dict:
        # Get bounding boxes with people
        boxes, scores, class_ids = self.yolo(image)
        if len(boxes) == 0:
            return []
        people = []
        for box in boxes:
            x1, y1, x2, y2 = list(map(int, box))
            human = np.copy(image[y1:y2, x1:x2, :])
            # Get human keypoints
            keypoints = self.pose.predict(cv2.cvtColor(human, cv2.COLOR_BGR2RGB))
            # Get angles between the shoulders and forearms
            angles = self.get_angles(keypoints[6:11:2], keypoints[5:10:2])
            # Get the rations between total and current distances
            ratios = self.get_ratios(keypoints[12], keypoints[11], keypoints[10], keypoints[9], keypoints[0])
            people.append([ratios[0] <= 0.3 and angles[0] <= 40 or ratios[1] <= 0.3 and angles[1] <= 40, box])
        return people
            
    @staticmethod
    def get_angles(rhand: list, lhand: list) -> tuple:
        # Right hand vector
        rx1 = rhand[0][0] - rhand[1][0]
        rx2 = rhand[2][0] - rhand[1][0]
        ry1 = rhand[0][1] - rhand[1][1]
        ry2 = rhand[2][1] - rhand[1][1]
        # Left hand vector
        lx1 = lhand[0][0] - lhand[1][0]
        lx2 = lhand[2][0] - lhand[1][0]
        ly1 = lhand[0][1] - lhand[1][1]
        ly2 = lhand[2][1] - lhand[1][1]
        # Angle of the right hand
        rcos = (rx1*rx2 + ry1*ry2) / (math.sqrt(rx1**2 + ry1**2) * math.sqrt(rx2**2 + ry2**2))
        r_angle = math.acos(rcos) * 180 / math.pi
        # Angle of the left hand
        lcos = (lx1*lx2 + ly1*ly2) / (math.sqrt(lx1**2 + ly1**2) * math.sqrt(lx2**2 + ly2**2))
        l_angle = math.acos(lcos) * 180 / math.pi
        
        return r_angle, l_angle
    
    @staticmethod
    def get_ratios(rhip: list, lhip: list, rwrist: list, lwrist: list, nose: list) -> tuple:
        # Keypoints coords
        rhx, rhy = rhip
        lhx, lhy = lhip
        rwx, rwy = rwrist
        lwx, lwy = lwrist
        nx, ny = nose
        # Get distances
        lhip2nose = math.sqrt((lhx - nx)**2 + (lhy - ny)**2)
        rhip2nose = math.sqrt((rhx - nx)**2 + (rhy - ny)**2)
        lwrist2nose = math.sqrt((lwx - nx)**2 + (lwy - ny)**2)
        rwrist2nose = math.sqrt((rwx - nx)**2 + (rwy - ny)**2)
        
        return rwrist2nose/rhip2nose, lwrist2nose/lhip2nose
    

        
    
        
        
        
        
            
            
            
    