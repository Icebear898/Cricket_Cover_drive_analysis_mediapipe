
"""
AthleteRise Cricket Analytics - Pose Estimation Module
Handles MediaPipe pose detection and keypoint extraction
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Tuple, Optional

class PoseEstimator:
    """Handles pose estimation using MediaPipe"""
    
    def __init__(self, model_complexity: int = 1, min_detection_confidence: float = 0.5, 
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe pose estimator
        
        Args:
            model_complexity: Model complexity (0, 1, or 2) - lower is faster
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Define keypoint mapping
        self.landmark_map = {
            'nose': self.mp_pose.PoseLandmark.NOSE,
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'left_ankle': self.mp_pose.PoseLandmark.LEFT_ANKLE,
            'right_ankle': self.mp_pose.PoseLandmark.RIGHT_ANKLE
        }
    
    def process_frame(self, frame: np.ndarray) -> Optional[object]:
        """
        Process a single frame for pose detection
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            MediaPipe pose results or None if detection fails
        """
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process pose
            results = self.pose.process(rgb_frame)
            
            return results
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None
    
    def extract_keypoints(self, landmarks) -> Dict[str, Tuple[float, float]]:
        """
        Extract relevant keypoints from MediaPipe landmarks
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Dictionary of keypoint names and normalized coordinates
        """
        if not landmarks:
            return {}
        
        keypoints = {}
        
        for name, landmark_id in self.landmark_map.items():
            try:
                landmark = landmarks.landmark[landmark_id]
                # Only use visible landmarks
                if landmark.visibility > 0.5:
                    keypoints[name] = (landmark.x, landmark.y)
            except (IndexError, AttributeError):
                # Skip if landmark is not available
                continue
        
        return keypoints
    
    def draw_pose_skeleton(self, frame: np.ndarray, keypoints: Dict[str, Tuple[float, float]], 
                          frame_width: int, frame_height: int) -> np.ndarray:
        """
        Draw pose skeleton on frame
        
        Args:
            frame: Input frame
            keypoints: Dictionary of keypoint coordinates (normalized)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Frame with pose skeleton drawn
        """
        # Convert normalized coordinates to pixel coordinates
        pixel_keypoints = {}
        for name, (x, y) in keypoints.items():
            pixel_keypoints[name] = (int(x * frame_width), int(y * frame_height))
        
        # Define skeleton connections
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            ('nose', 'left_shoulder'),
            ('nose', 'right_shoulder')
        ]
        
        # Draw connections
        for start, end in connections:
            if start in pixel_keypoints and end in pixel_keypoints:
                cv2.line(frame, pixel_keypoints[start], pixel_keypoints[end], (0, 255, 0), 2)
        
        # Draw keypoints
        for name, (x, y) in pixel_keypoints.items():
            # Different colors for different body parts
            if 'shoulder' in name or 'elbow' in name or 'wrist' in name:
                color = (0, 0, 255)  # Red for arms
            elif 'hip' in name or 'knee' in name or 'ankle' in name:
                color = (255, 0, 0)  # Blue for legs
            else:
                color = (255, 255, 0)  # Yellow for head
            
            cv2.circle(frame, (x, y), 5, color, -1)
            # Add label
            cv2.putText(frame, name.replace('_', ' '), (x + 5, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        return frame
    
    def get_pose_confidence(self, landmarks) -> float:
        """
        Calculate overall pose confidence based on visible landmarks
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Average confidence score (0-1)
        """
        if not landmarks:
            return 0.0
        
        confidences = []
        for landmark_id in self.landmark_map.values():
            try:
                landmark = landmarks.landmark[landmark_id]
                confidences.append(landmark.visibility)
            except (IndexError, AttributeError):
                continue
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def cleanup(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'pose'):
            self.pose.close()
