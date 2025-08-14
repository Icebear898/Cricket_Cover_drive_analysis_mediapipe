
"""
AthleteRise Cricket Analytics - Metrics Calculator Module
Handles biomechanical metrics computation from pose keypoints
"""

import numpy as np
import math
from typing import Dict, Tuple, Optional

class MetricsCalculator:
    """Calculates biomechanical metrics from pose keypoints"""
    
    def __init__(self):
        """Initialize metrics calculator with default thresholds"""
        self.thresholds = {
            'elbow_angle_good': (90, 130),
            'spine_lean_good': (10, 25),
            'head_alignment_good': 50,  # pixels
            'foot_angle_good': (-15, 15)  # degrees from perpendicular
        }
    
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], 
                       p3: Tuple[float, float]) -> float:
        """
        Calculate angle between three points (p2 is the vertex)
        
        Args:
            p1: First point coordinates
            p2: Vertex point coordinates  
            p3: Third point coordinates
            
        Returns:
            Angle in degrees
        """
        try:
            # Convert to numpy arrays
            a = np.array(p1)
            b = np.array(p2)
            c = np.array(p3)
            
            # Calculate vectors
            ba = a - b
            bc = c - b
            
            # Calculate angle
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
        except Exception as e:
            print(f"Error calculating angle: {e}")
            return 0.0
    
    def calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        Calculate Euclidean distance between two points
        
        Args:
            p1: First point coordinates
            p2: Second point coordinates
            
        Returns:
            Distance between points
        """
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def compute_front_elbow_angle(self, keypoints: Dict[str, Tuple[float, float]], 
                                 frame_width: int, frame_height: int) -> Optional[float]:
        """
        Compute front elbow angle (shoulder-elbow-wrist)
        Assumes right-handed batsman
        
        Args:
            keypoints: Dictionary of normalized keypoint coordinates
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Elbow angle in degrees or None if calculation fails
        """
        required_points = ['right_shoulder', 'right_elbow', 'right_wrist']
        
        if not all(point in keypoints for point in required_points):
            return None
        
        # Convert to pixel coordinates
        shoulder = (keypoints['right_shoulder'][0] * frame_width, 
                   keypoints['right_shoulder'][1] * frame_height)
        elbow = (keypoints['right_elbow'][0] * frame_width,
                keypoints['right_elbow'][1] * frame_height)
        wrist = (keypoints['right_wrist'][0] * frame_width,
                keypoints['right_wrist'][1] * frame_height)
        
        return self.calculate_angle(shoulder, elbow, wrist)
    
    def compute_spine_lean(self, keypoints: Dict[str, Tuple[float, float]], 
                          frame_width: int, frame_height: int) -> Optional[float]:
        """
        Compute spine lean angle (hip-shoulder line vs vertical)
        
        Args:
            keypoints: Dictionary of normalized keypoint coordinates
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Spine lean angle in degrees or None if calculation fails
        """
        required_points = ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']
        
        if not all(point in keypoints for point in required_points):
            return None
        
        try:
            # Convert to pixel coordinates and calculate centers
            hip_center = (
                (keypoints['left_hip'][0] + keypoints['right_hip'][0]) / 2 * frame_width,
                (keypoints['left_hip'][1] + keypoints['right_hip'][1]) / 2 * frame_height
            )
            shoulder_center = (
                (keypoints['left_shoulder'][0] + keypoints['right_shoulder'][0]) / 2 * frame_width,
                (keypoints['left_shoulder'][1] + keypoints['right_shoulder'][1]) / 2 * frame_height
            )
            
            # Calculate spine vector and vertical vector
            spine_vector = (shoulder_center[0] - hip_center[0], 
                           shoulder_center[1] - hip_center[1])
            vertical_vector = (0, -1)  # Pointing up
            
            # Calculate angle with vertical
            dot_product = spine_vector[0] * vertical_vector[0] + spine_vector[1] * vertical_vector[1]
            spine_magnitude = math.sqrt(spine_vector[0]**2 + spine_vector[1]**2)
            
            if spine_magnitude > 0:
                cos_angle = dot_product / spine_magnitude
                cos_angle = max(-1, min(1, cos_angle))
                spine_lean = math.degrees(math.acos(cos_angle))
                return spine_lean
            
        except Exception as e:
            print(f"Error computing spine lean: {e}")
        
        return None
    
    def compute_head_knee_alignment(self, keypoints: Dict[str, Tuple[float, float]], 
                                   frame_width: int, frame_height: int) -> Optional[float]:
        """
        Compute head-over-knee vertical alignment
        
        Args:
            keypoints: Dictionary of normalized keypoint coordinates
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Horizontal distance between head and knee in pixels or None if calculation fails
        """
        required_points = ['nose', 'right_knee']
        
        if not all(point in keypoints for point in required_points):
            return None
        
        # Convert to pixel coordinates
        head_x = keypoints['nose'][0] * frame_width
        knee_x = keypoints['right_knee'][0] * frame_width
        
        return abs(head_x - knee_x)
    
    def compute_front_foot_direction(self, keypoints: Dict[str, Tuple[float, float]], 
                                    frame_width: int, frame_height: int) -> Optional[float]:
        """
        Compute front foot direction angle
        
        Args:
            keypoints: Dictionary of normalized keypoint coordinates
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Foot angle in degrees from perpendicular to crease or None if calculation fails
        """
        required_points = ['right_knee', 'right_ankle']
        
        if not all(point in keypoints for point in required_points):
            return None
        
        try:
            # Convert to pixel coordinates
            knee = (keypoints['right_knee'][0] * frame_width,
                   keypoints['right_knee'][1] * frame_height)
            ankle = (keypoints['right_ankle'][0] * frame_width,
                    keypoints['right_ankle'][1] * frame_height)
            
            # Calculate foot vector
            foot_vector = (ankle[0] - knee[0], ankle[1] - knee[1])
            
            # Calculate angle with horizontal (crease direction)
            horizontal_vector = (1, 0)
            
            dot_product = foot_vector[0] * horizontal_vector[0] + foot_vector[1] * horizontal_vector[1]
            foot_magnitude = math.sqrt(foot_vector[0]**2 + foot_vector[1]**2)
            
            if foot_magnitude > 0:
                cos_angle = dot_product / foot_magnitude
                cos_angle = max(-1, min(1, cos_angle))
                foot_angle = math.degrees(math.acos(cos_angle))
                # Adjust to get angle from perpendicular to crease
                foot_angle = 90 - foot_angle
                return foot_angle
                
        except Exception as e:
            print(f"Error computing foot direction: {e}")
        
        return None
    
    def compute_all_metrics(self, keypoints: Dict[str, Tuple[float, float]], 
                           frame_width: int, frame_height: int) -> Dict[str, float]:
        """
        Compute all biomechanical metrics from keypoints
        
        Args:
            keypoints: Dictionary of normalized keypoint coordinates
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Compute each metric
        elbow_angle = self.compute_front_elbow_angle(keypoints, frame_width, frame_height)
        if elbow_angle is not None:
            metrics['front_elbow_angle'] = elbow_angle
        
        spine_lean = self.compute_spine_lean(keypoints, frame_width, frame_height)
        if spine_lean is not None:
            metrics['spine_lean'] = spine_lean
        
        head_alignment = self.compute_head_knee_alignment(keypoints, frame_width, frame_height)
        if head_alignment is not None:
            metrics['head_knee_alignment'] = head_alignment
        
        foot_direction = self.compute_front_foot_direction(keypoints, frame_width, frame_height)
        if foot_direction is not None:
            metrics['front_foot_direction'] = foot_direction
        
        return metrics
    
    def get_metric_feedback(self, metric_name: str, value: float) -> Tuple[str, bool]:
        """
        Get feedback message and status for a metric
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            
        Returns:
            Tuple of (feedback_message, is_good)
        """
        feedback_map = {
            'front_elbow_angle': {
                'good_range': self.thresholds['elbow_angle_good'],
                'good_msg': "✅ Good elbow",
                'bad_msg': "❌ Check elbow"
            },
            'spine_lean': {
                'good_range': self.thresholds['spine_lean_good'],
                'good_msg': "✅ Good posture",
                'bad_msg': "❌ Check posture"
            },
            'head_knee_alignment': {
                'good_threshold': self.thresholds['head_alignment_good'],
                'good_msg': "✅ Head aligned",
                'bad_msg': "❌ Head not over knee"
            },
            'front_foot_direction': {
                'good_range': self.thresholds['foot_angle_good'],
                'good_msg': "✅ Good footwork",
                'bad_msg': "❌ Check foot angle"
            }
        }
        
        if metric_name not in feedback_map:
            return "Unknown metric", False
        
        config = feedback_map[metric_name]
        
        # Check if metric is in good range
        if 'good_range' in config:
            min_val, max_val = config['good_range']
            is_good = min_val <= value <= max_val
        else:  # threshold-based
            is_good = value <= config['good_threshold']
        
        return config['good_msg'] if is_good else config['bad_msg'], is_good
    
    def update_thresholds(self, new_thresholds: Dict[str, any]):
        """
        Update metric thresholds
        
        Args:
            new_thresholds: Dictionary of new threshold values
        """
        self.thresholds.update(new_thresholds)
