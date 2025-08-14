
"""
AthleteRise Cricket Analytics - Utilities Module
Common utility functions and helpers
"""

import os
import time
import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class AnalysisUtils:
    """Utility functions for cricket analysis"""
    
    @staticmethod
    def ensure_output_directory(output_dir: str = "output") -> str:
        """
        Ensure output directory exists
        
        Args:
            output_dir: Output directory path
            
        Returns:
            Absolute path to output directory
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return os.path.abspath(output_dir)
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format duration in seconds to human readable format
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    @staticmethod
    def calculate_frame_rate(total_frames: int, duration: float) -> float:
        """
        Calculate frame rate from total frames and duration
        
        Args:
            total_frames: Total number of frames
            duration: Duration in seconds
            
        Returns:
            Frame rate in FPS
        """
        return total_frames / duration if duration > 0 else 0
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """
        Normalize angle to 0-360 degree range
        
        Args:
            angle: Input angle in degrees
            
        Returns:
            Normalized angle
        """
        while angle < 0:
            angle += 360
        while angle >= 360:
            angle -= 360
        return angle
    
    @staticmethod
    def calculate_percentage_change(old_value: float, new_value: float) -> float:
        """
        Calculate percentage change between two values
        
        Args:
            old_value: Original value
            new_value: New value
            
        Returns:
            Percentage change
        """
        if old_value == 0:
            return 0 if new_value == 0 else 100
        return ((new_value - old_value) / old_value) * 100
    
    @staticmethod
    def smooth_metrics(metrics_list: List[Dict[str, float]], window_size: int = 3) -> List[Dict[str, float]]:
        """
        Apply moving average smoothing to metrics
        
        Args:
            metrics_list: List of metrics dictionaries
            window_size: Size of smoothing window
            
        Returns:
            Smoothed metrics list
        """
        if len(metrics_list) < window_size:
            return metrics_list
        
        smoothed = []
        metric_names = set()
        for m in metrics_list:
            metric_names.update(m.keys())
        
        for i in range(len(metrics_list)):
            smoothed_frame = {}
            
            for metric_name in metric_names:
                values = []
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(metrics_list), i + window_size // 2 + 1)
                
                for j in range(start_idx, end_idx):
                    if metric_name in metrics_list[j]:
                        values.append(metrics_list[j][metric_name])
                
                if values:
                    smoothed_frame[metric_name] = sum(values) / len(values)
            
            smoothed.append(smoothed_frame)
        
        return smoothed
    
    @staticmethod
    def detect_outliers(values: List[float], threshold: float = 2.0) -> List[int]:
        """
        Detect outliers using z-score method
        
        Args:
            values: List of numeric values
            threshold: Z-score threshold for outlier detection
            
        Returns:
            List of outlier indices
        """
        if len(values) < 3:
            return []
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return []
        
        outliers = []
        for i, value in enumerate(values):
            z_score = abs((value - mean_val) / std_val)
            if z_score > threshold:
                outliers.append(i)
        
        return outliers
    
    @staticmethod
    def calculate_trend(values: List[float]) -> str:
        """
        Calculate trend direction from a series of values
        
        Args:
            values: List of numeric values
            
        Returns:
            Trend description ('improving', 'declining', 'stable')
        """
        if len(values) < 2:
            return 'stable'
        
        # Calculate linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.1:
            return 'improving'
        elif slope < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    @staticmethod
    def generate_color_from_score(score: float, max_score: float = 10.0) -> Tuple[int, int, int]:
        """
        Generate RGB color based on score (red to green gradient)
        
        Args:
            score: Score value
            max_score: Maximum possible score
            
        Returns:
            RGB color tuple
        """
        # Normalize score to 0-1 range
        normalized = max(0, min(1, score / max_score))
        
        # Red to green gradient
        red = int(255 * (1 - normalized))
        green = int(255 * normalized)
        blue = 0
        
        return (red, green, blue)
    
    @staticmethod
    def format_metric_name(metric_name: str) -> str:
        """
        Format metric name for display
        
        Args:
            metric_name: Raw metric name
            
        Returns:
            Formatted metric name
        """
        return metric_name.replace('_', ' ').title()
    
    @staticmethod
    def validate_keypoints(keypoints: Dict[str, Tuple[float, float]]) -> bool:
        """
        Validate keypoints dictionary
        
        Args:
            keypoints: Dictionary of keypoint coordinates
            
        Returns:
            True if valid, False otherwise
        """
        if not keypoints:
            return False
        
        for name, (x, y) in keypoints.items():
            if not (0 <= x <= 1 and 0 <= y <= 1):
                return False
        
        return True
    
    @staticmethod
    def calculate_body_orientation(keypoints: Dict[str, Tuple[float, float]]) -> Optional[float]:
        """
        Calculate body orientation angle from keypoints
        
        Args:
            keypoints: Dictionary of keypoint coordinates
            
        Returns:
            Body orientation angle in degrees or None if calculation fails
        """
        required_points = ['left_shoulder', 'right_shoulder']
        
        if not all(point in keypoints for point in required_points):
            return None
        
        left_shoulder = keypoints['left_shoulder']
        right_shoulder = keypoints['right_shoulder']
        
        # Calculate angle of shoulder line
        dx = right_shoulder[0] - left_shoulder[0]
        dy = right_shoulder[1] - left_shoulder[1]
        
        angle = math.degrees(math.atan2(dy, dx))
        return AnalysisUtils.normalize_angle(angle)
    
    @staticmethod
    def estimate_shot_phase(frame_metrics: List[Dict[str, float]], frame_index: int) -> str:
        """
        Estimate cricket shot phase based on metrics
        
        Args:
            frame_metrics: List of all frame metrics
            frame_index: Current frame index
            
        Returns:
            Estimated shot phase
        """
        if frame_index >= len(frame_metrics):
            return 'unknown'
        
        current_metrics = frame_metrics[frame_index]
        
        # Simple heuristic based on elbow angle
        if 'front_elbow_angle' in current_metrics:
            elbow_angle = current_metrics['front_elbow_angle']
            
            if elbow_angle > 140:
                return 'backswing'
            elif elbow_angle > 100:
                return 'downswing'
            elif elbow_angle > 80:
                return 'impact'
            else:
                return 'follow_through'
        
        return 'stance'
    
    @staticmethod
    def create_performance_summary(evaluation: Dict[str, Any]) -> str:
        """
        Create a concise performance summary
        
        Args:
            evaluation: Evaluation results dictionary
            
        Returns:
            Performance summary string
        """
        overall_score = evaluation.get('overall_score', 0)
        skill_level = evaluation.get('skill_level', 'Unknown')
        
        summary = f"Overall Performance: {overall_score}/10 ({skill_level})\n"
        
        if 'category_scores' in evaluation:
            best_category = max(evaluation['category_scores'].items(), key=lambda x: x[1])
            worst_category = min(evaluation['category_scores'].items(), key=lambda x: x[1])
            
            summary += f"Strongest Area: {AnalysisUtils.format_metric_name(best_category[0])} ({best_category[1]}/10)\n"
            summary += f"Area for Improvement: {AnalysisUtils.format_metric_name(worst_category[0])} ({worst_category[1]}/10)\n"
        
        return summary
    
    @staticmethod
    def log_analysis_info(message: str, level: str = "INFO"):
        """
        Log analysis information with timestamp
        
        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR)
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    @staticmethod
    def get_system_info() -> Dict[str, str]:
        """
        Get system information for debugging
        
        Returns:
            Dictionary of system information
        """
        import platform
        import sys
        
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': sys.version,
            'architecture': platform.architecture()[0]
        }
