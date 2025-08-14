
"""
AthleteRise Cricket Analytics - Video Processor Module
Handles video I/O, processing, and overlay generation
"""

import cv2
import os
import time
import yt_dlp
import numpy as np
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm

class VideoProcessor:
    """Handles video processing, downloading, and overlay generation"""
    
    def __init__(self):
        """Initialize video processor"""
        self.fps_counter = []
        
    def download_video(self, url: str, output_path: str = "input_video.mp4") -> Optional[str]:
        """
        Download video from YouTube using yt-dlp
        
        Args:
            url: YouTube video URL
            output_path: Output file path
            
        Returns:
            Path to downloaded video or None if failed
        """
        print(f"Downloading video from: {url}")
        
        ydl_opts = {
            'format': 'best[height<=720]',  # Limit resolution for speed
            'outtmpl': output_path,
            'quiet': True,
            'no_warnings': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"Video downloaded successfully: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None
    
    def get_video_properties(self, video_path: str) -> Dict[str, any]:
        """
        Get video properties
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary containing video properties
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}
        
        properties = {
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return properties
    
    def draw_metrics_overlay(self, frame: np.ndarray, metrics: Dict[str, float], 
                           feedback: Dict[str, Tuple[str, bool]], frame_count: int) -> np.ndarray:
        """
        Draw metrics overlay on frame
        
        Args:
            frame: Input frame
            metrics: Dictionary of metric values
            feedback: Dictionary of feedback messages and status
            frame_count: Current frame number
            
        Returns:
            Frame with metrics overlay
        """
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, y_offset), 
                   font, font_scale, (255, 255, 255), thickness)
        y_offset += 30
        
        # Draw metrics with feedback
        for metric_name, value in metrics.items():
            if metric_name in feedback:
                feedback_msg, is_good = feedback[metric_name]
                color = (0, 255, 0) if is_good else (0, 0, 255)
                
                # Format metric display
                if metric_name == 'front_elbow_angle':
                    display_text = f"Elbow: {value:.1f}° {feedback_msg}"
                elif metric_name == 'spine_lean':
                    display_text = f"Spine: {value:.1f}° {feedback_msg}"
                elif metric_name == 'head_knee_alignment':
                    display_text = f"Head-Knee: {value:.1f}px {feedback_msg}"
                elif metric_name == 'front_foot_direction':
                    display_text = f"Foot: {value:.1f}° {feedback_msg}"
                else:
                    display_text = f"{metric_name}: {value:.1f} {feedback_msg}"
                
                cv2.putText(frame, display_text, (10, y_offset), 
                           font, font_scale, color, thickness)
                y_offset += 25
        
        return frame
    
    def draw_performance_indicator(self, frame: np.ndarray, overall_score: float) -> np.ndarray:
        """
        Draw overall performance indicator
        
        Args:
            frame: Input frame
            overall_score: Overall performance score (0-10)
            
        Returns:
            Frame with performance indicator
        """
        height, width = frame.shape[:2]
        
        # Draw performance bar
        bar_width = 200
        bar_height = 20
        bar_x = width - bar_width - 20
        bar_y = 20
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Performance bar (colored based on score)
        fill_width = int((overall_score / 10.0) * bar_width)
        if overall_score >= 8:
            color = (0, 255, 0)  # Green
        elif overall_score >= 6:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), 
                     color, -1)
        
        # Score text
        cv2.putText(frame, f"Score: {overall_score:.1f}/10", 
                   (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_video_with_analysis(self, video_path: str, pose_estimator, metrics_calculator, 
                                   output_path: str = "output/annotated_video.mp4") -> Tuple[bool, List[Dict]]:
        """
        Process video with pose estimation and metrics analysis
        
        Args:
            video_path: Path to input video
            pose_estimator: PoseEstimator instance
            metrics_calculator: MetricsCalculator instance
            output_path: Path for output video
            
        Returns:
            Tuple of (success, list_of_frame_metrics)
        """
        print(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False, []
        
        # Get video properties
        properties = self.get_video_properties(video_path)
        fps = properties['fps']
        width = properties['width']
        height = properties['height']
        total_frames = properties['total_frames']
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        frame_metrics = []
        
        # Process frames with progress bar
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                
                # Process pose
                results = pose_estimator.process_frame(frame)
                
                if results and results.pose_landmarks:
                    # Extract keypoints
                    keypoints = pose_estimator.extract_keypoints(results.pose_landmarks)
                    
                    if keypoints:
                        # Compute metrics
                        metrics = metrics_calculator.compute_all_metrics(keypoints, width, height)
                        
                        # Store metrics for final evaluation
                        frame_metrics.append(metrics)
                        
                        # Get feedback for each metric
                        feedback = {}
                        for metric_name, value in metrics.items():
                            feedback[metric_name] = metrics_calculator.get_metric_feedback(metric_name, value)
                        
                        # Draw pose skeleton
                        frame = pose_estimator.draw_pose_skeleton(frame, keypoints, width, height)
                        
                        # Draw metrics overlay
                        frame = self.draw_metrics_overlay(frame, metrics, feedback, frame_count)
                
                # Write frame
                out.write(frame)
                
                # Track FPS
                frame_time = time.time() - frame_start
                if frame_time > 0:
                    current_fps = 1.0 / frame_time
                    self.fps_counter.append(current_fps)
                
                frame_count += 1
                pbar.update(1)
        
        # Cleanup
        cap.release()
        out.release()
        
        # Calculate average FPS
        if self.fps_counter:
            avg_fps = sum(self.fps_counter) / len(self.fps_counter)
            print(f"Average processing FPS: {avg_fps:.2f}")
        
        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Annotated video saved: {output_path}")
        
        return True, frame_metrics
    
    def create_summary_frame(self, template_frame: np.ndarray, evaluation: Dict) -> np.ndarray:
        """
        Create a summary frame with evaluation results
        
        Args:
            template_frame: Template frame for dimensions
            evaluation: Evaluation results dictionary
            
        Returns:
            Summary frame with evaluation overlay
        """
        height, width = template_frame.shape[:2]
        summary_frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(summary_frame, "AthleteRise Cricket Analysis Summary", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Overall score
        overall_score = evaluation.get('overall_score', 0)
        cv2.putText(summary_frame, f"Overall Score: {overall_score:.1f}/10", 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Category scores
        y_offset = 150
        if 'category_scores' in evaluation:
            cv2.putText(summary_frame, "Category Scores:", 
                       (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
            
            for category, score in evaluation['category_scores'].items():
                color = (0, 255, 0) if score >= 7 else (0, 255, 255) if score >= 5 else (0, 0, 255)
                cv2.putText(summary_frame, f"  {category.replace('_', ' ').title()}: {score:.1f}/10", 
                           (70, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                y_offset += 30
        
        return summary_frame
    
    def cleanup_temp_files(self, file_paths: List[str]):
        """
        Clean up temporary files
        
        Args:
            file_paths: List of file paths to remove
        """
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Cleaned up: {file_path}")
            except Exception as e:
                print(f"Error cleaning up {file_path}: {e}")
    
    def get_processing_stats(self) -> Dict[str, float]:
        """
        Get processing performance statistics
        
        Returns:
            Dictionary of processing statistics
        """
        if not self.fps_counter:
            return {}
        
        return {
            'average_fps': sum(self.fps_counter) / len(self.fps_counter),
            'min_fps': min(self.fps_counter),
            'max_fps': max(self.fps_counter),
            'total_frames': len(self.fps_counter)
        }
