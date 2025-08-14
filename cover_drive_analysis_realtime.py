#!/usr/bin/env python3
"""
AthleteRise – AI-Powered Cricket Analytics
Real-Time Cover Drive Analysis from Full Video

Author: AthleteRise Team
Date: 2025-08-14
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
import math
from typing import Dict, List, Tuple, Optional
import yt_dlp
from tqdm import tqdm

class CricketAnalyzer:
    def __init__(self):
        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Lightweight for speed
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Metrics storage
        self.frame_metrics = []
        self.fps_counter = []
        
        # Evaluation thresholds
        self.thresholds = {
            'elbow_angle_good': (90, 130),
            'spine_lean_good': (10, 25),
            'head_alignment_good': 50,  # pixels
            'foot_angle_good': (-15, 15)  # degrees from perpendicular
        }
        
    def download_video(self, url: str, output_path: str = "input_video.mp4") -> str:
        """Download video from YouTube using yt-dlp"""
        print(f"Downloading video from: {url}")
        
        ydl_opts = {
            'format': 'best[height<=720]',  # Limit resolution for speed
            'outtmpl': output_path,
            'quiet': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            print(f"Video downloaded successfully: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None
    
    def calculate_angle(self, p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        """Calculate angle between three points (p2 is the vertex)"""
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
        except:
            return 0.0
    
    def calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def extract_keypoints(self, landmarks) -> Dict[str, Tuple[float, float]]:
        """Extract relevant keypoints from MediaPipe landmarks"""
        if not landmarks:
            return {}
        
        keypoints = {}
        landmark_map = {
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
        
        for name, landmark_id in landmark_map.items():
            landmark = landmarks.landmark[landmark_id]
            if landmark.visibility > 0.5:  # Only use visible landmarks
                keypoints[name] = (landmark.x, landmark.y)
        
        return keypoints
    
    def compute_biomechanical_metrics(self, keypoints: Dict[str, Tuple[float, float]], 
                                    frame_width: int, frame_height: int) -> Dict[str, float]:
        """Compute biomechanical metrics from keypoints"""
        metrics = {}
        
        try:
            # Convert normalized coordinates to pixel coordinates
            pixel_keypoints = {}
            for name, (x, y) in keypoints.items():
                pixel_keypoints[name] = (x * frame_width, y * frame_height)
            
            # 1. Front elbow angle (assuming right-handed batsman, use right elbow)
            if all(k in pixel_keypoints for k in ['right_shoulder', 'right_elbow', 'right_wrist']):
                elbow_angle = self.calculate_angle(
                    pixel_keypoints['right_shoulder'],
                    pixel_keypoints['right_elbow'],
                    pixel_keypoints['right_wrist']
                )
                metrics['front_elbow_angle'] = elbow_angle
            
            # 2. Spine lean (hip-shoulder line vs vertical)
            if all(k in pixel_keypoints for k in ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']):
                # Calculate hip center and shoulder center
                hip_center = (
                    (pixel_keypoints['left_hip'][0] + pixel_keypoints['right_hip'][0]) / 2,
                    (pixel_keypoints['left_hip'][1] + pixel_keypoints['right_hip'][1]) / 2
                )
                shoulder_center = (
                    (pixel_keypoints['left_shoulder'][0] + pixel_keypoints['right_shoulder'][0]) / 2,
                    (pixel_keypoints['left_shoulder'][1] + pixel_keypoints['right_shoulder'][1]) / 2
                )
                
                # Calculate angle with vertical
                spine_vector = (shoulder_center[0] - hip_center[0], shoulder_center[1] - hip_center[1])
                vertical_vector = (0, -1)  # Pointing up
                
                dot_product = spine_vector[0] * vertical_vector[0] + spine_vector[1] * vertical_vector[1]
                spine_magnitude = math.sqrt(spine_vector[0]**2 + spine_vector[1]**2)
                
                if spine_magnitude > 0:
                    cos_angle = dot_product / spine_magnitude
                    cos_angle = max(-1, min(1, cos_angle))
                    spine_lean = math.degrees(math.acos(cos_angle))
                    metrics['spine_lean'] = spine_lean
            
            # 3. Head-over-knee vertical alignment
            if all(k in pixel_keypoints for k in ['nose', 'right_knee']):
                head_knee_distance = abs(pixel_keypoints['nose'][0] - pixel_keypoints['right_knee'][0])
                metrics['head_knee_alignment'] = head_knee_distance
            
            # 4. Front foot direction (using ankle and knee)
            if all(k in pixel_keypoints for k in ['right_knee', 'right_ankle']):
                foot_vector = (
                    pixel_keypoints['right_ankle'][0] - pixel_keypoints['right_knee'][0],
                    pixel_keypoints['right_ankle'][1] - pixel_keypoints['right_knee'][1]
                )
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
                    metrics['front_foot_direction'] = foot_angle
                    
        except Exception as e:
            print(f"Error computing metrics: {e}")
        
        return metrics
    
    def draw_overlays(self, frame: np.ndarray, keypoints: Dict[str, Tuple[float, float]], 
                     metrics: Dict[str, float], frame_count: int) -> np.ndarray:
        """Draw pose skeleton and metrics overlays on frame"""
        frame_height, frame_width = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        pixel_keypoints = {}
        for name, (x, y) in keypoints.items():
            pixel_keypoints[name] = (int(x * frame_width), int(y * frame_height))
        
        # Draw pose skeleton
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
            ('right_knee', 'right_ankle')
        ]
        
        # Draw connections
        for start, end in connections:
            if start in pixel_keypoints and end in pixel_keypoints:
                cv2.line(frame, pixel_keypoints[start], pixel_keypoints[end], (0, 255, 0), 2)
        
        # Draw keypoints
        for name, (x, y) in pixel_keypoints.items():
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        # Draw metrics overlay
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, y_offset), font, font_scale, (255, 255, 255), thickness)
        y_offset += 30
        
        # Metrics with feedback
        for metric_name, value in metrics.items():
            if metric_name == 'front_elbow_angle':
                color = (0, 255, 0) if self.thresholds['elbow_angle_good'][0] <= value <= self.thresholds['elbow_angle_good'][1] else (0, 0, 255)
                feedback = "✅ Good elbow" if color == (0, 255, 0) else "❌ Check elbow"
                cv2.putText(frame, f"Elbow: {value:.1f}° {feedback}", (10, y_offset), font, font_scale, color, thickness)
                
            elif metric_name == 'spine_lean':
                color = (0, 255, 0) if self.thresholds['spine_lean_good'][0] <= value <= self.thresholds['spine_lean_good'][1] else (0, 0, 255)
                feedback = "✅ Good posture" if color == (0, 255, 0) else "❌ Check posture"
                cv2.putText(frame, f"Spine: {value:.1f}° {feedback}", (10, y_offset), font, font_scale, color, thickness)
                
            elif metric_name == 'head_knee_alignment':
                color = (0, 255, 0) if value <= self.thresholds['head_alignment_good'] else (0, 0, 255)
                feedback = "✅ Head aligned" if color == (0, 255, 0) else "❌ Head not over knee"
                cv2.putText(frame, f"Head-Knee: {value:.1f}px {feedback}", (10, y_offset), font, font_scale, color, thickness)
                
            elif metric_name == 'front_foot_direction':
                color = (0, 255, 0) if self.thresholds['foot_angle_good'][0] <= value <= self.thresholds['foot_angle_good'][1] else (0, 0, 255)
                feedback = "✅ Good footwork" if color == (0, 255, 0) else "❌ Check foot angle"
                cv2.putText(frame, f"Foot: {value:.1f}° {feedback}", (10, y_offset), font, font_scale, color, thickness)
            
            y_offset += 25
        
        return frame
    
    def evaluate_shot(self) -> Dict[str, any]:
        """Generate final shot evaluation based on collected metrics"""
        if not self.frame_metrics:
            return {"error": "No metrics collected"}
        
        # Calculate average metrics
        avg_metrics = {}
        metric_names = ['front_elbow_angle', 'spine_lean', 'head_knee_alignment', 'front_foot_direction']
        
        for metric in metric_names:
            values = [frame[metric] for frame in self.frame_metrics if metric in frame]
            if values:
                avg_metrics[metric] = sum(values) / len(values)
        
        # Score each category (1-10)
        scores = {}
        feedback = {}
        
        # Footwork (based on foot direction)
        if 'front_foot_direction' in avg_metrics:
            foot_angle = avg_metrics['front_foot_direction']
            if self.thresholds['foot_angle_good'][0] <= foot_angle <= self.thresholds['foot_angle_good'][1]:
                scores['footwork'] = 8 + min(2, (15 - abs(foot_angle)) / 7.5)
                feedback['footwork'] = ["Excellent foot positioning", "Maintain this alignment"]
            else:
                scores['footwork'] = max(3, 7 - abs(foot_angle) / 5)
                feedback['footwork'] = ["Adjust front foot direction", "Aim for perpendicular to crease"]
        else:
            scores['footwork'] = 5
            feedback['footwork'] = ["Foot positioning unclear", "Ensure clear foot visibility"]
        
        # Head Position (based on head-knee alignment)
        if 'head_knee_alignment' in avg_metrics:
            alignment = avg_metrics['head_knee_alignment']
            if alignment <= self.thresholds['head_alignment_good']:
                scores['head_position'] = 8 + min(2, (50 - alignment) / 25)
                feedback['head_position'] = ["Great head position", "Keep head over front knee"]
            else:
                scores['head_position'] = max(3, 8 - alignment / 25)
                feedback['head_position'] = ["Move head over front knee", "Improve balance and timing"]
        else:
            scores['head_position'] = 5
            feedback['head_position'] = ["Head position unclear", "Focus on head stability"]
        
        # Swing Control (based on elbow angle)
        if 'front_elbow_angle' in avg_metrics:
            elbow = avg_metrics['front_elbow_angle']
            if self.thresholds['elbow_angle_good'][0] <= elbow <= self.thresholds['elbow_angle_good'][1]:
                scores['swing_control'] = 8 + min(2, (130 - abs(elbow - 110)) / 20)
                feedback['swing_control'] = ["Excellent elbow position", "Maintain this swing path"]
            else:
                scores['swing_control'] = max(3, 7 - abs(elbow - 110) / 15)
                feedback['swing_control'] = ["Adjust elbow angle", "Aim for 90-130 degree range"]
        else:
            scores['swing_control'] = 5
            feedback['swing_control'] = ["Swing mechanics unclear", "Focus on elbow positioning"]
        
        # Balance (based on spine lean)
        if 'spine_lean' in avg_metrics:
            spine = avg_metrics['spine_lean']
            if self.thresholds['spine_lean_good'][0] <= spine <= self.thresholds['spine_lean_good'][1]:
                scores['balance'] = 8 + min(2, (25 - abs(spine - 17.5)) / 7.5)
                feedback['balance'] = ["Excellent balance", "Good spine angle maintained"]
            else:
                scores['balance'] = max(3, 7 - abs(spine - 17.5) / 10)
                feedback['balance'] = ["Improve spine angle", "Maintain slight forward lean"]
        else:
            scores['balance'] = 5
            feedback['balance'] = ["Balance assessment unclear", "Focus on posture"]
        
        # Follow-through (based on consistency of metrics)
        consistency_score = 0
        for metric in metric_names:
            values = [frame[metric] for frame in self.frame_metrics if metric in frame]
            if values and len(values) > 1:
                std_dev = np.std(values)
                max_val = max(values)
                if max_val > 0:
                    consistency = 1 - (std_dev / max_val)
                    consistency_score += max(0, consistency)
        
        if len(metric_names) > 0:
            consistency_score /= len(metric_names)
            scores['follow_through'] = max(3, min(10, 5 + consistency_score * 5))
            if consistency_score > 0.7:
                feedback['follow_through'] = ["Smooth follow-through", "Consistent technique"]
            else:
                feedback['follow_through'] = ["Work on consistency", "Practice smooth follow-through"]
        else:
            scores['follow_through'] = 5
            feedback['follow_through'] = ["Follow-through unclear", "Focus on completion"]
        
        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores)
        
        evaluation = {
            "overall_score": round(overall_score, 1),
            "category_scores": {k: round(v, 1) for k, v in scores.items()},
            "feedback": feedback,
            "average_metrics": {k: round(v, 2) for k, v in avg_metrics.items()},
            "total_frames_analyzed": len(self.frame_metrics),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return evaluation
    
    def process_video(self, video_path: str, output_path: str = "output/annotated_video.mp4") -> bool:
        """Process the entire video with pose estimation and generate annotated output"""
        print(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        # Process frames with progress bar
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_start = time.time()
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process pose
                results = self.pose.process(rgb_frame)
                
                if results.pose_landmarks:
                    # Extract keypoints
                    keypoints = self.extract_keypoints(results.pose_landmarks)
                    
                    if keypoints:
                        # Compute metrics
                        metrics = self.compute_biomechanical_metrics(keypoints, width, height)
                        
                        # Store metrics for final evaluation
                        self.frame_metrics.append(metrics)
                        
                        # Draw overlays
                        frame = self.draw_overlays(frame, keypoints, metrics, frame_count)
                
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
        
        return True
    
    def save_evaluation(self, evaluation: Dict, output_path: str = "output/evaluation.json"):
        """Save evaluation results to file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(evaluation, f, indent=2)
            print(f"Evaluation saved: {output_path}")
            
            # Also save as readable text
            txt_path = output_path.replace('.json', '.txt')
            with open(txt_path, 'w') as f:
                f.write("AthleteRise Cricket Cover Drive Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Overall Score: {evaluation['overall_score']}/10\n\n")
                
                f.write("Category Scores:\n")
                for category, score in evaluation['category_scores'].items():
                    f.write(f"  {category.replace('_', ' ').title()}: {score}/10\n")
                
                f.write("\nDetailed Feedback:\n")
                for category, feedback_list in evaluation['feedback'].items():
                    f.write(f"\n{category.replace('_', ' ').title()}:\n")
                    for feedback in feedback_list:
                        f.write(f"  • {feedback}\n")
                
                f.write(f"\nFrames Analyzed: {evaluation['total_frames_analyzed']}\n")
                f.write(f"Analysis Date: {evaluation['timestamp']}\n")
            
            print(f"Readable report saved: {txt_path}")
            return True
            
        except Exception as e:
            print(f"Error saving evaluation: {e}")
            return False

def main():
    """Main function to run the cricket analysis"""
    print("AthleteRise – AI-Powered Cricket Analytics")
    print("Real-Time Cover Drive Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = CricketAnalyzer()
    
    # Video URL from assignment
    video_url = "https://youtube.com/shorts/vSX3IRxGnNY"
    
    # Download video
    video_path = analyzer.download_video(video_url, "input_video.mp4")
    if not video_path:
        print("Failed to download video. Exiting.")
        return
    
    # Process video
    success = analyzer.process_video(video_path, "output/annotated_video.mp4")
    if not success:
        print("Failed to process video. Exiting.")
        return
    
    # Generate evaluation
    print("\nGenerating shot evaluation...")
    evaluation = analyzer.evaluate_shot()
    
    # Save evaluation
    analyzer.save_evaluation(evaluation)
    
    # Print summary
    print("\n" + "=" * 50)
    print("ANALYSIS COMPLETE")
    print("=" * 50)
    print(f"Overall Score: {evaluation['overall_score']}/10")
    print("\nCategory Scores:")
    for category, score in evaluation['category_scores'].items():
        print(f"  {category.replace('_', ' ').title()}: {score}/10")
    
    print(f"\nFiles generated:")
    print(f"  • Annotated video: output/annotated_video.mp4")
    print(f"  • Evaluation report: output/evaluation.json")
    print(f"  • Readable report: output/evaluation.txt")
    
    # Cleanup downloaded video
    try:
        os.remove(video_path)
        print(f"  • Cleaned up: {video_path}")
    except:
        pass

if __name__ == "__main__":
    main()
