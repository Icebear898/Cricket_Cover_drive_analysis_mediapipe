
"""
AthleteRise Cricket Analytics - Performance Evaluator Module
Handles shot evaluation and scoring based on collected metrics
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any

class PerformanceEvaluator:
    """Evaluates cricket shot performance based on biomechanical metrics"""
    
    def __init__(self):
        """Initialize performance evaluator with scoring criteria"""
        self.thresholds = {
            'elbow_angle_good': (90, 130),
            'spine_lean_good': (10, 25),
            'head_alignment_good': 50,  # pixels
            'foot_angle_good': (-15, 15)  # degrees from perpendicular
        }
        
        # Scoring weights for different aspects
        self.weights = {
            'footwork': 1.0,
            'head_position': 1.0,
            'swing_control': 1.2,  # Slightly more important
            'balance': 1.0,
            'follow_through': 0.8  # Based on consistency
        }
    
    def calculate_average_metrics(self, frame_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate average metrics across all frames
        
        Args:
            frame_metrics: List of metrics dictionaries from each frame
            
        Returns:
            Dictionary of average metric values
        """
        if not frame_metrics:
            return {}
        
        avg_metrics = {}
        metric_names = ['front_elbow_angle', 'spine_lean', 'head_knee_alignment', 'front_foot_direction']
        
        for metric in metric_names:
            values = [frame[metric] for frame in frame_metrics if metric in frame]
            if values:
                avg_metrics[metric] = sum(values) / len(values)
        
        return avg_metrics
    
    def calculate_consistency_score(self, frame_metrics: List[Dict[str, float]]) -> float:
        """
        Calculate consistency score based on metric variance
        
        Args:
            frame_metrics: List of metrics dictionaries from each frame
            
        Returns:
            Consistency score (0-1, higher is better)
        """
        if not frame_metrics:
            return 0.0
        
        metric_names = ['front_elbow_angle', 'spine_lean', 'head_knee_alignment', 'front_foot_direction']
        consistency_scores = []
        
        for metric in metric_names:
            values = [frame[metric] for frame in frame_metrics if metric in frame]
            if values and len(values) > 1:
                std_dev = np.std(values)
                mean_val = np.mean(values)
                if mean_val > 0:
                    # Coefficient of variation (lower is more consistent)
                    cv = std_dev / abs(mean_val)
                    # Convert to consistency score (0-1, higher is better)
                    consistency = max(0, 1 - cv)
                    consistency_scores.append(consistency)
        
        return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0
    
    def score_footwork(self, avg_metrics: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Score footwork based on front foot direction
        
        Args:
            avg_metrics: Dictionary of average metrics
            
        Returns:
            Tuple of (score, feedback_list)
        """
        if 'front_foot_direction' not in avg_metrics:
            return 5.0, ["Foot positioning unclear", "Ensure clear foot visibility"]
        
        foot_angle = avg_metrics['front_foot_direction']
        min_angle, max_angle = self.thresholds['foot_angle_good']
        
        if min_angle <= foot_angle <= max_angle:
            # Excellent range
            score = 8 + min(2, (15 - abs(foot_angle)) / 7.5)
            feedback = ["Excellent foot positioning", "Maintain this alignment"]
        else:
            # Outside optimal range
            deviation = min(abs(foot_angle - min_angle), abs(foot_angle - max_angle))
            score = max(3, 7 - deviation / 5)
            feedback = ["Adjust front foot direction", "Aim for perpendicular to crease"]
        
        return score, feedback
    
    def score_head_position(self, avg_metrics: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Score head position based on head-knee alignment
        
        Args:
            avg_metrics: Dictionary of average metrics
            
        Returns:
            Tuple of (score, feedback_list)
        """
        if 'head_knee_alignment' not in avg_metrics:
            return 5.0, ["Head position unclear", "Focus on head stability"]
        
        alignment = avg_metrics['head_knee_alignment']
        threshold = self.thresholds['head_alignment_good']
        
        if alignment <= threshold:
            # Good alignment
            score = 8 + min(2, (threshold - alignment) / (threshold / 2))
            feedback = ["Great head position", "Keep head over front knee"]
        else:
            # Poor alignment
            score = max(3, 8 - (alignment - threshold) / 25)
            feedback = ["Move head over front knee", "Improve balance and timing"]
        
        return score, feedback
    
    def score_swing_control(self, avg_metrics: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Score swing control based on elbow angle
        
        Args:
            avg_metrics: Dictionary of average metrics
            
        Returns:
            Tuple of (score, feedback_list)
        """
        if 'front_elbow_angle' not in avg_metrics:
            return 5.0, ["Swing mechanics unclear", "Focus on elbow positioning"]
        
        elbow = avg_metrics['front_elbow_angle']
        min_angle, max_angle = self.thresholds['elbow_angle_good']
        optimal_angle = (min_angle + max_angle) / 2
        
        if min_angle <= elbow <= max_angle:
            # Good range
            deviation_from_optimal = abs(elbow - optimal_angle)
            score = 8 + min(2, (20 - deviation_from_optimal) / 20)
            feedback = ["Excellent elbow position", "Maintain this swing path"]
        else:
            # Outside optimal range
            deviation = min(abs(elbow - min_angle), abs(elbow - max_angle))
            score = max(3, 7 - deviation / 15)
            feedback = ["Adjust elbow angle", f"Aim for {min_angle}-{max_angle} degree range"]
        
        return score, feedback
    
    def score_balance(self, avg_metrics: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Score balance based on spine lean
        
        Args:
            avg_metrics: Dictionary of average metrics
            
        Returns:
            Tuple of (score, feedback_list)
        """
        if 'spine_lean' not in avg_metrics:
            return 5.0, ["Balance assessment unclear", "Focus on posture"]
        
        spine = avg_metrics['spine_lean']
        min_lean, max_lean = self.thresholds['spine_lean_good']
        optimal_lean = (min_lean + max_lean) / 2
        
        if min_lean <= spine <= max_lean:
            # Good range
            deviation_from_optimal = abs(spine - optimal_lean)
            score = 8 + min(2, (7.5 - deviation_from_optimal) / 7.5)
            feedback = ["Excellent balance", "Good spine angle maintained"]
        else:
            # Outside optimal range
            deviation = min(abs(spine - min_lean), abs(spine - max_lean))
            score = max(3, 7 - deviation / 10)
            feedback = ["Improve spine angle", "Maintain slight forward lean"]
        
        return score, feedback
    
    def score_follow_through(self, consistency_score: float) -> Tuple[float, List[str]]:
        """
        Score follow-through based on technique consistency
        
        Args:
            consistency_score: Consistency score (0-1)
            
        Returns:
            Tuple of (score, feedback_list)
        """
        # Convert consistency score to 1-10 scale
        base_score = 5 + consistency_score * 5
        score = max(3, min(10, base_score))
        
        if consistency_score > 0.7:
            feedback = ["Smooth follow-through", "Consistent technique"]
        elif consistency_score > 0.4:
            feedback = ["Good consistency", "Minor technique variations"]
        else:
            feedback = ["Work on consistency", "Practice smooth follow-through"]
        
        return score, feedback
    
    def evaluate_shot(self, frame_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Generate comprehensive shot evaluation
        
        Args:
            frame_metrics: List of metrics from all processed frames
            
        Returns:
            Complete evaluation dictionary
        """
        if not frame_metrics:
            return {"error": "No metrics collected"}
        
        # Calculate average metrics and consistency
        avg_metrics = self.calculate_average_metrics(frame_metrics)
        consistency_score = self.calculate_consistency_score(frame_metrics)
        
        # Score each category
        scores = {}
        feedback = {}
        
        # Footwork
        footwork_score, footwork_feedback = self.score_footwork(avg_metrics)
        scores['footwork'] = footwork_score * self.weights['footwork']
        feedback['footwork'] = footwork_feedback
        
        # Head Position
        head_score, head_feedback = self.score_head_position(avg_metrics)
        scores['head_position'] = head_score * self.weights['head_position']
        feedback['head_position'] = head_feedback
        
        # Swing Control
        swing_score, swing_feedback = self.score_swing_control(avg_metrics)
        scores['swing_control'] = swing_score * self.weights['swing_control']
        feedback['swing_control'] = swing_feedback
        
        # Balance
        balance_score, balance_feedback = self.score_balance(avg_metrics)
        scores['balance'] = balance_score * self.weights['balance']
        feedback['balance'] = balance_feedback
        
        # Follow-through
        follow_score, follow_feedback = self.score_follow_through(consistency_score)
        scores['follow_through'] = follow_score * self.weights['follow_through']
        feedback['follow_through'] = follow_feedback
        
        # Calculate overall score (weighted average)
        total_weight = sum(self.weights.values())
        overall_score = sum(scores.values()) / total_weight
        
        # Determine skill level
        if overall_score >= 8.5:
            skill_level = "Advanced"
        elif overall_score >= 6.5:
            skill_level = "Intermediate"
        else:
            skill_level = "Beginner"
        
        # Create comprehensive evaluation
        evaluation = {
            "overall_score": round(overall_score, 1),
            "skill_level": skill_level,
            "category_scores": {k: round(v / self.weights[k], 1) for k, v in scores.items()},
            "weighted_scores": {k: round(v, 1) for k, v in scores.items()},
            "feedback": feedback,
            "average_metrics": {k: round(v, 2) for k, v in avg_metrics.items()},
            "consistency_score": round(consistency_score, 3),
            "total_frames_analyzed": len(frame_metrics),
            "frames_with_complete_data": len([f for f in frame_metrics if len(f) >= 3]),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_quality": self._assess_analysis_quality(frame_metrics)
        }
        
        return evaluation
    
    def _assess_analysis_quality(self, frame_metrics: List[Dict[str, float]]) -> str:
        """
        Assess the quality of the analysis based on data completeness
        
        Args:
            frame_metrics: List of metrics from all processed frames
            
        Returns:
            Quality assessment string
        """
        if not frame_metrics:
            return "No data"
        
        total_frames = len(frame_metrics)
        complete_frames = len([f for f in frame_metrics if len(f) >= 3])
        completeness_ratio = complete_frames / total_frames
        
        if completeness_ratio >= 0.8:
            return "Excellent"
        elif completeness_ratio >= 0.6:
            return "Good"
        elif completeness_ratio >= 0.4:
            return "Fair"
        else:
            return "Poor"
    
    def save_evaluation(self, evaluation: Dict[str, Any], 
                       output_path: str = "output/evaluation.json") -> bool:
        """
        Save evaluation results to files
        
        Args:
            evaluation: Evaluation results dictionary
            output_path: Output file path for JSON
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Save JSON format
            with open(output_path, 'w') as f:
                json.dump(evaluation, f, indent=2)
            print(f"Evaluation saved: {output_path}")
            
            # Save readable text format
            txt_path = output_path.replace('.json', '.txt')
            with open(txt_path, 'w') as f:
                f.write("AthleteRise Cricket Cover Drive Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Overall Score: {evaluation['overall_score']}/10\n")
                f.write(f"Skill Level: {evaluation['skill_level']}\n")
                f.write(f"Analysis Quality: {evaluation['analysis_quality']}\n\n")
                
                f.write("Category Scores:\n")
                for category, score in evaluation['category_scores'].items():
                    f.write(f"  {category.replace('_', ' ').title()}: {score}/10\n")
                
                f.write(f"\nConsistency Score: {evaluation['consistency_score']}\n\n")
                
                f.write("Detailed Feedback:\n")
                for category, feedback_list in evaluation['feedback'].items():
                    f.write(f"\n{category.replace('_', ' ').title()}:\n")
                    for feedback in feedback_list:
                        f.write(f"  • {feedback}\n")
                
                f.write(f"\nFrames Analyzed: {evaluation['total_frames_analyzed']}\n")
                f.write(f"Complete Data Frames: {evaluation['frames_with_complete_data']}\n")
                f.write(f"Analysis Date: {evaluation['timestamp']}\n")
                
                if 'average_metrics' in evaluation:
                    f.write(f"\nAverage Biomechanical Metrics:\n")
                    for metric, value in evaluation['average_metrics'].items():
                        f.write(f"  {metric.replace('_', ' ').title()}: {value}\n")
            
            print(f"Readable report saved: {txt_path}")
            return True
            
        except Exception as e:
            print(f"Error saving evaluation: {e}")
            return False
    
    def generate_improvement_plan(self, evaluation: Dict[str, Any]) -> List[str]:
        """
        Generate personalized improvement recommendations
        
        Args:
            evaluation: Evaluation results dictionary
            
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        scores = evaluation.get('category_scores', {})
        
        # Priority order for improvements
        priority_order = ['swing_control', 'balance', 'head_position', 'footwork', 'follow_through']
        
        for category in priority_order:
            if category in scores and scores[category] < 7:
                if category == 'swing_control':
                    recommendations.append("Focus on elbow positioning during the swing")
                elif category == 'balance':
                    recommendations.append("Work on maintaining proper spine angle and posture")
                elif category == 'head_position':
                    recommendations.append("Practice keeping head over front knee during shot")
                elif category == 'footwork':
                    recommendations.append("Improve front foot positioning perpendicular to crease")
                elif category == 'follow_through':
                    recommendations.append("Practice consistent technique and smooth follow-through")
        
        # Add consistency recommendation if needed
        if evaluation.get('consistency_score', 0) < 0.6:
            recommendations.append("Focus on developing consistent technique across all shots")
        
        return recommendations[:3]  # Return top 3 recommendations
