#!/usr/bin/env python3
"""
AthleteRise – AI-Powered Cricket Analytics
Refactored Main Script - Real-Time Cover Drive Analysis

This is the refactored version using modular components.
Author: AthleteRise Team
Date: 2025-08-14
"""

import os
import sys
import time
from typing import Optional

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules import (
    PoseEstimator,
    MetricsCalculator,
    VideoProcessor,
    PerformanceEvaluator,
    AnalysisUtils
)

class CricketAnalyzer:
    """Main orchestrator for cricket cover drive analysis"""
    
    def __init__(self):
        """Initialize the cricket analyzer with all components"""
        print("Initializing AthleteRise Cricket Analyzer...")
        
        # Initialize all components
        self.pose_estimator = PoseEstimator(
            model_complexity=1,  # Lightweight for speed
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.metrics_calculator = MetricsCalculator()
        self.video_processor = VideoProcessor()
        self.evaluator = PerformanceEvaluator()
        self.utils = AnalysisUtils()
        
        # Ensure output directory exists
        self.output_dir = self.utils.ensure_output_directory("output")
        
        print("✅ All components initialized successfully!")
    
    def analyze_video_from_url(self, url: str, output_name: str = "annotated_video") -> bool:
        """
        Analyze cricket video from YouTube URL
        
        Args:
            url: YouTube video URL
            output_name: Base name for output files
            
        Returns:
            True if analysis successful, False otherwise
        """
        print(f"🏏 Starting analysis of video: {url}")
        
        try:
            # Step 1: Download video
            print("📥 Downloading video...")
            temp_video_path = "temp_input_video.mp4"
            video_path = self.video_processor.download_video(url, temp_video_path)
            
            if not video_path:
                print("❌ Failed to download video")
                return False
            
            # Step 2: Process video with analysis
            print("🔍 Processing video with pose estimation...")
            output_video_path = os.path.join(self.output_dir, f"{output_name}.mp4")
            
            success, frame_metrics = self.video_processor.process_video_with_analysis(
                video_path=video_path,
                pose_estimator=self.pose_estimator,
                metrics_calculator=self.metrics_calculator,
                output_path=output_video_path
            )
            
            if not success:
                print("❌ Failed to process video")
                return False
            
            # Step 3: Generate evaluation
            print("📊 Generating performance evaluation...")
            evaluation = self.evaluator.evaluate_shot(frame_metrics)
            
            # Step 4: Save evaluation
            json_path = os.path.join(self.output_dir, f"{output_name}_evaluation.json")
            self.evaluator.save_evaluation(evaluation, json_path)
            
            # Step 5: Generate improvement recommendations
            recommendations = self.evaluator.generate_improvement_plan(evaluation)
            
            # Step 6: Display results
            self._display_results(evaluation, recommendations)
            
            # Step 7: Cleanup
            self.video_processor.cleanup_temp_files([video_path])
            
            return True
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            return False
    
    def analyze_local_video(self, video_path: str, output_name: str = "annotated_video") -> bool:
        """
        Analyze local cricket video file
        
        Args:
            video_path: Path to local video file
            output_name: Base name for output files
            
        Returns:
            True if analysis successful, False otherwise
        """
        print(f"🏏 Starting analysis of local video: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return False
        
        try:
            # Step 1: Process video with analysis
            print("🔍 Processing video with pose estimation...")
            output_video_path = os.path.join(self.output_dir, f"{output_name}.mp4")
            
            success, frame_metrics = self.video_processor.process_video_with_analysis(
                video_path=video_path,
                pose_estimator=self.pose_estimator,
                metrics_calculator=self.metrics_calculator,
                output_path=output_video_path
            )
            
            if not success:
                print("❌ Failed to process video")
                return False
            
            # Step 2: Generate evaluation
            print("📊 Generating performance evaluation...")
            evaluation = self.evaluator.evaluate_shot(frame_metrics)
            
            # Step 3: Save evaluation
            json_path = os.path.join(self.output_dir, f"{output_name}_evaluation.json")
            self.evaluator.save_evaluation(evaluation, json_path)
            
            # Step 4: Generate improvement recommendations
            recommendations = self.evaluator.generate_improvement_plan(evaluation)
            
            # Step 5: Display results
            self._display_results(evaluation, recommendations)
            
            return True
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            return False
    
    def _display_results(self, evaluation: dict, recommendations: list):
        """Display analysis results in a formatted way"""
        print("\n" + "="*60)
        print("🏏 ATHLETERISE CRICKET ANALYSIS RESULTS")
        print("="*60)
        
        # Overall performance
        overall_score = evaluation.get('overall_score', 0)
        skill_level = evaluation.get('skill_level', 'Unknown')
        print(f"📊 Overall Score: {overall_score}/10 ({skill_level})")
        print(f"🎯 Analysis Quality: {evaluation.get('analysis_quality', 'Unknown')}")
        
        # Category scores
        print(f"\n📈 Category Breakdown:")
        if 'category_scores' in evaluation:
            for category, score in evaluation['category_scores'].items():
                emoji = "🟢" if score >= 7 else "🟡" if score >= 5 else "🔴"
                print(f"  {emoji} {self.utils.format_metric_name(category)}: {score}/10")
        
        # Key insights
        print(f"\n💡 Key Insights:")
        if 'category_scores' in evaluation:
            scores = evaluation['category_scores']
            best_category = max(scores.items(), key=lambda x: x[1])
            worst_category = min(scores.items(), key=lambda x: x[1])
            
            print(f"  ✅ Strongest: {self.utils.format_metric_name(best_category[0])} ({best_category[1]}/10)")
            print(f"  ⚠️  Needs Work: {self.utils.format_metric_name(worst_category[0])} ({worst_category[1]}/10)")
        
        # Improvement recommendations
        if recommendations:
            print(f"\n🎯 Top Improvement Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        # Technical details
        print(f"\n📋 Technical Details:")
        print(f"  • Frames Analyzed: {evaluation.get('total_frames_analyzed', 0)}")
        print(f"  • Complete Data Frames: {evaluation.get('frames_with_complete_data', 0)}")
        print(f"  • Consistency Score: {evaluation.get('consistency_score', 0):.3f}")
        
        # Processing stats
        stats = self.video_processor.get_processing_stats()
        if stats:
            print(f"  • Average Processing FPS: {stats.get('average_fps', 0):.1f}")
        
        # Output files
        print(f"\n📁 Generated Files:")
        print(f"  • Annotated Video: {self.output_dir}/annotated_video.mp4")
        print(f"  • JSON Report: {self.output_dir}/annotated_video_evaluation.json")
        print(f"  • Text Report: {self.output_dir}/annotated_video_evaluation.txt")
        
        print("="*60)
    
    def get_system_info(self):
        """Display system information for debugging"""
        print("\n🔧 System Information:")
        info = self.utils.get_system_info()
        for key, value in info.items():
            print(f"  • {key}: {value}")
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up resources...")
        if hasattr(self.pose_estimator, 'cleanup'):
            self.pose_estimator.cleanup()
        print("✅ Cleanup complete!")

def main():
    """Main function to run the cricket analysis"""
    print("🏏 AthleteRise – AI-Powered Cricket Analytics")
    print("Real-Time Cover Drive Analysis (Refactored Version)")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = CricketAnalyzer()
    
    try:
        # Default video URL from assignment
        video_url = "https://youtube.com/shorts/vSX3IRxGnNY"
        
        print(f"\n🎬 Analyzing assignment video...")
        success = analyzer.analyze_video_from_url(video_url, "cover_drive_analysis")
        
        if success:
            print("\n🎉 Analysis completed successfully!")
            print("📱 You can also run the Streamlit app for interactive analysis:")
            print("   streamlit run app.py")
        else:
            print("\n❌ Analysis failed. Please check the logs above.")
            
            # Try with a local demo video if available
            demo_videos = ["demo.mp4", "sample.mp4", "test.mp4"]
            for demo in demo_videos:
                if os.path.exists(demo):
                    print(f"\n🔄 Trying with local demo video: {demo}")
                    success = analyzer.analyze_local_video(demo, "demo_analysis")
                    if success:
                        print("✅ Demo analysis completed!")
                    break
    
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
    finally:
        analyzer.cleanup()

if __name__ == "__main__":
    main()
