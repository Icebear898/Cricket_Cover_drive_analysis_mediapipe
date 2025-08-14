#!/usr/bin/env python3
"""
AthleteRise Cricket Analytics - Streamlit Web Interface
Interactive web app for cricket cover drive analysis
"""

import streamlit as st
import cv2
import os
import json
import time
import sys
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules import (
    PoseEstimator,
    MetricsCalculator,
    VideoProcessor,
    PerformanceEvaluator,
    AnalysisUtils
)

st.set_page_config(
    page_title="AthleteRise Cricket Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .score-excellent {
        color: #28a745;
        font-weight: bold;
    }
    .score-good {
        color: #ffc107;
        font-weight: bold;
    }
    .score-needs-work {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def get_score_color_class(score):
    """Get CSS class based on score"""
    if score >= 8:
        return "score-excellent"
    elif score >= 6:
        return "score-good"
    else:
        return "score-needs-work"

def create_score_gauge(score, title):
    """Create a gauge chart for scores"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 5},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 5], 'color': "lightgray"},
                {'range': [5, 8], 'color': "yellow"},
                {'range': [8, 10], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 9
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def main():
    st.markdown('<h1 class="main-header">🏏 AthleteRise Cricket Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Real-Time Cover Drive Analysis</p>', unsafe_allow_html=True)
    
    st.sidebar.title("Analysis Options")
    
    analysis_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["YouTube URL", "Upload Video", "Demo Analysis"]
    )
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'evaluation_data' not in st.session_state:
        st.session_state.evaluation_data = None
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Video Analysis")
        
        if analysis_mode == "YouTube URL":
            video_url = st.text_input(
                "Enter YouTube URL:",
                value="https://youtube.com/shorts/vSX3IRxGnNY",
                help="Enter a YouTube video URL for analysis"
            )
            
            if st.button("Analyze Video", type="primary"):
                if video_url:
                    analyze_video_from_url(video_url)
                else:
                    st.error("Please enter a valid YouTube URL")
        
        elif analysis_mode == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload Cricket Video",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Upload a cricket video file for analysis"
            )
            
            if uploaded_file and st.button("Analyze Uploaded Video", type="primary"):
                analyze_uploaded_video(uploaded_file)
        
        elif analysis_mode == "Demo Analysis":
            st.info("Demo mode will analyze the default YouTube video from the assignment.")
            if st.button("Run Demo Analysis", type="primary"):
                demo_url = "https://youtube.com/shorts/vSX3IRxGnNY"
                analyze_video_from_url(demo_url)
    
    with col2:
        st.subheader("Analysis Settings")
        
        # Threshold adjustments
        with st.expander("Adjust Thresholds"):
            elbow_min = st.slider("Elbow Angle Min", 70, 100, 90)
            elbow_max = st.slider("Elbow Angle Max", 120, 150, 130)
            spine_min = st.slider("Spine Lean Min", 5, 15, 10)
            spine_max = st.slider("Spine Lean Max", 20, 35, 25)
            head_threshold = st.slider("Head Alignment Threshold", 30, 80, 50)
            foot_range = st.slider("Foot Angle Range", 10, 25, 15)
        
        # Performance info
        st.info("""
        **Analysis Features:**
        - Real-time pose estimation
        - Biomechanical metrics
        - Performance scoring
        - Visual feedback overlays
        - Detailed evaluation report
        """)
    
    # Results section
    if st.session_state.analysis_complete and st.session_state.evaluation_data:
        display_results()

class ModularCricketAnalyzer:
    """Streamlit-compatible wrapper for modular cricket analyzer"""
    
    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.metrics_calculator = MetricsCalculator()
        self.video_processor = VideoProcessor()
        self.evaluator = PerformanceEvaluator()
        self.utils = AnalysisUtils()
        self.utils.ensure_output_directory("output")

def analyze_video_from_url(video_url):
    """Analyze video from YouTube URL"""
    try:
        # Initialize modular analyzer
        analyzer = ModularCricketAnalyzer()
        
        with st.spinner("Downloading and analyzing video..."):
            # Create progress bars
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Download video
            status_text.text("Downloading video...")
            progress_bar.progress(20)
            
            video_path = analyzer.video_processor.download_video(video_url, "temp_video.mp4")
            if not video_path:
                st.error("Failed to download video. Please check the URL.")
                return
            
            # Process video
            status_text.text("Processing video frames...")
            progress_bar.progress(40)
            
            success, frame_metrics = analyzer.video_processor.process_video_with_analysis(
                video_path=video_path,
                pose_estimator=analyzer.pose_estimator,
                metrics_calculator=analyzer.metrics_calculator,
                output_path="output/annotated_video.mp4"
            )
            
            if not success:
                st.error("Failed to process video.")
                return
            
            progress_bar.progress(80)
            status_text.text("Generating evaluation...")
            
            # Generate evaluation
            evaluation = analyzer.evaluator.evaluate_shot(frame_metrics)
            analyzer.evaluator.save_evaluation(evaluation)
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
            # Store results
            st.session_state.analyzer = analyzer
            st.session_state.evaluation_data = evaluation
            st.session_state.analysis_complete = True
            
            # Cleanup
            analyzer.video_processor.cleanup_temp_files([video_path])
            
            st.success("Analysis completed successfully!")
            st.rerun()
            
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")

def analyze_uploaded_video(uploaded_file):
    """Analyze uploaded video file"""
    try:
        # Save uploaded file
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize modular analyzer
        analyzer = ModularCricketAnalyzer()
        
        with st.spinner("Analyzing uploaded video..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Processing video frames...")
            progress_bar.progress(50)
            
            success, frame_metrics = analyzer.video_processor.process_video_with_analysis(
                video_path=temp_path,
                pose_estimator=analyzer.pose_estimator,
                metrics_calculator=analyzer.metrics_calculator,
                output_path="output/annotated_video.mp4"
            )
            
            if not success:
                st.error("Failed to process video.")
                return
            
            progress_bar.progress(80)
            status_text.text("Generating evaluation...")
            
            evaluation = analyzer.evaluator.evaluate_shot(frame_metrics)
            analyzer.evaluator.save_evaluation(evaluation)
            
            progress_bar.progress(100)
            status_text.text("Analysis complete!")
            
            # Store results
            st.session_state.analyzer = analyzer
            st.session_state.evaluation_data = evaluation
            st.session_state.analysis_complete = True
            
            # Cleanup
            analyzer.video_processor.cleanup_temp_files([temp_path])
            
            st.success("Analysis completed successfully!")
            st.rerun()
            
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")

def display_results():
    """Display analysis results"""
    evaluation = st.session_state.evaluation_data
    
    st.markdown("---")
    st.header("📊 Analysis Results")
    
    # Overall score
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        overall_score = evaluation['overall_score']
        st.metric(
            label="Overall Performance Score",
            value=f"{overall_score}/10",
            delta=f"{overall_score - 5:.1f} vs Average"
        )
    
    # Category scores with gauges
    st.subheader("Category Performance")
    
    categories = evaluation['category_scores']
    cols = st.columns(len(categories))
    
    for i, (category, score) in enumerate(categories.items()):
        with cols[i]:
            fig = create_score_gauge(score, category.replace('_', ' ').title())
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed feedback
    st.subheader("Detailed Feedback")
    
    for category, feedback_list in evaluation['feedback'].items():
        with st.expander(f"{category.replace('_', ' ').title()} - {categories[category]:.1f}/10"):
            for feedback in feedback_list:
                if "excellent" in feedback.lower() or "great" in feedback.lower():
                    st.success(f"✅ {feedback}")
                elif "good" in feedback.lower():
                    st.info(f"ℹ️ {feedback}")
                else:
                    st.warning(f"⚠️ {feedback}")
    
    # Metrics visualization
    if 'average_metrics' in evaluation:
        st.subheader("Biomechanical Metrics")
        
        metrics_data = evaluation['average_metrics']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Metrics table
            metrics_df = pd.DataFrame([
                {"Metric": "Front Elbow Angle", "Value": f"{metrics_data.get('front_elbow_angle', 0):.1f}°", "Optimal": "90-130°"},
                {"Metric": "Spine Lean", "Value": f"{metrics_data.get('spine_lean', 0):.1f}°", "Optimal": "10-25°"},
                {"Metric": "Head-Knee Alignment", "Value": f"{metrics_data.get('head_knee_alignment', 0):.1f}px", "Optimal": "<50px"},
                {"Metric": "Front Foot Direction", "Value": f"{metrics_data.get('front_foot_direction', 0):.1f}°", "Optimal": "±15°"}
            ])
            st.dataframe(metrics_df, use_container_width=True)
        
        with col2:
            # Radar chart with normalized values
            if len(metrics_data) >= 3:
                # Define optimal ranges and normalize values to 0-10 scale
                metric_configs = {
                    'front_elbow_angle': {'optimal_range': (90, 130), 'current': metrics_data.get('front_elbow_angle', 0)},
                    'spine_lean': {'optimal_range': (10, 25), 'current': metrics_data.get('spine_lean', 0)},
                    'head_knee_alignment': {'optimal_max': 50, 'current': metrics_data.get('head_knee_alignment', 0)},
                    'front_foot_direction': {'optimal_range': (-15, 15), 'current': metrics_data.get('front_foot_direction', 0)}
                }
                
                # Calculate normalized scores (0-10)
                normalized_values = []
                categories = []
                
                for metric, config in metric_configs.items():
                    if metric in metrics_data:
                        current_value = config['current']
                        
                        if 'optimal_range' in config:
                            # For range-based metrics
                            min_val, max_val = config['optimal_range']
                            if min_val <= current_value <= max_val:
                                # Perfect score if in optimal range
                                score = 10
                            else:
                                # Penalty based on distance from optimal range
                                distance = min(abs(current_value - min_val), abs(current_value - max_val))
                                score = max(0, 10 - (distance / 10))  # Adjust penalty factor
                        else:
                            # For threshold-based metrics (lower is better)
                            threshold = config['optimal_max']
                            if current_value <= threshold:
                                score = 10 - (current_value / threshold) * 2  # Scale to 8-10 range
                            else:
                                score = max(0, 8 - ((current_value - threshold) / threshold) * 8)
                        
                        normalized_values.append(score)
                        # Format category names for display
                        category_name = metric.replace('_', ' ').title()
                        if metric == 'head_knee_alignment':
                            category_name = 'Head Alignment'
                        elif metric == 'front_foot_direction':
                            category_name = 'Foot Direction'
                        elif metric == 'front_elbow_angle':
                            category_name = 'Elbow Angle'
                        categories.append(category_name)
                
                if len(normalized_values) >= 3:
                    # Create radar chart
                    fig = go.Figure()
                    
                    # Add current performance
                    fig.add_trace(go.Scatterpolar(
                        r=normalized_values,
                        theta=categories,
                        fill='toself',
                        name='Current Performance',
                        line=dict(color='#1f77b4'),
                        fillcolor='rgba(31, 119, 180, 0.3)'
                    ))
                    
                    # Add optimal performance reference (all 10s)
                    fig.add_trace(go.Scatterpolar(
                        r=[10] * len(categories),
                        theta=categories,
                        fill='toself',
                        name='Optimal Performance',
                        line=dict(color='#2ca02c', dash='dash'),
                        fillcolor='rgba(44, 160, 44, 0.1)'
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 10],
                                tickvals=[2, 4, 6, 8, 10],
                                ticktext=['2', '4', '6', '8', '10']
                            ),
                            angularaxis=dict(
                                tickfont=dict(size=10)
                            )
                        ),
                        showlegend=True,
                        title="Biomechanical Profile (Normalized 0-10 Scale)",
                        height=400,
                        font=dict(size=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 3 metrics for radar chart visualization")
            else:
                st.info("Insufficient biomechanical data for radar chart")
    
    # Download section
    st.subheader("Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if os.path.exists("output/annotated_video.mp4"):
            with open("output/annotated_video.mp4", "rb") as file:
                st.download_button(
                    label="📹 Download Annotated Video",
                    data=file.read(),
                    file_name="annotated_cricket_analysis.mp4",
                    mime="video/mp4"
                )
    
    with col2:
        if os.path.exists("output/evaluation.json"):
            with open("output/evaluation.json", "r") as file:
                st.download_button(
                    label="📊 Download JSON Report",
                    data=file.read(),
                    file_name="cricket_analysis_report.json",
                    mime="application/json"
                )
    
    with col3:
        if os.path.exists("output/evaluation.txt"):
            with open("output/evaluation.txt", "r") as file:
                st.download_button(
                    label="📄 Download Text Report",
                    data=file.read(),
                    file_name="cricket_analysis_report.txt",
                    mime="text/plain"
                )
    
    # Analysis summary
    st.subheader("Analysis Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Frames Analyzed", evaluation.get('total_frames_analyzed', 0))
    
    with col2:
        st.metric("Analysis Date", evaluation.get('timestamp', 'N/A'))
    
    with col3:
        if st.button("🔄 New Analysis"):
            st.session_state.analysis_complete = False
            st.session_state.evaluation_data = None
            st.rerun()

if __name__ == "__main__":
    main()
