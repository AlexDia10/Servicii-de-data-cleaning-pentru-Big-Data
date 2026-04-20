"""
Interactive Dashboard for Anomaly Detection System

Streamlit-based GUI providing:
- Data upload and exploration
- Algorithm selection and configuration
- Real-time anomaly detection
- Performance visualization and benchmarking
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import time
from typing import Optional, Tuple

from data_loader import DataLoader
from anomaly_algorithms import AnomalyDetectionEngine, AnomalyResult
from spark_engine import SparkEngine
from benchmark import Benchmark, BenchmarkMetrics
from utils import setup_logger


logger = setup_logger(__name__)


# Page configuration
st.set_page_config(
    page_title="Anomaly Detection System",
    page_icon="chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success {
        color: #28a745;
        font-weight: bold;
    }
    .danger {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = {}
    if 'benchmark_results' not in st.session_state:
        st.session_state.benchmark_results = None


def sidebar_controls() -> Tuple[str, dict, int, int]:
    """
    Sidebar controls for data selection and algorithm configuration.
    
    Returns:
        Tuple (data_source, algorithm_params, n_cores, window_size)
    """
    st.sidebar.title("Configuration")
    st.sidebar.divider()
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Select data source:",
        options=["Upload CSV", "NAB Dataset"],
        index=0
    )
    
    # Algorithm selection
    st.sidebar.subheader("Algorithm Settings")
    algorithm = st.sidebar.selectbox(
        "Select algorithm:",
        options=[
            "Rolling Stats Z-Score",
            "Prediction Error",
            "Hybrid Anomaly Score",
            "Ensemble (Vote)"
        ]
    )
    
    # Threshold settings
    if algorithm == "Rolling Stats Z-Score":
        threshold = st.sidebar.slider(
            "Z-Score Threshold:",
            min_value=1.0,
            max_value=4.0,
            value=2.5,
            step=0.1
        )
        window_size = st.sidebar.slider(
            "Rolling Window Size:",
            min_value=5,
            max_value=50,
            value=20,
            step=1
        )
        algorithm_params = {
            'threshold': threshold,
            'algorithm': 'rolling_stats',
            'window_size': window_size
        }
    
    elif algorithm == "Prediction Error":
        threshold = st.sidebar.slider(
            "Error Threshold (σ multiplier):",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.1
        )
        forecast_window = st.sidebar.slider(
            "Forecast Window:",
            min_value=5,
            max_value=30,
            value=10,
            step=1
        )
        algorithm_params = {
            'threshold': threshold,
            'algorithm': 'prediction_error',
            'forecast_window': forecast_window
        }
    
    elif algorithm == "Hybrid Anomaly Score":
        zscore_thresh = st.sidebar.slider(
            "Z-Score Threshold:",
            min_value=1.0,
            max_value=4.0,
            value=2.0,
            step=0.1
        )
        
        st.sidebar.markdown("**Weights Distribution:**")
        col1, col2, col3 = st.sidebar.columns(3)
        
        with col1:
            w_zscore = st.slider(
                "Z-Score",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="w_zscore"
            )
        
        with col2:
            w_trend = st.slider(
                "Trend",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key="w_trend"
            )
        
        with col3:
            w_volatility = st.slider(
                "Volatility",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key="w_volatility"
            )
        
        # Normalize weights to sum = 1
        total_weight = w_zscore + w_trend + w_volatility
        if total_weight > 0:
            weights = (
                w_zscore / total_weight,
                w_trend / total_weight,
                w_volatility / total_weight
            )
        else:
            weights = (0.5, 0.3, 0.2)
        
        st.sidebar.info(f"Normalized: Z={weights[0]:.2f}, Trend={weights[1]:.2f}, Vol={weights[2]:.2f}")
        
        algorithm_params = {
            'zscore_threshold': zscore_thresh,
            'algorithm': 'hybrid',
            'weights': weights
        }
    
    else:  # Ensemble
        algorithm_params = {
            'algorithm': 'ensemble'
        }
    
    # Spark settings
    st.sidebar.subheader("Distributed Processing")
    use_spark = st.sidebar.checkbox("Use Spark", value=False)
    n_cores = st.sidebar.select_slider(
        "Number of Cores:",
        options=[1, 2, 4, 8],
        value=4 if use_spark else 1
    )
    
    algorithm_params['use_spark'] = use_spark
    algorithm_params['n_cores'] = n_cores
    
    # Normalization
    st.sidebar.subheader("Normalization")
    normalize = st.sidebar.checkbox("Normalize Data", value=True)
    normalization_method = st.sidebar.selectbox(
        "Normalization Method:",
        options=["standard", "minmax"],
        disabled=not normalize
    )
    
    algorithm_params['normalize'] = normalize
    algorithm_params['normalization_method'] = normalization_method
    
    return data_source, algorithm_params, n_cores, algorithm_params.get(
        'window_size', 20
    )


def plot_timeseries(
    df: pd.DataFrame,
    anomaly_labels: np.ndarray,
    anomaly_scores: np.ndarray
) -> go.Figure:
    """
    Create interactive time series plot with anomaly markers.
    
    Args:
        df: DataFrame with timestamp and value columns
        anomaly_labels: Binary anomaly labels
        anomaly_scores: Anomaly scores [0, 1]
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Time series line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['value'],
        mode='lines',
        name='Time Series',
        line=dict(color='blue', width=2),
        hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
    ))
    
    # Anomaly points
    anomaly_indices = np.where(anomaly_labels == 1)[0]
    if len(anomaly_indices) > 0:
        fig.add_trace(go.Scatter(
            x=df.iloc[anomaly_indices]['timestamp'],
            y=df.iloc[anomaly_indices]['value'],
            mode='markers',
            name='Anomalies',
            marker=dict(
                size=8,
                color='red',
                symbol='x',
                line=dict(width=2)
            ),
            hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
        ))
    
    # Anomaly background shading
    for i in anomaly_indices:
        fig.add_vrect(
            x0=df.iloc[i]['timestamp'],
            x1=df.iloc[min(i+1, len(df)-1)]['timestamp'],
            fillcolor="red",
            opacity=0.1,
            layer="below",
            line_width=0
        )
    
    fig.update_layout(
        title="Time Series with Detected Anomalies",
        xaxis_title="Timestamp",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified",
        height=500
    )
    
    return fig


def plot_histogram(df: pd.DataFrame) -> go.Figure:
    """Create histogram of values."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['value'],
        nbinsx=30,
        name='Values',
        marker=dict(color='skyblue', line=dict(color='navy', width=1))
    ))
    
    fig.update_layout(
        title="Distribution of Values",
        xaxis_title="Value",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_anomaly_scores(
    df: pd.DataFrame,
    anomaly_scores: np.ndarray
) -> go.Figure:
    """Create anomaly score visualization."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=anomaly_scores,
        mode='lines',
        name='Anomaly Score',
        fill='tozeroy',
        line=dict(color='orange'),
        hovertemplate='<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>'
    ))
    
    # Add threshold line
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Threshold"
    )
    
    fig.update_layout(
        title="Anomaly Scores Over Time",
        xaxis_title="Timestamp",
        yaxis_title="Anomaly Score",
        template="plotly_white",
        height=400
    )
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    algorithm_name: str
) -> go.Figure:
    """Create confusion matrix heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Normal', 'Anomaly'],
        y=['Normal', 'Anomaly'],
        text=cm,
        texttemplate='%{text}',
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title=f"Confusion Matrix - {algorithm_name}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400
    )
    
    return fig


def main():
    """Main dashboard application."""
    initialize_session_state()
    
    # Header
    st.title("Advanced Anomaly Detection System")
    st.markdown(
        "**Scalable time-series anomaly detection with multiple algorithms and "
        "Spark integration**"
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Size", "---", "rows")
    with col2:
        st.metric("Anomalies Found", "---", "detections")
    with col3:
        st.metric("Execution Time", "---", "seconds")
    
    st.divider()
    
    # Sidebar controls
    data_source, algorithm_params, n_cores, window_size = sidebar_controls()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data",
        "Detection",
        "Results",
        "Benchmark"
    ])
    
    # TAB 1: DATA UPLOAD AND EXPLORATION
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload or Select Data")
            
            if data_source == "Upload CSV":
                uploaded_file = st.file_uploader(
                    "Upload CSV file (timestamp, value columns):",
                    type=['csv']
                )
                
                if uploaded_file:
                    try:
                        df = pd.read_csv(uploaded_file)
                        st.success(f"✓ Loaded {len(df)} rows")
                        st.session_state.data = df
                        st.session_state.data_loaded = True
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
            
            else:  # NAB Dataset
                loader = DataLoader()
                datasets = loader.list_nab_datasets()
                
                selected_dataset = st.selectbox(
                    "Select NAB dataset:",
                    options=datasets,
                    index=0 if datasets else None
                )
                
                if selected_dataset and st.button("Load Dataset", use_container_width=True):
                    try:
                        with st.spinner("Loading..."):
                            df = loader.preprocess(selected_dataset)
                            st.session_state.data = df
                            st.session_state.data_loaded = True
                            st.session_state.current_dataset = selected_dataset
                            st.success(f"Loaded {len(df)} rows from {selected_dataset}")
                    except Exception as e:
                        st.error(f"Error loading dataset: {e}")
        
        with col2:
            if st.session_state.data_loaded:
                st.subheader("Dataset Statistics")
                df = st.session_state.data
                
                stats_col1, stats_col2 = st.columns(2)
                with stats_col1:
                    st.metric("Rows", len(df))
                    st.metric("Mean", f"{df['value'].mean():.2f}")
                    st.metric("Min", f"{df['value'].min():.2f}")
                
                with stats_col2:
                    st.metric("Columns", len(df.columns))
                    st.metric("Std Dev", f"{df['value'].std():.2f}")
                    st.metric("Max", f"{df['value'].max():.2f}")
        
        if st.session_state.data_loaded:
            st.divider()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Data Preview")
                st.dataframe(
                    st.session_state.data.head(10),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("Distribution")
                fig_hist = plot_histogram(st.session_state.data)
                st.plotly_chart(fig_hist, use_container_width=True)
    
    # TAB 2: ANOMALY DETECTION
    with tab2:
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload or select data first")
        else:
            st.subheader("Run Anomaly Detection")
            
            # Algorithm info
            algo_name = algorithm_params['algorithm']
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info(
                    f"**Algorithm:** {algorithm_params.get('algorithm', 'unknown')} | "
                    f"**Cores:** {algorithm_params['n_cores']} | "
                    f"**Normalize:** {algorithm_params['normalize']}"
                )
            
            with col2:
                if st.button("Run Detection", use_container_width=True):
                    with st.spinner("Processing..."):
                        try:
                            df = st.session_state.data
                            values = df['value'].values
                            
                            # Run algorithm
                            start_time = time.time()
                            
                            engine = AnomalyDetectionEngine()
                            
                            if algo_name == 'ensemble':
                                result = engine.ensemble_vote(values)
                            else:
                                result = engine.detect_single(
                                    algo_name,
                                    values,
                                    **{k: v for k, v in algorithm_params.items()
                                       if k not in ['algorithm', 'use_spark', 'n_cores',
                                                   'normalize', 'normalization_method']}
                                )
                            
                            exec_time = time.time() - start_time
                            
                            # Store results
                            st.session_state.detection_results = {
                                'labels': result.labels,
                                'scores': result.scores,
                                'algorithm': result.algorithm,
                                'exec_time': exec_time
                            }
                            
                            st.success(
                                f"✓ Detection completed in {exec_time:.3f}s"
                            )
                            
                        except Exception as e:
                            st.error(f"Error: {e}")
                            logger.exception("Error in detection")
            
            # Display results
            if st.session_state.detection_results:
                st.divider()
                
                results = st.session_state.detection_results
                df = st.session_state.data
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    n_anomalies = int(results['labels'].sum())
                    st.metric("Anomalies Detected", n_anomalies)
                with col2:
                    pct_anomalies = 100 * n_anomalies / len(df)
                    st.metric("% of Data", f"{pct_anomalies:.2f}%")
                with col3:
                    st.metric("Exec Time", f"{results['exec_time']:.3f}s")
                
                st.divider()
                
                # Plots
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig_ts = plot_timeseries(
                        df,
                        results['labels'],
                        results['scores']
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)
                
                with col2:
                    fig_scores = plot_anomaly_scores(df, results['scores'])
                    st.plotly_chart(fig_scores, use_container_width=True)
    
    # TAB 3: DETAILED RESULTS
    with tab3:
        if not st.session_state.detection_results:
            st.warning("⚠️ Run detection first")
        else:
            results = st.session_state.detection_results
            
            # Anomaly details table
            st.subheader("Anomalies Detected")
            
            anomaly_indices = np.where(results['labels'] == 1)[0]
            
            if len(anomaly_indices) > 0:
                anomaly_data = []
                df = st.session_state.data
                
                for idx in anomaly_indices[:100]:  # Show first 100
                    anomaly_data.append({
                        'Index': idx,
                        'Timestamp': df.iloc[idx]['timestamp'],
                        'Value': f"{df.iloc[idx]['value']:.2f}",
                        'Score': f"{results['scores'][idx]:.3f}"
                    })
                
                anomaly_df = pd.DataFrame(anomaly_data)
                st.dataframe(anomaly_df, use_container_width=True)
            else:
                st.info("No anomalies detected")
    
    # TAB 4: BENCHMARKING
    with tab4:
        st.subheader("Performance Benchmarking")
        
        st.markdown("""
        **Evaluare cu intervale NAB:**
        - **TP (True Positive)**: Interval care contine cel putin 1 timestamp detectat
        - **FN (False Negative)**: Interval care NU contine niciun timestamp detectat
        - **FP (False Positive)**: Timestamp detectat care NU se afla in niciun interval
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Run Benchmark", use_container_width=True):
                with st.spinner("Benchmarking..."):
                    try:
                        if not st.session_state.data_loaded:
                            st.error("Load data first")
                        else:
                            df = st.session_state.data
                            values = df['value'].values
                            
                            # Load NAB intervals if available
                            try:
                                from nab_integration import NABIntegration
                                nab = NABIntegration()
                                
                                # Extract dataset name from current data
                                dataset_name = st.session_state.get('current_dataset', None)
                                
                                if dataset_name:
                                    nab_intervals = nab.get_anomaly_intervals(dataset_name, df)
                                    
                                    if nab_intervals:
                                        # Run detection
                                        engine = AnomalyDetectionEngine()
                                        result = engine.detect_single(
                                            algorithm_params['algorithm'],
                                            values,
                                            **{k: v for k, v in algorithm_params.items()
                                               if k not in ['algorithm', 'use_spark', 'n_cores',
                                                           'normalize', 'normalization_method']}
                                        )
                                        
                                        start_time = time.time()
                                        # Re-run for timing
                                        if algorithm_params['algorithm'] == 'ensemble':
                                            result = engine.ensemble_vote(values)
                                        exec_time = time.time() - start_time
                                        
                                        # Get detected anomalies
                                        anomaly_indices = np.where(result.labels == 1)[0]
                                        
                                        # Evaluate with NAB intervals
                                        benchmark = Benchmark()
                                        metrics = benchmark.evaluate_with_nab_intervals(
                                            anomaly_indices,
                                            nab_intervals,
                                            len(values),
                                            exec_time
                                        )
                                        
                                        st.session_state.benchmark_results = metrics
                                        st.success(
                                            f"Benchmark completed: {metrics.detected_anomalies} TP, "
                                            f"{metrics.missed_anomalies} FN, {metrics.false_alarms} FP | "
                                            f"Precision={metrics.precision:.3f}, Recall={metrics.recall:.3f}, "
                                            f"F1={metrics.f1_score:.3f}"
                                        )
                                    else:
                                        st.warning("No NAB intervals found for this dataset")
                                else:
                                    st.warning("Use NAB dataset for benchmarking")
                            except Exception as e:
                                st.warning(f"NAB benchmark not available: {e}")
                                # Fallback to simple benchmark
                                st.info("Running simple performance test...")
                                
                    except Exception as e:
                        st.error(f"Error: {e}")
                        import traceback
                        st.error(traceback.format_exc())
        
        # Display results
        if st.session_state.benchmark_results:
            metrics = st.session_state.benchmark_results
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("True Positives (TP)", f"{metrics.detected_anomalies}/{metrics.total_anomalies}")
            with col2:
                st.metric("False Negatives (FN)", metrics.missed_anomalies)
            with col3:
                st.metric("False Positives (FP)", metrics.false_alarms)
            with col4:
                st.metric("NAB Score", f"{metrics.nab_score:.1f}")
            
            st.divider()
            
            # Summary table
            summary_data = {
                'Metric': ['True Positives (TP)', 'False Negatives (FN)', 'False Positives (FP)', 'Recall', 'Precision', 'F1-Score', 'Execution Time'],
                'Value': [
                    f"{metrics.detected_anomalies}/{metrics.total_anomalies}",
                    metrics.missed_anomalies,
                    metrics.false_alarms,
                    f"{metrics.recall:.3f}",
                    f"{metrics.precision:.3f}",
                    f"{metrics.f1_score:.3f}",
                    f"{metrics.execution_time:.3f}s"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)


if __name__ == "__main__":
    main()

