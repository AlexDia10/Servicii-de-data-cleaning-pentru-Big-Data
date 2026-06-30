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
import hashlib
import json
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
    if 'gui_mode' not in st.session_state:
        st.session_state.gui_mode = 'simple'
    if 'presets_loaded' not in st.session_state:
        st.session_state.presets_loaded = False


def load_presets() -> dict:
    """Load algorithm presets from presets.json."""
    import json
    from pathlib import Path
    
    presets_file = Path(__file__).parent / 'presets.json'
    try:
        with open(presets_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load presets: {e}")
        # Fallback presets if file not found
        return {
            'algorithms': {
                'rolling_stats': {
                    'name': 'Rolling Statistics Z-Score',
                    'simple_mode': {'window_size': 20, 'threshold': 4.0}
                },
                'prediction_error': {
                    'name': 'Prediction Error Anomaly',
                    'simple_mode': {'forecast_window': 5, 'threshold': 4.5}
                },
                'hybrid': {
                    'name': 'Hybrid Score (RECOMMENDED)',
                    'simple_mode': {
                        'zscore_threshold': 3.5,
                        'trend_threshold': 1.2,
                        'volatility_threshold': 3.5,
                        'weights': [0.5, 0.3, 0.2],
                        'label_threshold': 0.80
                    }
                }
            }
        }


def sidebar_controls() -> Tuple[str, dict, int, int]:
    """
    Sidebar controls for data selection and algorithm configuration.
    Supports Simple Mode (recommended presets) and Advanced Mode (expert tuning).
    
    Returns:
        Tuple (data_source, algorithm_params, n_cores, window_size)
    """
    st.sidebar.title("Configuration")
    st.sidebar.divider()
    
    # LOAD PRESETS
    if not st.session_state.presets_loaded:
        st.session_state.presets = load_presets()
        st.session_state.presets_loaded = True
    presets = st.session_state.presets
    
    # MODE SELECTOR (Simple vs Advanced)
    st.sidebar.subheader("Mode Selection")
    mode_col1, mode_col2 = st.sidebar.columns(2)
    
    with mode_col1:
        if st.button("Simple", use_container_width=True, 
                    key="btn_simple",
                    type="primary" if st.session_state.gui_mode == 'simple' else "secondary"):
            st.session_state.gui_mode = 'simple'
            st.rerun()
    
    with mode_col2:
        if st.button("Advanced", use_container_width=True,
                    key="btn_advanced",
                    type="primary" if st.session_state.gui_mode == 'advanced' else "secondary"):
            st.session_state.gui_mode = 'advanced'
            st.rerun()
    
    if st.session_state.gui_mode == 'simple':
        st.sidebar.divider()
        # DETECTION SENSITIVITY (Simple Mode Only)
        st.sidebar.subheader("Detection Sensitivity")
        sensitivity_options = ["Low (fewer false alarms)", "Medium (balanced)", "High (detect more)"]
        sensitivity = st.sidebar.radio(
            "Choose sensitivity level:",
            options=sensitivity_options,
            index=1  # Medium is default
        )

        sensitivity_k_map = {
            "Low (fewer false alarms)": 4.0,
            "Medium (balanced)": 3.0,
            "High (detect more)": 2.0
        }
        k_std_value = sensitivity_k_map[sensitivity]

        st.sidebar.markdown(
            f"**Threshold**: mean + **{k_std_value}σ** (adaptive)"
        )
    
    st.sidebar.divider()
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    _ds_options = ["Upload CSV", "NAB Dataset"]
    _ds_default = _ds_options.index(
        st.session_state.get('_data_source_choice', "Upload CSV")
    )
    data_source = st.sidebar.radio(
        "Select data source:",
        options=_ds_options,
        index=_ds_default,
        key="data_source_radio"
    )
    st.session_state['_data_source_choice'] = data_source
    
    # Algorithm selection
    st.sidebar.subheader("Algorithm")
    
    if st.session_state.gui_mode == 'simple':
        algorithm_options = [
            "Rolling Stats Z-Score",
            "Prediction Error",
            "Hybrid Anomaly Score",
            "Mean Shift Detector",
            "Isolation Forest"
        ]
        algorithm = st.sidebar.selectbox(
            "Select algorithm:",
            options=algorithm_options,
            index=0
        )

        # Normalize algorithm name to internal name
        if "Hybrid" in algorithm:
            algorithm = "Hybrid Anomaly Score"
        elif "Rolling" in algorithm:
            algorithm = "Rolling Stats Z-Score"
        elif "Isolation" in algorithm:
            algorithm = "Isolation Forest"
        elif "Mean Shift" in algorithm:
            algorithm = "Mean Shift Detector"
        else:
            algorithm = "Prediction Error"
        
        # SIMPLE MODE: Apply preset parameters (read-only display)
        st.sidebar.markdown("**Using Recommended Preset Parameters:**")
        
        if algorithm == "Rolling Stats Z-Score":
            preset = presets['algorithms']['rolling_stats']['simple_mode']
            st.sidebar.markdown(
                f"- Window Size: **{preset['window_size']}**\n"
                f"- Threshold: **mean + {k_std_value}σ** (adaptive)"
            )
            algorithm_params = {
                'threshold_method': 'adaptive_std',
                'k_std': k_std_value,
                'algorithm': 'rolling_stats',
                'window_size': preset['window_size']
            }

        elif algorithm == "Prediction Error":
            preset = presets['algorithms']['prediction_error']['simple_mode']
            st.sidebar.markdown(
                f"- Forecast Window: **{preset['forecast_window']}**\n"
                f"- Threshold: **mean + {k_std_value}σ** (adaptive)"
            )
            algorithm_params = {
                'threshold_method': 'adaptive_std',
                'k_std': k_std_value,
                'algorithm': 'prediction_error',
                'forecast_window': preset['forecast_window']
            }

        elif algorithm == "Isolation Forest":
            preset = presets['algorithms']['isolation_forest']['simple_mode']
            contamination_map = {
                "Low (fewer false alarms)": 0.002,
                "Medium (balanced)": 0.005,
                "High (detect more)": 0.05
            }
            contamination_value = contamination_map[sensitivity]
            st.sidebar.markdown(
                f"- Contamination: **{contamination_value}** ({sensitivity})\n"
                f"- Trees: **{preset['n_estimators']}**\n"
                f"- Window: **{preset['window_size']}**"
            )
            algorithm_params = {
                'contamination': contamination_value,
                'n_estimators': preset['n_estimators'],
                'window_size': preset['window_size'],
                'algorithm': 'isolation_forest'
            }

        elif algorithm == "Mean Shift Detector":
            preset = presets['algorithms']['mean_shift']['simple_mode']
            st.sidebar.markdown(
                f"- Window: **{preset['window']}** puncte\n"
                f"- Threshold: **mean + {k_std_value}σ** (adaptive)"
            )
            algorithm_params = {
                'threshold_method': 'adaptive_std',
                'k_std': k_std_value,
                'window': preset['window'],
                'algorithm': 'mean_shift'
            }

        else:  # Hybrid (recommended)
            preset = presets['algorithms']['hybrid']['simple_mode']
            st.sidebar.markdown(
                f"- Threshold: **mean + {k_std_value}σ** (adaptive)\n"
                f"- Weights: **{preset['weights']}**"
            )
            algorithm_params = {
                'threshold_method': 'adaptive_std',
                'k_std': k_std_value,
                'weights': preset['weights'],
                'algorithm': 'hybrid'
            }
    
    else:
        # ADVANCED MODE: Full parameter control
        algorithm = st.sidebar.selectbox(
            "Select algorithm:",
            options=[
                "Rolling Stats Z-Score",
                "Prediction Error",
                "Hybrid Anomaly Score",
                "Mean Shift Detector",
                "Isolation Forest"
            ]
        )

        # Normalize algorithm name
        if "Rolling" in algorithm:
            algorithm = "Rolling Stats Z-Score"
        elif "Prediction" in algorithm:
            algorithm = "Prediction Error"
        elif "Isolation" in algorithm:
            algorithm = "Isolation Forest"
        elif "Mean Shift" in algorithm:
            algorithm = "Mean Shift Detector"
        else:
            algorithm = "Hybrid Anomaly Score"
        
        # THRESHOLD METHOD SELECTION (Advanced Mode)
        # Isolation Forest nu foloseste prag — pragul e determinat intern prin contamination
        threshold_method_value = 'percentile'
        if algorithm != "Isolation Forest":
            st.sidebar.subheader("Threshold Configuration")
            threshold_method = st.sidebar.radio(
                "Threshold Method:",
                options=["Fixed threshold", "Adaptive threshold"],
                index=1  # Adaptive is default
            )

            threshold_method_value = 'fixed' if 'Fixed' in threshold_method else 'adaptive_std'

            if 'Adaptive' in threshold_method:
                threshold_type = st.sidebar.selectbox(
                    "Adaptive Method:",
                    options=["Percentile", "Mean + k * Std"],
                    index=0
                )
                threshold_method_value = 'percentile' if threshold_type == "Percentile" else 'adaptive_std'
        
        # ADVANCED MODE: Manual parameter control
        if algorithm == "Rolling Stats Z-Score":
            advanced = presets['algorithms']['rolling_stats']['advanced_mode']
            
            window_size = st.sidebar.slider(
                "Rolling Window Size:",
                min_value=advanced['window_size']['min'],
                max_value=advanced['window_size']['max'],
                value=advanced['window_size']['default'],
                step=advanced['window_size']['step']
            )
            
            if threshold_method_value == 'percentile':
                percentile = st.sidebar.slider(
                    "Percentile:",
                    min_value=advanced['percentile']['min'],
                    max_value=advanced['percentile']['max'],
                    value=advanced['percentile']['default'],
                    step=advanced['percentile']['step']
                )
                algorithm_params = {
                    'threshold_method': 'percentile',
                    'percentile': percentile,
                    'algorithm': 'rolling_stats',
                    'window_size': window_size
                }
            elif threshold_method_value == 'adaptive_std':
                k_std = st.sidebar.slider(
                    "K (Std Multiplier):",
                    min_value=advanced['k_std']['min'],
                    max_value=advanced['k_std']['max'],
                    value=advanced['k_std']['default'],
                    step=advanced['k_std']['step']
                )
                algorithm_params = {
                    'threshold_method': 'adaptive_std',
                    'k_std': k_std,
                    'algorithm': 'rolling_stats',
                    'window_size': window_size
                }
            else:  # fixed
                threshold = st.sidebar.slider(
                    "Fixed Threshold:",
                    min_value=advanced['fixed_threshold']['min'],
                    max_value=advanced['fixed_threshold']['max'],
                    value=advanced['fixed_threshold']['default'],
                    step=advanced['fixed_threshold']['step']
                )
                algorithm_params = {
                    'threshold_method': 'fixed',
                    'threshold': threshold,
                    'algorithm': 'rolling_stats',
                    'window_size': window_size
                }
        
        elif algorithm == "Prediction Error":
            advanced = presets['algorithms']['prediction_error']['advanced_mode']
            
            forecast_window = st.sidebar.slider(
                "Forecast Window:",
                min_value=advanced['forecast_window']['min'],
                max_value=advanced['forecast_window']['max'],
                value=advanced['forecast_window']['default'],
                step=advanced['forecast_window']['step']
            )
            
            if threshold_method_value == 'percentile':
                percentile = st.sidebar.slider(
                    "Percentile:",
                    min_value=advanced['percentile']['min'],
                    max_value=advanced['percentile']['max'],
                    value=advanced['percentile']['default'],
                    step=advanced['percentile']['step']
                )
                algorithm_params = {
                    'threshold_method': 'percentile',
                    'percentile': percentile,
                    'algorithm': 'prediction_error',
                    'forecast_window': forecast_window
                }
            elif threshold_method_value == 'adaptive_std':
                k_std = st.sidebar.slider(
                    "K (Std Multiplier):",
                    min_value=advanced['k_std']['min'],
                    max_value=advanced['k_std']['max'],
                    value=advanced['k_std']['default'],
                    step=advanced['k_std']['step']
                )
                algorithm_params = {
                    'threshold_method': 'adaptive_std',
                    'k_std': k_std,
                    'algorithm': 'prediction_error',
                    'forecast_window': forecast_window
                }
            else:  # fixed
                threshold = st.sidebar.slider(
                    "Fixed Threshold:",
                    min_value=advanced['fixed_threshold']['min'],
                    max_value=advanced['fixed_threshold']['max'],
                    value=advanced['fixed_threshold']['default'],
                    step=advanced['fixed_threshold']['step']
                )
                algorithm_params = {
                    'threshold_method': 'fixed',
                    'threshold': threshold,
                    'algorithm': 'prediction_error',
                    'forecast_window': forecast_window
                }
        
        elif algorithm == "Hybrid Anomaly Score":
            advanced = presets['algorithms']['hybrid']['advanced_mode']
            
            # Weights are always configurable in Advanced Mode
            st.sidebar.markdown("**Weights Distribution:**")
            col1, col2 = st.sidebar.columns(2)

            with col1:
                w_zscore = st.slider(
                    "Z-Score",
                    min_value=advanced['weights']['zscore_weight']['min'],
                    max_value=advanced['weights']['zscore_weight']['max'],
                    value=advanced['weights']['zscore_weight']['default'],
                    step=advanced['weights']['zscore_weight']['step'],
                    key="adv_w_zscore"
                )
                w_trend = st.slider(
                    "Trend",
                    min_value=advanced['weights']['trend_weight']['min'],
                    max_value=advanced['weights']['trend_weight']['max'],
                    value=advanced['weights']['trend_weight']['default'],
                    step=advanced['weights']['trend_weight']['step'],
                    key="adv_w_trend"
                )

            with col2:
                w_volatility = st.slider(
                    "Volatility",
                    min_value=advanced['weights']['volatility_weight']['min'],
                    max_value=advanced['weights']['volatility_weight']['max'],
                    value=advanced['weights']['volatility_weight']['default'],
                    step=advanced['weights']['volatility_weight']['step'],
                    key="adv_w_volatility"
                )

            # Normalize weights so they always sum to 1
            total_weight = w_zscore + w_trend + w_volatility
            if total_weight > 0:
                weights = (
                    w_zscore     / total_weight,
                    w_trend      / total_weight,
                    w_volatility / total_weight,
                )
            else:
                weights = (0.5, 0.3, 0.2)
            
            if threshold_method_value == 'percentile':
                percentile = st.sidebar.slider(
                    "Percentile:",
                    min_value=advanced['percentile']['min'],
                    max_value=advanced['percentile']['max'],
                    value=advanced['percentile']['default'],
                    step=advanced['percentile']['step']
                )
                algorithm_params = {
                    'threshold_method': 'percentile',
                    'percentile': percentile,
                    'weights': weights,
                    'algorithm': 'hybrid'
                }
            elif threshold_method_value == 'adaptive_std':
                k_std = st.sidebar.slider(
                    "K (Std Multiplier):",
                    min_value=advanced['k_std']['min'],
                    max_value=advanced['k_std']['max'],
                    value=advanced['k_std']['default'],
                    step=advanced['k_std']['step']
                )
                algorithm_params = {
                    'threshold_method': 'adaptive_std',
                    'k_std': k_std,
                    'weights': weights,
                    'algorithm': 'hybrid'
                }
            else:  # fixed
                zscore_thresh = st.sidebar.slider(
                    "Z-Score Threshold:",
                    min_value=advanced['zscore_threshold']['min'],
                    max_value=advanced['zscore_threshold']['max'],
                    value=advanced['zscore_threshold']['default'],
                    step=advanced['zscore_threshold']['step'],
                    key="adv_zscore"
                )
                
                trend_thresh = st.sidebar.slider(
                    "Trend Threshold:",
                    min_value=advanced['trend_threshold']['min'],
                    max_value=advanced['trend_threshold']['max'],
                    value=advanced['trend_threshold']['default'],
                    step=advanced['trend_threshold']['step'],
                    key="adv_trend"
                )
                
                volatility_thresh = st.sidebar.slider(
                    "Volatility Threshold:",
                    min_value=advanced['volatility_threshold']['min'],
                    max_value=advanced['volatility_threshold']['max'],
                    value=advanced['volatility_threshold']['default'],
                    step=advanced['volatility_threshold']['step'],
                    key="adv_volatility"
                )
                
                algorithm_params = {
                    'threshold_method': 'fixed',
                    'zscore_threshold': zscore_thresh,
                    'trend_threshold': trend_thresh,
                    'volatility_threshold': volatility_thresh,
                    'weights': weights,
                    'algorithm': 'hybrid'
                }
            
            st.sidebar.info(f"Weights (normalized): Z-Score={weights[0]:.2f}, Trend={weights[1]:.2f}, Volatility={weights[2]:.2f}")

        elif algorithm == "Isolation Forest":
            advanced = presets['algorithms']['isolation_forest']['advanced_mode']

            contamination = st.sidebar.slider(
                "Contamination (expected anomaly fraction):",
                min_value=advanced['contamination']['min'],
                max_value=advanced['contamination']['max'],
                value=advanced['contamination']['default'],
                step=advanced['contamination']['step'],
                format="%.3f"
            )
            n_estimators = st.sidebar.slider(
                "Number of Trees:",
                min_value=advanced['n_estimators']['min'],
                max_value=advanced['n_estimators']['max'],
                value=advanced['n_estimators']['default'],
                step=advanced['n_estimators']['step']
            )
            window_size_if = st.sidebar.slider(
                "Rolling Window Size:",
                min_value=advanced['window_size']['min'],
                max_value=advanced['window_size']['max'],
                value=advanced['window_size']['default'],
                step=advanced['window_size']['step']
            )
            algorithm_params = {
                'contamination': contamination,
                'n_estimators': n_estimators,
                'window_size': window_size_if,
                'algorithm': 'isolation_forest'
            }

        elif algorithm == "Mean Shift Detector":
            advanced = presets['algorithms']['mean_shift']['advanced_mode']

            ms_window = st.sidebar.slider(
                "Window Size (half):",
                min_value=advanced['window']['min'],
                max_value=advanced['window']['max'],
                value=advanced['window']['default'],
                step=advanced['window']['step'],
                help="Fiecare punct este comparat cu window puncte inainte si window puncte dupa."
            )

            if threshold_method_value == 'percentile':
                percentile = st.sidebar.slider(
                    "Percentile:",
                    min_value=advanced['percentile']['min'],
                    max_value=advanced['percentile']['max'],
                    value=advanced['percentile']['default'],
                    step=advanced['percentile']['step']
                )
                algorithm_params = {
                    'threshold_method': 'percentile',
                    'percentile': percentile,
                    'window': ms_window,
                    'algorithm': 'mean_shift'
                }
            elif threshold_method_value == 'adaptive_std':
                k_std = st.sidebar.slider(
                    "K (Sigma Multiplier):",
                    min_value=advanced['k_std']['min'],
                    max_value=advanced['k_std']['max'],
                    value=advanced['k_std']['default'],
                    step=advanced['k_std']['step']
                )
                algorithm_params = {
                    'threshold_method': 'adaptive_std',
                    'k_std': k_std,
                    'window': ms_window,
                    'algorithm': 'mean_shift'
                }
            else:  # fixed
                fixed_thr = st.sidebar.slider(
                    "Fixed Threshold (sigma units):",
                    min_value=advanced['fixed_threshold']['min'],
                    max_value=advanced['fixed_threshold']['max'],
                    value=advanced['fixed_threshold']['default'],
                    step=advanced['fixed_threshold']['step']
                )
                algorithm_params = {
                    'threshold_method': 'fixed',
                    'threshold': fixed_thr,
                    'window': ms_window,
                    'algorithm': 'mean_shift'
                }

    algorithm_params['use_spark'] = False
    algorithm_params['n_cores'] = 1

    # Pasează sensitivitatea in Simple Mode pentru afisare in bara de info
    if st.session_state.gui_mode == 'simple':
        algorithm_params['sensitivity'] = sensitivity

    # Normalization (only in Advanced Mode)
    if st.session_state.gui_mode == 'advanced':
        st.sidebar.subheader("Normalization")
        normalize = st.sidebar.checkbox("Normalize Data", value=False)
        normalization_method = st.sidebar.selectbox(
            "Normalization Method:",
            options=["standard", "minmax"],
            disabled=not normalize
        )
    else:
        # Simple mode: use raw data (no normalization)
        normalize = False
        normalization_method = None
    
    algorithm_params['normalize'] = normalize
    if normalization_method:
        algorithm_params['normalization_method'] = normalization_method
    
    return data_source, algorithm_params, 1, algorithm_params.get(
        'window_size', 20
    )


def downsample_data(df: pd.DataFrame, max_points: int = 2000) -> pd.DataFrame:
    """
    Min-max downsample: for each bucket keep both the min-value and max-value
    point so spikes and troughs are never silently dropped by uniform sampling.
    """
    if len(df) <= max_points:
        return df

    n_buckets = max_points // 2
    step = len(df) / n_buckets
    kept = set()
    for i in range(n_buckets):
        start = int(i * step)
        end   = int((i + 1) * step)
        if start >= len(df):
            break
        bucket = df.iloc[start:end]
        if len(bucket) == 0:
            continue
        kept.add(bucket['value'].idxmin())
        kept.add(bucket['value'].idxmax())

    return df.loc[sorted(kept)].reset_index(drop=True)


@st.cache_data
def plot_timeseries_cached(
    timestamps: tuple,
    values: tuple,
    anomaly_ts: tuple,
    anomaly_vals: tuple
) -> go.Figure:
    """Create time series plot with anomaly markers (CACHED - optimized)."""
    fig = go.Figure()

    # Time series line (thinner for performance)
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=values,
        mode='lines',
        name='Time Series',
        line=dict(color='blue', width=1),
        hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
    ))

    # Anomaly points — use pre-extracted timestamps/values from original df
    if len(anomaly_ts) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_ts,
            y=anomaly_vals,
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
    
    fig.update_layout(
        title="Time Series with Detected Anomalies",
        xaxis_title="Timestamp",
        yaxis_title="Value",
        template="plotly_white",
        hovermode="x unified",
        height=500
    )
    
    return fig


_ANOMALY_TYPE_STYLE = {
    'Spike':       {'color': '#e74c3c', 'symbol': 'x',           'size': 10},
    'Trend':       {'color': '#3498db', 'symbol': 'diamond',      'size': 10},
    'Volatility':  {'color': '#e67e22', 'symbol': 'star',         'size': 12},
    'Mean Shift':  {'color': '#9b59b6', 'symbol': 'triangle-up',  'size': 10},
    'Flatline':    {'color': '#7f8c8d', 'symbol': 'square',       'size': 10},
    'Level Shift': {'color': '#8e44ad', 'symbol': 'triangle-up',  'size': 10},
}


def plot_timeseries(
    df: pd.DataFrame,
    anomaly_labels: np.ndarray,
    detection_types: np.ndarray = None
) -> go.Figure:
    """Wrapper for plot_timeseries_cached with downsampling.

    When detection_types is provided, renders one scatter trace per anomaly
    type using distinct colours and marker symbols (bypasses cache).
    """
    anom_mask = anomaly_labels == 1
    anom_df = df[anom_mask]

    # Downsample for display performance, but always keep anomaly points in the
    # line so markers never appear disconnected from the series (floating in air).
    df_sampled = downsample_data(df, max_points=2000)
    if len(anom_df) > 0:
        df_display = (
            pd.concat([df_sampled, anom_df])
            .drop_duplicates('timestamp')
            .sort_values('timestamp')
            .reset_index(drop=True)
        )
    else:
        df_display = df_sampled

    # ── typed figure (not cached — one trace per anomaly type) ───────────
    if detection_types is not None and anom_mask.any():
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_display['timestamp'],
            y=df_display['value'],
            mode='lines',
            name='Time Series',
            line=dict(color='blue', width=1),
            hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
        ))
        anom_types = detection_types[anom_mask]
        for type_name, style in _ANOMALY_TYPE_STYLE.items():
            tmask = anom_types == type_name
            if not tmask.any():
                continue
            tdf = anom_df[tmask]
            fig.add_trace(go.Scatter(
                x=tdf['timestamp'],
                y=tdf['value'],
                mode='markers',
                name=type_name,
                marker=dict(
                    symbol=style['symbol'],
                    size=style['size'],
                    color=style['color'],
                    line=dict(width=1.5, color='rgba(0,0,0,0.4)')
                ),
                hovertemplate=(
                    f'<b>%{{x}}</b><br>Value: %{{y:.2f}}<br>'
                    f'<b>{type_name}</b><extra></extra>'
                )
            ))
        fig.update_layout(
            title='Time Series with Detected Anomalies',
            xaxis_title='Timestamp',
            yaxis_title='Value',
            template='plotly_white',
            hovermode='x unified',
            height=500,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        return fig

    # ── cached single-colour fallback ─────────────────────────────────────
    anomaly_ts   = tuple(anom_df['timestamp'].values)
    anomaly_vals = tuple(anom_df['value'].values)
    timestamps = tuple(df_display['timestamp'].values)
    values     = tuple(df_display['value'].values)
    return plot_timeseries_cached(timestamps, values, anomaly_ts, anomaly_vals)


def plot_benchmark_comparison(
    df: pd.DataFrame,
    nab_intervals: list,
    anomaly_indices: list,
    tolerance: int = 100
) -> go.Figure:
    """Time series with NAB ground truth bands and TP/FP coloured detections."""
    n = len(df)

    # Downsample for display but always keep detected anomaly points in the line
    # so markers never appear disconnected from the series (floating in air).
    df_sampled = downsample_data(df, max_points=2000)
    if anomaly_indices:
        anom_rows = df.iloc[[i for i in anomaly_indices if 0 <= i < n]]
        df_display = (
            pd.concat([df_sampled, anom_rows])
            .drop_duplicates('timestamp')
            .sort_values('timestamp')
            .reset_index(drop=True)
        )
    else:
        df_display = df_sampled

    fig = go.Figure()

    # ── time series line ──────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df_display['timestamp'],
        y=df_display['value'],
        mode='lines',
        name='Time Series',
        line=dict(color='#4a90d9', width=1),
        hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
    ))

    # ── NAB ground truth intervals (vrect bands) ──────────────────────
    tp_label_added = fp_label_added = False
    for start, end in nab_intervals:
        ts_start = df.iloc[max(0, start)]['timestamp']
        ts_end   = df.iloc[min(end, n - 1)]['timestamp']

        fuzzy_s = max(0, start - tolerance)
        fuzzy_e = min(n - 1, end + tolerance)
        hit = any(fuzzy_s <= idx <= fuzzy_e for idx in anomaly_indices)

        if hit:
            fill = 'rgba(0, 180, 0, 0.15)'
            line_c = 'rgba(0, 160, 0, 0.6)'
            label = 'TP – detected interval'
            show = not tp_label_added
            tp_label_added = True
        else:
            fill = 'rgba(255, 120, 0, 0.15)'
            line_c = 'rgba(220, 80, 0, 0.6)'
            label = 'FN – missed interval'
            show = not fp_label_added
            fp_label_added = True

        fig.add_vrect(
            x0=ts_start, x1=ts_end,
            fillcolor=fill,
            line=dict(color=line_c, width=1.5),
            annotation_text='TP' if hit else 'FN',
            annotation_position='top left',
            annotation_font_size=11,
            annotation_font_color=line_c,
            # legend entry only for first of each kind
            **(dict(name=label, showlegend=show, legendgroup=label) if show else {})
        )

    # ── detected anomalies coloured by TP / FP ───────────────────────
    tp_ts, tp_vals, fp_ts, fp_vals = [], [], [], []
    for idx in anomaly_indices:
        in_any = any(
            max(0, s - tolerance) <= idx <= min(n - 1, e + tolerance)
            for s, e in nab_intervals
        )
        ts  = df.iloc[idx]['timestamp']
        val = df.iloc[idx]['value']
        if in_any:
            tp_ts.append(ts); tp_vals.append(val)
        else:
            fp_ts.append(ts); fp_vals.append(val)

    if tp_ts:
        fig.add_trace(go.Scatter(
            x=tp_ts, y=tp_vals,
            mode='markers',
            name=f'TP detections ({len(tp_ts)})',
            marker=dict(color='green', size=9, symbol='circle',
                        line=dict(color='darkgreen', width=1)),
            hovertemplate='TP<br>%{x}<br>Value: %{y:.2f}<extra></extra>'
        ))

    if fp_ts:
        fig.add_trace(go.Scatter(
            x=fp_ts, y=fp_vals,
            mode='markers',
            name=f'FP detections ({len(fp_ts)})',
            marker=dict(color='red', size=9, symbol='x',
                        line=dict(width=2)),
            hovertemplate='FP<br>%{x}<br>Value: %{y:.2f}<extra></extra>'
        ))

    fig.update_layout(
        title='Detected Anomalies vs NAB Ground Truth',
        xaxis_title='Timestamp',
        yaxis_title='Value',
        template='plotly_white',
        height=480,
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
        hovermode='x unified'
    )
    return fig


@st.cache_data
def plot_histogram(values: tuple) -> go.Figure:
    """Create histogram of values (CACHED)."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=values,
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


@st.cache_data
def plot_anomaly_scores(
    timestamps: tuple,
    anomaly_scores: tuple
) -> go.Figure:
    """Create anomaly score visualization (CACHED)."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
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


@st.cache_data
def plot_confusion_matrix(
    cm_data: str,
    algorithm_name: str
) -> go.Figure:
    """Create confusion matrix heatmap (CACHED).
    
    Args:
        cm_data: JSON string of confusion matrix for caching
        algorithm_name: Name of algorithm
    """
    import json
    cm = json.loads(cm_data)
    
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
    st.title("Anomaly Detection System")
    st.divider()
    
    # Sidebar controls
    data_source, algorithm_params, n_cores, window_size = sidebar_controls()

    # ── Clear loaded data when the data source type changes ──────────
    _src_hash = hashlib.md5(str(data_source).encode()).hexdigest()
    if st.session_state.get('_src_hash') != _src_hash:
        st.session_state['_src_hash'] = _src_hash
        st.session_state.data_loaded            = False
        st.session_state.detection_results      = None
        st.session_state.benchmark_results      = None
        st.session_state.benchmark_nab_intervals  = None
        st.session_state.benchmark_anomaly_indices = None
        st.session_state['current_dataset']     = None

    # ── Clear detection/benchmark results when algorithm config changes
    _cfg_hash = hashlib.md5(
        json.dumps(algorithm_params, sort_keys=True, default=str).encode()
    ).hexdigest()
    if st.session_state.get('_cfg_hash') != _cfg_hash:
        st.session_state['_cfg_hash'] = _cfg_hash
        if st.session_state.get('detection_results') is not None:
            st.session_state.detection_results      = None
            st.session_state.benchmark_results      = None
            st.session_state.benchmark_nab_intervals  = None
            st.session_state.benchmark_anomaly_indices = None

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "Data",
        "Detection",
        "Results",
        "Benchmark",
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
                    # Clear previous data immediately — if validation fails below,
                    # the old dataset must not remain active.
                    _prev_file = st.session_state.get('_uploaded_filename')
                    if _prev_file != uploaded_file.name:
                        st.session_state['_uploaded_filename'] = uploaded_file.name
                        st.session_state.data_loaded            = False
                        st.session_state.detection_results      = None
                        st.session_state.benchmark_results      = None
                        st.session_state.benchmark_nab_intervals  = None
                        st.session_state.benchmark_anomaly_indices = None

                    try:
                        df = pd.read_csv(uploaded_file)
                        missing = [c for c in ['timestamp', 'value'] if c not in df.columns]
                        if missing:
                            st.session_state.data_loaded = False
                            st.error(
                                f"CSV-ul trebuie sa contina coloanele **timestamp** si **value**. "
                                f"Coloane gasite: {list(df.columns)}. "
                                f"Lipsesc: {missing}."
                            )
                        else:
                            st.success(f"[OK] Loaded {len(df)} rows")
                            st.session_state.data = df
                            st.session_state.data_loaded = True
                    except Exception as e:
                        st.session_state.data_loaded = False
                        st.error(f"Error loading file: {e}")
            
            else:  # NAB Dataset
                loader = DataLoader()
                all_datasets = loader.list_nab_datasets()
                
                # Serii recomandate fixe
                recommended_paths = {
                    "realAWSCloudwatch/ec2_cpu_utilization_5f5533",
                    "realAWSCloudwatch/ec2_cpu_utilization_24ae8d",
                    "realAWSCloudwatch/rds_cpu_utilization_e47b3b",
                    "realKnownCause/nyc_taxi",
                }
                recommended_order = [
                    "realAWSCloudwatch/ec2_cpu_utilization_5f5533",
                    "realAWSCloudwatch/ec2_cpu_utilization_24ae8d",
                    "realAWSCloudwatch/rds_cpu_utilization_e47b3b",
                    "realKnownCause/nyc_taxi",
                ]

                # Build sorted dataset list
                if st.session_state.gui_mode == 'simple':
                    # Simple Mode: Recommended first (with star), then all others
                    recommended_sorted = [p for p in recommended_order if p in all_datasets]
                    other_datasets = [d for d in all_datasets if d not in recommended_paths]

                    dataset_display = {}
                    for ds in recommended_sorted:
                        dataset_display[ds] = f"* {ds}"
                    for ds in other_datasets:
                        dataset_display[ds] = f"   {ds}"
                    
                    # Sorted options: recommended first, then others
                    sorted_options = recommended_sorted + other_datasets
                    
                    st.markdown("**NAB Dataset Selection (Recommended datasets marked with \*)**")
                    selected_dataset = st.selectbox(
                        "Select dataset:",
                        options=sorted_options,
                        format_func=lambda x: dataset_display[x],
                        index=0 if sorted_options else None
                    )
                else:
                    # Advanced Mode: Show all datasets with recommended marked
                    dataset_display = {}
                    for ds in all_datasets:
                        if ds in recommended_paths:
                            dataset_display[ds] = f"* {ds}"
                        else:
                            dataset_display[ds] = f"   {ds}"
                    
                    st.markdown("**All Available NAB Datasets (Advanced Mode)**")
                    selected_dataset = st.selectbox(
                        "Select dataset:",
                        options=all_datasets,
                        format_func=lambda x: dataset_display[x],
                        index=0 if all_datasets else None
                    )
                
                if selected_dataset and st.button("Load Dataset", use_container_width=True):
                    try:
                        with st.spinner("Loading..."):
                            # Load WITHOUT normalization - let user choose later
                            df = loader.preprocess(selected_dataset, normalize=False)
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
                
                # Create display dataframe with optional normalized columns
                display_df = st.session_state.data.copy()
                
                # If Advanced Mode with normalization enabled, add normalized columns
                if st.session_state.gui_mode == 'advanced' and algorithm_params.get('normalize', False):
                    normalization_method = algorithm_params.get('normalization_method', 'standard')
                    
                    try:
                        _lo, _hi = np.percentile(display_df['value'].values, [1, 99])
                        _clipped = np.clip(display_df['value'].values, _lo, _hi)
                        if normalization_method == 'standard':
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                            normalized = scaler.fit_transform(_clipped.reshape(-1, 1)).flatten()
                            display_df['value_normalized_standard'] = normalized
                        elif normalization_method == 'minmax':
                            from sklearn.preprocessing import MinMaxScaler
                            scaler = MinMaxScaler()
                            normalized = scaler.fit_transform(_clipped.reshape(-1, 1)).flatten()
                            display_df['value_normalized_minmax'] = normalized
                        
                        st.info(f"✓ Data normalized using {normalization_method} method")
                    except Exception as e:
                        st.error(f"Error normalizing data: {e}")
                
                st.dataframe(
                    display_df.head(10),
                    use_container_width=True
                )
            
            with col2:
                st.subheader("Distribution")
                fig_hist = plot_histogram(tuple(st.session_state.data['value'].values))
                st.plotly_chart(fig_hist, use_container_width=True)
    
    # TAB 2: ANOMALY DETECTION
    with tab2:
        if not st.session_state.data_loaded:
            st.warning("[!] Please upload or select data first")
        else:
            st.subheader("Run Anomaly Detection")
            
            # Algorithm info
            algo_name = algorithm_params['algorithm']
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.session_state.gui_mode == 'simple':
                    st.info(
                        f"**Algorithm:** {algorithm_params.get('algorithm', 'unknown')} | "
                        f"**Sensitivity:** {algorithm_params.get('sensitivity', 'Medium (balanced)')}"
                    )
                else:
                    st.info(
                        f"**Algorithm:** {algorithm_params.get('algorithm', 'unknown')} | "
                        f"**Normalize:** {algorithm_params['normalize']}"
                    )
            
            with col2:
                if st.button("Run Detection", use_container_width=True):
                    # Placeholder pentru progress bar și messages
                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()
                    
                    try:
                        df = st.session_state.data
                        values = df['value'].values
                        n_samples = len(values)
                        
                        # Apply normalization if requested (Advanced Mode only).
                        # Robust scaling: clip to the 1st-99th percentile before scaling,
                        # so a few extreme points don't single-handedly set the scale
                        # (standard practice — also means normalization now genuinely
                        # changes which points are flagged, instead of being a no-op
                        # affine transform that cancels out in the detectors' math).
                        if algorithm_params.get('normalize', False):
                            normalization_method = algorithm_params.get('normalization_method', 'standard')
                            lo, hi = np.percentile(values, [1, 99])
                            values = np.clip(values, lo, hi)
                            if normalization_method == 'standard':
                                from sklearn.preprocessing import StandardScaler
                                scaler = StandardScaler()
                                values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
                            elif normalization_method == 'minmax':
                                from sklearn.preprocessing import MinMaxScaler
                                scaler = MinMaxScaler()
                                values = scaler.fit_transform(values.reshape(-1, 1)).flatten()
                        
                        # Progress: 10% - Preparing data
                        with progress_placeholder.container():
                            col_prog1, col_prog2 = st.columns([3, 1])
                            with col_prog1:
                                st.progress(10)
                            with col_prog2:
                                st.caption("Preparing data...")
                        
                        # Progress: 20% - Initializing engine
                        with progress_placeholder.container():
                            col_prog1, col_prog2 = st.columns([3, 1])
                            with col_prog1:
                                st.progress(20)
                            with col_prog2:
                                st.caption("Initializing engine...")
                        
                        engine = AnomalyDetectionEngine()

                        # Progress: 40% - Running detection
                        with progress_placeholder.container():
                            col_prog1, col_prog2 = st.columns([3, 1])
                            with col_prog1:
                                st.progress(40)
                            with col_prog2:
                                st.caption(f"Running {algo_name}...")

                        start_time = time.time()

                        use_spark = algorithm_params.get('use_spark', False)
                        n_cores   = algorithm_params.get('n_cores', 4)

                        if use_spark and algo_name == 'rolling_stats':
                            from spark_engine import SparkRollingZScore
                            spark_detector = SparkRollingZScore()
                            _gui_keys = {'algorithm', 'use_spark', 'n_cores',
                                         'normalize', 'normalization_method', 'sensitivity'}
                            extra = {k: v for k, v in algorithm_params.items()
                                     if k not in _gui_keys}
                            result = spark_detector.detect(values, n_cores=n_cores, **extra)
                        elif algo_name == 'ensemble':
                            result = engine.ensemble_vote(values)
                        else:
                            _gui_keys = {'algorithm', 'use_spark', 'n_cores',
                                         'normalize', 'normalization_method', 'sensitivity'}
                            result = engine.detect_single(
                                algo_name,
                                values,
                                **{k: v for k, v in algorithm_params.items()
                                   if k not in _gui_keys}
                            )
                        
                        exec_time = time.time() - start_time
                        
                        # Progress: 80% - Processing results
                        with progress_placeholder.container():
                            col_prog1, col_prog2 = st.columns([3, 1])
                            with col_prog1:
                                st.progress(80)
                            with col_prog2:
                                st.caption("Processing results...")
                        
                        # Store results
                        st.session_state.detection_results = {
                            'labels': result.labels,
                            'scores': result.scores,
                            'algorithm': result.algorithm,
                            'exec_time': exec_time,
                            'detection_type': result.detection_type,
                        }
                        
                        # Progress: 100% - Done
                        with progress_placeholder.container():
                            col_prog1, col_prog2 = st.columns([3, 1])
                            with col_prog1:
                                st.progress(100)
                            with col_prog2:
                                st.caption("Done!")
                        
                        # Results summary
                        n_anomalies = int(result.labels.sum())
                        with status_placeholder.container():
                            st.success(
                                f"Detection completed in {exec_time:.3f}s: "
                                f"{n_anomalies} anomalies detected ({100*n_anomalies/n_samples:.1f}%)"
                            )
                        
                        # Clear progress bar after 2 seconds
                        time.sleep(2)
                        progress_placeholder.empty()
                        status_placeholder.empty()
                        
                    except Exception as e:
                        st.error(f"Error: {e}")
                        progress_placeholder.empty()
                        import traceback
                        st.error(traceback.format_exc())
            
            # Display results
            if st.session_state.detection_results:
                st.divider()

                results = st.session_state.detection_results
                df = st.session_state.data

                # Stale results from a different dataset — prompt re-run
                if len(results['labels']) != len(df):
                    st.warning("Dataset changed since last detection. Please re-run detection.")
                    st.stop()

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
                        detection_types=results.get('detection_type')
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)
                
                with col2:
                    df_sampled = downsample_data(df, max_points=2000)
                    fig_scores = plot_anomaly_scores(
                        tuple(df_sampled['timestamp'].values),
                        tuple(results['scores'][:len(df_sampled)])
                    )
                    st.plotly_chart(fig_scores, use_container_width=True)
    
    # TAB 3: DETAILED RESULTS
    with tab3:
        if not st.session_state.detection_results:
            st.warning("[!] Run detection first")
        else:
            results = st.session_state.detection_results
            
            # Anomaly details table
            st.subheader("Anomalies Detected")
            
            anomaly_indices = np.where(results['labels'] == 1)[0]
            
            if len(anomaly_indices) > 0:
                anomaly_data = []
                df = st.session_state.data
                dtypes = results.get('detection_type')

                for idx in anomaly_indices[:100]:  # Show first 100
                    row = {
                        'Index': idx,
                        'Timestamp': df.iloc[idx]['timestamp'],
                        'Value': f"{df.iloc[idx]['value']:.2f}",
                        'Score': f"{results['scores'][idx]:.3f}",
                    }
                    if dtypes is not None and idx < len(dtypes) and dtypes[idx]:
                        row['Type'] = dtypes[idx]
                    anomaly_data.append(row)

                anomaly_df = pd.DataFrame(anomaly_data)
                st.dataframe(anomaly_df, use_container_width=True)
            else:
                st.info("No anomalies detected")
    
    # TAB 4: BENCHMARKING
    with tab4:
        st.subheader("Performance Benchmarking")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not st.session_state.detection_results:
                st.info("Run detection first to benchmark against NAB intervals")
            elif st.button("Run Benchmark", use_container_width=True):
                try:
                    if not st.session_state.data_loaded:
                        st.error("Load data first")
                    else:
                        progress_placeholder = st.empty()
                        status_placeholder = st.empty()

                        df     = st.session_state.data
                        det    = st.session_state.detection_results
                        labels = det['labels']          # from the exact Detection run
                        exec_time = det.get('exec_time', 0.0)

                        # Stage 1 – load NAB intervals
                        with progress_placeholder.container():
                            st.progress(0.3)
                            status_placeholder.write("Loading NAB intervals...")
                        time.sleep(0.3)

                        try:
                            from nab_integration import NABIntegration
                            nab = NABIntegration()
                            dataset_name = st.session_state.get('current_dataset', None)

                            if dataset_name:
                                nab_intervals = nab.get_anomaly_intervals(dataset_name, df)

                                if nab_intervals:
                                    # Stage 2 – evaluate (reuse detection labels, no re-run)
                                    with progress_placeholder.container():
                                        st.progress(0.7)
                                        status_placeholder.write("Evaluating against NAB intervals...")
                                    time.sleep(0.3)

                                    anomaly_indices = np.where(labels == 1)[0]

                                    benchmark = Benchmark()
                                    metrics = benchmark.evaluate_with_nab_intervals(
                                        anomaly_indices,
                                        nab_intervals,
                                        len(labels),
                                        exec_time
                                    )

                                    with progress_placeholder.container():
                                        st.progress(1.0)
                                        status_placeholder.write("Done.")
                                    time.sleep(1.5)
                                    progress_placeholder.empty()
                                    status_placeholder.empty()

                                    st.session_state.benchmark_results = metrics
                                    st.session_state.benchmark_nab_intervals = nab_intervals
                                    st.session_state.benchmark_anomaly_indices = anomaly_indices.tolist()
                                else:
                                    progress_placeholder.empty()
                                    status_placeholder.empty()
                                    st.warning("No NAB intervals found for this dataset")
                            else:
                                progress_placeholder.empty()
                                status_placeholder.empty()
                                st.warning("Use a NAB dataset for benchmarking")
                        except Exception as e:
                            st.warning(f"NAB benchmark not available: {e}")
                            progress_placeholder.empty()
                            status_placeholder.empty()

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
                st.metric("Score", f"{metrics.nab_score:.1f}")
            
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

            # ── comparison chart ──────────────────────────────────────
            st.divider()
            st.subheader("Detected vs NAB Ground Truth")
            st.markdown(
                "**Green** = detected NAB intervals (TP) &nbsp;|&nbsp; "
                "**Orange** = missed NAB intervals (FN) &nbsp;|&nbsp; "
                "🟢 = detected anomalies (TP) &nbsp;|&nbsp; "
                "🔴 = false alarms (FP)"
            )

            _nab_iv  = st.session_state.get('benchmark_nab_intervals', [])
            _det_idx = st.session_state.get('benchmark_anomaly_indices', [])

            if _nab_iv and st.session_state.data_loaded:
                bench_fig = plot_benchmark_comparison(
                    st.session_state.data, _nab_iv, _det_idx
                )
                st.plotly_chart(bench_fig, use_container_width=True)
            else:
                st.info("Run the benchmark to see the visual comparison.")




if __name__ == "__main__":
    main()

