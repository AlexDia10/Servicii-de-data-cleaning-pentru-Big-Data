"""
Custom Anomaly Detection Algorithms

Implements 5 distinct algorithms for time-series anomaly detection:
1. Rolling Statistics Z-Score
2. Prediction Error Based
3. Hybrid Anomaly Score
4. Isolation Forest (ML-based)
5. Mean Shift Detector
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass

from utils import setup_logger, safe_division, calculate_trend


logger = setup_logger(__name__)


@dataclass
class AnomalyResult:
    """Container for anomaly detection results."""
    labels: np.ndarray          # Binary labels (0/1)
    scores: np.ndarray          # Anomaly scores [0, 1]
    thresholds: Dict[str, float]  # Applied thresholds
    algorithm: str              # Algorithm name
    parameters: Dict            # Algorithm parameters
    detection_type: np.ndarray = None  # Per-point type string ('Spike', 'Flatline', …)


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detection algorithms."""
    
    def __init__(self, name: str):
        """Initialize detector."""
        self.name = name
        self.logger = logger
    
    @abstractmethod
    def detect(self, data: np.ndarray, **kwargs) -> AnomalyResult:
        """Detect anomalies in data."""
        pass


class RollingStatsZScore(AnomalyDetector):
    """
    Rolling Mean + Rolling Std Dev + Z-Score Anomaly Detection
    
    Formula:
        z_score = (x - rolling_mean) / rolling_std
        anomaly = |z_score| > threshold
    
    Advantages:
    - Simple and interpretable
    - Adaptive to local changes
    - Fast computation
    """
    
    def __init__(self, window_size: int = 20):
        """
        Initialize detector.
        
        Args:
            window_size: Window size for rolling statistics
        """
        super().__init__("Rolling Stats Z-Score")
        self.window_size = window_size
    
    def detect(
        self,
        data: np.ndarray,
        threshold: float = 4.0,
        min_periods: int = None,
        window_size: int = None,
        threshold_method: str = 'fixed',
        percentile: float = None,
        k_std: float = None
    ) -> AnomalyResult:
        """
        Detect anomalies using rolling statistics.
        
        Args:
            data: Time series data
            threshold: Z-score threshold for fixed mode (default: 4.0)
            min_periods: Minimum periods for rolling calculation
            window_size: Override window size if provided
            threshold_method: 'fixed', 'percentile', or 'adaptive_std'
            percentile: Percentile for threshold (e.g., 95) if using percentile method
            k_std: K multiplier for mean+k*std if using adaptive_std method
            
        Returns:
            AnomalyResult with labels and scores
        """
        # Use provided window_size or instance value
        window = window_size if window_size is not None else self.window_size
        
        if min_periods is None:
            min_periods = window // 2
        
        self.logger.info(
            f"Running {self.name}: window={window}, method={threshold_method}"
        )
        
        # Convert to pandas for rolling operations
        series = pd.Series(data)
        
        # Calculate rolling statistics (trailing window: uses only past data,
        # so spike values don't contaminate their own baseline statistics)
        rolling_mean = series.rolling(
            window=window,
            min_periods=min_periods,
            center=False
        ).mean()

        rolling_std = series.rolling(
            window=window,
            min_periods=min_periods,
            center=False
        ).std()
        
        # Calculate z-scores
        z_scores = np.abs(safe_division(
            (data - rolling_mean.values),
            rolling_std.values
        ))
        
        # Determine threshold based on method
        if threshold_method == 'percentile':
            percentile = percentile or 95
            threshold_value = np.nanpercentile(z_scores, percentile)
            threshold_info = {
                'method': 'percentile',
                'percentile': float(percentile),
                'value': float(threshold_value)
            }
        elif threshold_method == 'adaptive_std':
            k_std = k_std or 2.0
            z_mean = np.nanmean(z_scores)
            z_std = np.nanstd(z_scores)
            threshold_value = z_mean + k_std * z_std
            threshold_info = {
                'method': 'adaptive_std',
                'k_std': float(k_std),
                'z_mean': float(z_mean),
                'z_std': float(z_std),
                'value': float(threshold_value)
            }
        else:  # fixed
            threshold_value = threshold
            threshold_info = {
                'method': 'fixed',
                'value': float(threshold_value)
            }
        
        # Normalize scores to [0, 1]
        if threshold_value > 0:
            anomaly_scores = np.minimum(z_scores / threshold_value, 1.0)
        else:
            anomaly_scores = np.zeros_like(z_scores)
        
        # Create binary labels — score >= 1.0 means z-score exactly at or above threshold
        labels = (anomaly_scores >= 1.0).astype(int)

        dtype_arr = np.full(len(labels), '', dtype=object)
        dtype_arr[labels == 1] = 'Spike'

        return AnomalyResult(
            labels=labels,
            scores=anomaly_scores,
            thresholds=threshold_info,
            algorithm=self.name,
            parameters={
                'window_size': window,
                'min_periods': min_periods,
                'threshold_method': threshold_method,
                'percentile': percentile,
                'k_std': k_std,
                **threshold_info
            },
            detection_type=dtype_arr
        )


class PredictionErrorAnomaly(AnomalyDetector):
    """
    Prediction Error Based Anomaly Detection
    
    Uses moving average forecast to predict next value.
    
    Formula:
        predicted = moving_average(history)
        error = |actual - predicted|
        anomaly = error > threshold * std(error)
    
    Advantages:
    - Captures deviations from expected behavior
    - Works with trending data
    - Sensitive to sudden changes
    """
    
    def __init__(self, forecast_window: int = 10):
        """
        Initialize detector.
        
        Args:
            forecast_window: Window for moving average forecast
        """
        super().__init__("Prediction Error")
        self.forecast_window = forecast_window
    
    def detect(
        self,
        data: np.ndarray,
        threshold: float = 4.5,
        smoothing_window: int = 1,
        forecast_window: int = None,
        threshold_method: str = 'fixed',
        percentile: float = None,
        k_std: float = None
    ) -> AnomalyResult:
        """
        Detect anomalies using prediction errors.
        
        Args:
            data: Time series data
            threshold: Error threshold for fixed mode (default: 4.5)
            smoothing_window: Window for smoothing error signal
            forecast_window: Override forecast window if provided
            threshold_method: 'fixed', 'percentile', or 'adaptive_std'
            percentile: Percentile for threshold if using percentile method
            k_std: K multiplier for mean+k*std if using adaptive_std method
            
        Returns:
            AnomalyResult with labels and scores
        """
        # Use provided forecast_window or instance value
        fw = forecast_window if forecast_window is not None else self.forecast_window
        
        self.logger.info(
            f"Running {self.name}: forecast_window={fw}, method={threshold_method}"
        )
        
        series = pd.Series(data)
        
        # Generate predictions using moving average
        predictions = np.zeros_like(data)
        predictions[:fw] = data[:fw].mean()
        
        for i in range(fw, len(data)):
            predictions[i] = data[i - fw:i].mean()
        
        # Calculate prediction errors
        errors = np.abs(data - predictions)
        
        # Smooth error signal
        error_series = pd.Series(errors)
        error_smooth = error_series.rolling(
            window=smoothing_window,
            center=True,
            min_periods=1
        ).mean().values
        
        # Determine threshold based on method
        if threshold_method == 'percentile':
            percentile = percentile or 95
            threshold_value = np.nanpercentile(error_smooth, percentile)
            threshold_info = {
                'method': 'percentile',
                'percentile': float(percentile),
                'value': float(threshold_value)
            }
        elif threshold_method == 'adaptive_std':
            k_std = k_std or 2.0
            error_mean = np.nanmean(error_smooth)
            error_std = np.nanstd(error_smooth)
            threshold_value = error_mean + k_std * error_std
            threshold_info = {
                'method': 'adaptive_std',
                'k_std': float(k_std),
                'error_mean': float(error_mean),
                'error_std': float(error_std),
                'value': float(threshold_value)
            }
        else:  # fixed
            error_std = np.std(errors)
            error_mean = np.mean(errors)
            threshold_value = error_mean + threshold * error_std
            threshold_info = {
                'method': 'fixed',
                'value': float(threshold_value),
                'error_mean': float(error_mean),
                'error_std': float(error_std)
            }
        
        # Normalize anomaly scores
        if threshold_value > 0:
            anomaly_scores = np.minimum(error_smooth / threshold_value, 1.0)
        else:
            anomaly_scores = np.zeros_like(error_smooth)
        
        # Create binary labels — score >= 1.0 means error exactly at or above threshold
        labels = (anomaly_scores >= 1.0).astype(int)

        dtype_arr = np.full(len(labels), '', dtype=object)
        dtype_arr[labels == 1] = 'Spike'

        return AnomalyResult(
            labels=labels,
            scores=anomaly_scores,
            thresholds=threshold_info,
            algorithm=self.name,
            parameters={
                'forecast_window': fw,
                'smoothing_window': smoothing_window,
                'threshold_method': threshold_method,
                'percentile': percentile,
                'k_std': k_std,
                **threshold_info
            },
            detection_type=dtype_arr
        )


class HybridAnomalyScore(AnomalyDetector):
    """
    Hybrid Anomaly Detection combining multiple signals

    Combines five detection channels:
    - Z-Score:      Deviation from rolling mean (spike detection)
    - Trend:        Abrupt changes in local trend slope
    - Volatility:   Sudden increase in local variance
    - Mean Shift:   Abrupt sustained level change between two adjacent windows
    - Flatline:     Near-zero rolling range — frozen/stuck sensor (independent channel)

    The first four channels are combined into a weighted anomaly score.
    Flatline is OR-ed independently so a frozen sensor (low amplitude,
    low variance) is not diluted by zero scores on the other channels.

    Formula:
        score   = (w1·z_norm + w2·trend_norm + w3·vol_norm + w4·ms_norm)
                  / (w1 + w2 + w3 + w4)
        anomaly = score >= 0.80  OR  flatline_detected
    """
    
    def __init__(
        self,
        window_size: int = 20,
        trend_window: int = 5,
        volatility_window: int = 10
    ):
        """
        Initialize detector.
        
        Args:
            window_size: Window for z-score calculation
            trend_window: Window for trend calculation
            volatility_window: Window for volatility calculation
        """
        super().__init__("Hybrid Anomaly Score")
        self.window_size = window_size
        self.trend_window = trend_window
        self.volatility_window = volatility_window
    
    def detect(
        self,
        data: np.ndarray,
        zscore_threshold: float = 3.5,
        trend_threshold: float = 1.2,
        volatility_threshold: float = 3.5,
        weights: Tuple[float, ...] = (0.5, 0.3, 0.2),
        window_size: int = None,
        trend_window: int = None,
        volatility_window: int = None,
        threshold_method: str = 'fixed',
        percentile: float = None,
        k_std: float = None
    ) -> AnomalyResult:
        """
        Detect anomalies using hybrid scoring.
        
        Args:
            data: Time series data
            zscore_threshold: Z-score threshold (or k for adaptive methods)
            trend_threshold: Trend change threshold (or k for adaptive methods)
            volatility_threshold: Volatility spike threshold (or k for adaptive methods)
            weights: Weights for [z_score, trend, volatility]
            window_size: Override window size if provided
            trend_window: Override trend window if provided
            volatility_window: Override volatility window if provided
            threshold_method: 'fixed', 'percentile', or 'adaptive_std'
            percentile: Percentile value if using percentile method
            k_std: K multiplier if using adaptive_std method
            
        Returns:
            AnomalyResult with labels and scores
        """
        # Use provided windows or instance values
        ws = window_size if window_size is not None else self.window_size
        tw = trend_window if trend_window is not None else self.trend_window
        vw = volatility_window if volatility_window is not None else self.volatility_window
        
        self.logger.info(f"Running {self.name} with method: {threshold_method}")
        
        series = pd.Series(data)
        n = len(data)
        
        # 1. Z-Score component
        rolling_mean = series.rolling(
            window=ws,
            min_periods=ws // 2,
            center=True
        ).mean()
        
        rolling_std = series.rolling(
            window=ws,
            min_periods=ws // 2,
            center=True
        ).std()
        
        z_scores = np.abs(safe_division(
            data - rolling_mean.values,
            rolling_std.values
        ))
        
        # 2. Trend deviation component
        trends = calculate_trend(data, window=tw)
        trend_changes = np.abs(np.gradient(trends))
        
        # 3. Volatility spike component (unusually HIGH local variance)
        volatility = series.rolling(
            window=vw,
            min_periods=vw // 2,
            center=True
        ).std().values

        volatility_mean = np.nanmean(volatility)
        volatility_std  = np.nanstd(volatility)

        volatility_spike = np.maximum(
            safe_division(
                volatility - volatility_mean,
                volatility_std
            ), 0
        )

        # 4. Flatline detection — rolling range (max-min) over a backward-looking
        # window. A frozen sensor repeats the same value → range ≈ 0. A quiet
        # but alive baseline always has some oscillation → range > 0. Using
        # range rather than std is more discriminating because it is sensitive
        # to ANY value change, not just statistical spread.
        flat_ws = ws * 2
        roll_max = series.rolling(window=flat_ws, min_periods=flat_ws).max()
        roll_min = series.rolling(window=flat_ws, min_periods=flat_ws).min()
        data_range = float(np.nanmax(data) - np.nanmin(data)) or 1.0
        roll_range = (roll_max - roll_min).fillna(data_range).values

        # Calculate thresholds based on method
        if threshold_method == 'percentile':
            percentile = percentile or 95
            z_thresh     = np.nanpercentile(z_scores,         percentile)
            trend_thresh = np.nanpercentile(trend_changes,    percentile)
            vol_thresh   = np.nanpercentile(volatility_spike, percentile)
            threshold_info = {
                'method': 'percentile',
                'percentile': float(percentile),
                'z_score': float(z_thresh),
                'trend': float(trend_thresh),
                'volatility': float(vol_thresh),
            }
        elif threshold_method == 'adaptive_std':
            k_std = k_std or 2.0
            z_thresh     = np.nanmean(z_scores)         + k_std * np.nanstd(z_scores)
            trend_thresh = np.nanmean(trend_changes)    + k_std * np.nanstd(trend_changes)
            vol_thresh   = np.nanmean(volatility_spike) + k_std * np.nanstd(volatility_spike)
            threshold_info = {
                'method': 'adaptive_std',
                'k_std': float(k_std),
                'z_score': float(z_thresh),
                'trend': float(trend_thresh),
                'volatility': float(vol_thresh),
            }
        else:  # fixed
            z_thresh     = zscore_threshold
            trend_thresh = trend_threshold
            vol_thresh   = volatility_threshold
            threshold_info = {
                'method': 'fixed',
                'z_score': float(z_thresh),
                'trend': float(trend_thresh),
                'volatility': float(vol_thresh),
            }

        # Normalize three channels to [0, 1]
        z_scores_norm = np.minimum(z_scores / z_thresh, 1.0) if z_thresh > 0 \
                        else np.zeros_like(z_scores)
        trend_changes_norm = np.minimum(trend_changes / trend_thresh, 1.0) if trend_thresh > 0 \
                             else np.zeros_like(trend_changes)
        volatility_spike_norm = np.minimum(volatility_spike / vol_thresh, 1.0) if vol_thresh > 0 \
                                else np.zeros_like(volatility_spike)

        # Flatline: 1 % of the full signal range as guard.
        range_guard = max(data_range * 0.01, 1e-6)
        skip_start = np.zeros(len(data), dtype=bool)
        skip_start[:flat_ws] = True
        flatline_mask = (~skip_start) & (roll_range < range_guard)
        flatline_detected = flatline_mask.copy()
        flatline_detected[1:] &= ~flatline_mask[:-1]
        threshold_info['flatline_range_guard'] = float(range_guard)
        threshold_info['flatline_detections']  = int(flatline_detected.sum())

        # Weighted combination of three channels.
        w_list = list(weights)[:3]
        while len(w_list) < 3:
            w_list.append(0.0)
        w1, w2, w3 = w_list
        total_w = (w1 + w2 + w3) or 1.0
        anomaly_scores = (
            w1 * z_scores_norm +
            w2 * trend_changes_norm +
            w3 * volatility_spike_norm
        ) / total_w

        # Volatility independent OR channel: fires when vol spike is extreme
        # (≥ 2.5× its threshold), bypassing the combined 0.80 score threshold.
        vol_strong = (volatility_spike >= 2.5 * vol_thresh) if vol_thresh > 0 \
                     else np.zeros(n, dtype=bool)
        vol_edge = vol_strong.copy()
        vol_edge[1:] &= ~vol_strong[:-1]

        labels = (
            (anomaly_scores >= 0.80) | vol_edge | flatline_detected
        ).astype(int)

        anomaly_scores = np.where(
            vol_edge | flatline_detected,
            np.maximum(anomaly_scores, 0.85),
            anomaly_scores
        )

        # Per-point detection type for visual distinction in the dashboard.
        # Precompute trend-sign consistency over an 11-point window:
        #   high value (≥ 0.5) → slope is consistently positive or negative → Trend
        #   low value  (< 0.5) → slope alternates rapidly             → Volatility burst
        _trend_pos = np.convolve((trends > 0).astype(np.int8),
                                 np.ones(11, dtype=np.int8), mode='same')
        _trend_neg = np.convolve((trends < 0).astype(np.int8),
                                 np.ones(11, dtype=np.int8), mode='same')
        _sign_consis = np.abs(_trend_pos - _trend_neg) / 11.0

        # Points where z-score is the dominant channel are spike centers.
        # Dilate ±5 points so the rising/falling edge of a spike (which has
        # high trend score but is caused by the spike) is also caught.
        _z_dominant = (z_scores_norm >= trend_changes_norm) & \
                      (z_scores_norm >= volatility_spike_norm)
        _spike_nearby = np.convolve(_z_dominant.astype(np.int8),
                                    np.ones(11, dtype=np.int8), mode='same') > 0

        dtype_arr = np.full(n, '', dtype=object)
        for _i in np.where(labels == 1)[0]:
            if flatline_detected[_i]:
                dtype_arr[_i] = 'Flatline'
            elif vol_edge[_i]:
                dtype_arr[_i] = 'Volatility'
            else:
                ch = {
                    'Spike':      float(z_scores_norm[_i]),
                    'Trend':      float(trend_changes_norm[_i]),
                    'Volatility': float(volatility_spike_norm[_i]),
                }
                best = max(ch, key=ch.get)
                # Spike edges: the slope on the rising/falling side of a spike
                # looks like a trend but is caused by the spike itself.
                if best == 'Trend' and _spike_nearby[_i]:
                    best = 'Spike'
                # Oscillating burst: consistent slope direction required for Trend.
                elif best == 'Trend' and _sign_consis[_i] < 0.5:
                    best = 'Volatility'
                dtype_arr[_i] = best

        return AnomalyResult(
            labels=labels,
            scores=anomaly_scores,
            thresholds=threshold_info,
            algorithm=self.name,
            parameters={
                'window_size': self.window_size,
                'trend_window': self.trend_window,
                'volatility_window': self.volatility_window,
                'weights': [w1, w2, w3],
                'threshold_method': threshold_method,
                'percentile': percentile,
                'k_std': k_std,
                'flatline_detections': int(flatline_detected.sum()),
                **threshold_info
            },
            detection_type=dtype_arr
        )


class IsolationForestDetector(AnomalyDetector):
    """
    Isolation Forest Anomaly Detection (Machine Learning)

    Isolates anomalies by randomly partitioning the feature space.
    Uses rolling temporal features (mean, std, diff) so it can detect
    contextual anomalies that statistical z-score methods miss.

    Advantages:
    - Detects contextual anomalies (not just statistical spikes)
    - No assumption about data distribution
    - Effective when anomalies are rare and structurally different
    """

    def __init__(self, n_estimators: int = 100, window_size: int = 20):
        super().__init__("Isolation Forest")
        self.n_estimators = n_estimators
        self.window_size = window_size

    def detect(
        self,
        data: np.ndarray,
        contamination: float = 0.01,
        n_estimators: int = None,
        window_size: int = None,
        random_state: int = 42,
        **kwargs
    ) -> AnomalyResult:
        """
        Detect anomalies using Isolation Forest on rolling temporal features.

        Args:
            data: Time series data
            contamination: Expected fraction of anomalies (0.001–0.1)
            n_estimators: Number of isolation trees (default 100)
            window_size: Rolling window for feature extraction
            random_state: Random seed for reproducibility

        Returns:
            AnomalyResult with labels and scores
        """
        from sklearn.ensemble import IsolationForest

        ws = window_size if window_size is not None else self.window_size
        n_est = n_estimators if n_estimators is not None else self.n_estimators

        self.logger.info(
            f"Running {self.name}: contamination={contamination}, "
            f"n_estimators={n_est}, window={ws}"
        )

        series = pd.Series(data)

        # Rolling temporal features give the model temporal context,
        # enabling contextual anomaly detection beyond point spikes.
        rolling_mean = (
            series.rolling(window=ws, min_periods=ws // 2, center=True)
            .mean()
            .fillna(series.mean())
            .values
        )
        rolling_std = (
            series.rolling(window=ws, min_periods=ws // 2, center=True)
            .std()
            .fillna(series.std())
            .values
        )
        rolling_diff = series.diff().abs().fillna(0).values

        # Normalize each value by the TYPICAL spike height so that repeating
        # spikes form a dense cluster at ratio≈1.0 (not isolated → not anomalous)
        # while a spike that exceeds the typical level gets ratio>1.0 (isolated).
        #
        # "Typical spike height" = median of all values above the 99th percentile.
        # This robustly captures the recurring spike level without being pulled
        # up by the single large anomaly (median is more resistant than mean).
        p99 = np.percentile(data, 99)
        spike_zone = data[data > p99]
        typical_spike = float(np.median(spike_zone)) if len(spike_zone) >= 2 else float(p99)
        value_ratio = data / (typical_spike + 1e-8)

        # Flatline detection feature: how far BELOW normal is the local variance?
        # A frozen sensor (flatline) has rolling_std ≈ 0 while normal data has
        # substantial variance.
        std_mean = rolling_std.mean()
        std_std  = rolling_std.std() + 1e-8
        flatline_score = np.maximum(0.0, (std_mean - rolling_std) / std_std)

        features = np.column_stack([
            data, rolling_mean, rolling_std, rolling_diff, value_ratio, flatline_score
        ])

        model = IsolationForest(
            n_estimators=n_est,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(features)

        # decision_function: lower (more negative) = more anomalous
        raw_scores = model.decision_function(features)

        # Invert and normalize to [0, 1] so higher = more anomalous
        inverted = -raw_scores
        s_min, s_max = inverted.min(), inverted.max()
        if s_max > s_min:
            anomaly_scores = (inverted - s_min) / (s_max - s_min)
        else:
            anomaly_scores = np.zeros_like(inverted)

        # model.predict returns -1 for anomalies, 1 for normal
        pred = model.predict(features)
        labels = (pred == -1).astype(int)

        dtype_arr = np.full(len(labels), '', dtype=object)
        dtype_arr[labels == 1] = 'Spike'

        return AnomalyResult(
            labels=labels,
            scores=anomaly_scores,
            thresholds={
                'contamination': contamination,
            },
            algorithm=self.name,
            parameters={
                'n_estimators': n_est,
                'contamination': contamination,
                'window_size': ws,
                'features': ['value', 'rolling_mean', 'rolling_std', 'rolling_diff',
                             'value_ratio_to_typical_spike', 'flatline_score']
            },
            detection_type=dtype_arr
        )


class MeanShiftDetector(AnomalyDetector):
    """
    Mean Shift Anomaly Detection

    Compares the means of two adjacent sliding windows (before/after each
    point) to detect abrupt level changes in the time series.

    Score formula:
        score[i] = |mean(after) - mean(before)| / (std(before) + epsilon)

    where epsilon = 1e-8 prevents division by zero on constant segments.
    Edge points (first and last `window` indices) receive score 0.

    Effective for: step changes, regime shifts, sudden level drops/rises.
    Less effective for: isolated spikes (both windows are dominated by
    baseline, so the difference is small).
    """

    def __init__(self, window: int = 50):
        super().__init__("Mean Shift Detector")
        self.window = window

    def detect(
        self,
        data: np.ndarray,
        window: int = None,
        threshold: float = 3.0,
        threshold_method: str = 'adaptive_std',
        percentile: float = None,
        k_std: float = None,
    ) -> AnomalyResult:
        """
        Detect abrupt mean shifts using dual sliding windows.

        Args:
            data: Time series data
            window: Half-window size (uses instance default if None)
            threshold: Fixed threshold for 'fixed' method
            threshold_method: 'fixed', 'percentile', or 'adaptive_std'
            percentile: Percentile cutoff when using 'percentile' method
            k_std: Sigma multiplier when using 'adaptive_std' method

        Returns:
            AnomalyResult with continuous scores and binary labels
        """
        w = window if window is not None else self.window
        n = len(data)

        self.logger.info(
            f"Running {self.name}: window={w}, method={threshold_method}"
        )

        scores = np.zeros(n)
        # Use a data-scale-relative floor for the denominator so that perfectly
        # flat segments (std=0) don't generate astronomically inflated scores.
        _min_denom = max(float(data.std()) * 0.01, 1e-4)
        for i in range(w, n - w):
            before = data[i - w:i]
            after  = data[i:i + w]
            scores[i] = abs(after.mean() - before.mean()) / max(before.std(), _min_denom)

        if threshold_method == 'percentile':
            pct = percentile or 95.0
            threshold_value = np.nanpercentile(scores, pct)
            threshold_info = {
                'method': 'percentile',
                'percentile': float(pct),
                'value': float(threshold_value),
            }
        elif threshold_method == 'adaptive_std':
            k = k_std or 3.0
            s_mean = np.nanmean(scores)
            s_std  = np.nanstd(scores)
            threshold_value = s_mean + k * s_std
            threshold_info = {
                'method': 'adaptive_std',
                'k_std': float(k),
                'score_mean': float(s_mean),
                'score_std':  float(s_std),
                'value': float(threshold_value),
            }
        else:  # fixed
            threshold_value = float(threshold)
            threshold_info = {
                'method': 'fixed',
                'value': threshold_value,
            }

        if threshold_value > 0:
            anomaly_scores = np.minimum(scores / threshold_value, 1.0)
        else:
            anomaly_scores = np.zeros_like(scores)

        # Peak-of-run: within each consecutive run of points where the
        # normalised score ≥ 1.0, keep only the single point with the highest
        # RAW score (= exact transition where before-window is fully old level
        # and after-window is fully new level). Rising-edge would fire at the
        # start of the run — before the transition is complete — placing the
        # marker visually inside the wrong segment.
        above = anomaly_scores >= 1.0
        labels = np.zeros(n, dtype=int)
        i = 0
        while i < n:
            if above[i]:
                run_start = i
                while i < n and above[i]:
                    i += 1
                peak = run_start + int(np.argmax(scores[run_start:i]))
                labels[peak] = 1
            else:
                i += 1

        dtype_arr = np.full(n, '', dtype=object)
        dtype_arr[labels == 1] = 'Level Shift'

        return AnomalyResult(
            labels=labels,
            scores=anomaly_scores,
            thresholds=threshold_info,
            algorithm=self.name,
            parameters={
                'window': w,
                'threshold_method': threshold_method,
                'threshold': threshold_value,
                'percentile': percentile,
                'k_std': k_std,
                **threshold_info,
            },
            detection_type=dtype_arr
        )


class AnomalyDetectionEngine:
    """
    Orchestrates multiple anomaly detection algorithms.
    Allows comparison and ensemble approaches.
    """
    
    def __init__(self):
        """Initialize detection engine with available algorithms."""
        self.detectors = {
            'rolling_stats': RollingStatsZScore(),
            'prediction_error': PredictionErrorAnomaly(),
            'hybrid': HybridAnomalyScore(),
            'isolation_forest': IsolationForestDetector(),
            'mean_shift': MeanShiftDetector(),
        }
        self.results = {}
        self.logger = logger
    
    def detect_all(self, data: np.ndarray, **kwargs) -> Dict[str, AnomalyResult]:
        """
        Run all anomaly detection algorithms.
        
        Args:
            data: Time series data
            **kwargs: Algorithm-specific parameters
            
        Returns:
            Dictionary mapping algorithm names to results
        """
        self.logger.info("Running all anomaly detection algorithms")
        self.results = {}
        
        for algorithm_name, detector in self.detectors.items():
            try:
                self.results[algorithm_name] = detector.detect(data, **kwargs)
                self.logger.info(f"[OK] {algorithm_name} completed")
            except Exception as e:
                self.logger.error(f"✗ {algorithm_name} failed: {e}")
        
        return self.results
    
    def detect_single(
        self,
        algorithm: str,
        data: np.ndarray,
        **kwargs
    ) -> AnomalyResult:
        """
        Run single algorithm.
        
        Args:
            algorithm: Algorithm name
            data: Time series data
            **kwargs: Algorithm-specific parameters
            
        Returns:
            AnomalyResult
        """
        if algorithm not in self.detectors:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return self.detectors[algorithm].detect(data, **kwargs)
    
    def ensemble_vote(self, data: np.ndarray, **kwargs) -> AnomalyResult:
        """
        Ensemble detection using majority voting.
        
        Args:
            data: Time series data
            **kwargs: Algorithm parameters
            
        Returns:
            Ensemble AnomalyResult
        """
        self.logger.info("Running ensemble detection with majority voting")
        
        results = self.detect_all(data, **kwargs)
        
        # Stack all labels and compute majority vote
        all_labels = np.stack([r.labels for r in results.values()], axis=0)
        ensemble_labels = (np.mean(all_labels, axis=0) >= 0.5).astype(int)
        
        # Average scores
        all_scores = np.stack([r.scores for r in results.values()], axis=0)
        ensemble_scores = np.mean(all_scores, axis=0)
        
        return AnomalyResult(
            labels=ensemble_labels,
            scores=ensemble_scores,
            thresholds={},
            algorithm="Ensemble (Voting)",
            parameters={'method': 'majority_vote'}
        )


if __name__ == "__main__":
    # Example usage
    from data_loader import DataLoader
    
    loader = DataLoader()
    data = loader.preprocess("art_daily_small_noise")
    values = data['value'].values
    
    engine = AnomalyDetectionEngine()
    results = engine.detect_all(values)
    
    for name, result in results.items():
        print(f"{name}: {result.labels.sum()} anomalies detected")
