"""
Custom Anomaly Detection Algorithms

Implements 4 distinct algorithms for time-series anomaly detection:
1. Rolling Statistics Z-Score
2. Prediction Error Based
3. Hybrid Anomaly Score
4. Isolation Forest (ML-based)
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
        
        # Calculate rolling statistics
        rolling_mean = series.rolling(
            window=window,
            min_periods=min_periods,
            center=True
        ).mean()
        
        rolling_std = series.rolling(
            window=window,
            min_periods=min_periods,
            center=True
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
            }
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
        smoothing_window: int = 5,
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
            }
        )


class HybridAnomalyScore(AnomalyDetector):
    """
    Hybrid Anomaly Detection combining multiple signals
    
    Combines:
    - Z-Score: Deviation from mean
    - Trend Deviation: Local trend changes
    - Volatility Spike: Sudden increase in variance
    
    Formula:
        anomaly_score = w1*z_score + w2*trend_dev + w3*volatility_spike
    
    Advantages:
    - Robust to different anomaly types
    - Reduces false positives from single signals
    - Captures complex patterns
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
        weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
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
        
        # 3. Volatility spike component
        volatility = series.rolling(
            window=vw,
            min_periods=vw // 2,
            center=True
        ).std().values
        
        volatility_mean = np.nanmean(volatility)
        volatility_std = np.nanstd(volatility)
        
        volatility_spike = np.maximum(
            safe_division(
                volatility - volatility_mean,
                volatility_std
            ), 0
        )
        
        # Calculate thresholds based on method
        if threshold_method == 'percentile':
            percentile = percentile or 95
            z_thresh = np.nanpercentile(z_scores, percentile)
            trend_thresh = np.nanpercentile(trend_changes, percentile)
            vol_thresh = np.nanpercentile(volatility_spike, percentile)
            
            threshold_info = {
                'method': 'percentile',
                'percentile': float(percentile),
                'z_score': float(z_thresh),
                'trend': float(trend_thresh),
                'volatility': float(vol_thresh)
            }
        elif threshold_method == 'adaptive_std':
            k_std = k_std or 2.0
            z_mean = np.nanmean(z_scores)
            z_std_val = np.nanstd(z_scores)
            z_thresh = z_mean + k_std * z_std_val
            
            trend_mean = np.nanmean(trend_changes)
            trend_std = np.nanstd(trend_changes)
            trend_thresh = trend_mean + k_std * trend_std
            
            vol_mean = np.nanmean(volatility_spike)
            vol_std = np.nanstd(volatility_spike)
            vol_thresh = vol_mean + k_std * vol_std
            
            threshold_info = {
                'method': 'adaptive_std',
                'k_std': float(k_std),
                'z_score': float(z_thresh),
                'trend': float(trend_thresh),
                'volatility': float(vol_thresh)
            }
        else:  # fixed
            z_thresh = zscore_threshold
            trend_thresh = trend_threshold
            vol_thresh = volatility_threshold
            
            threshold_info = {
                'method': 'fixed',
                'z_score': float(z_thresh),
                'trend': float(trend_thresh),
                'volatility': float(vol_thresh)
            }
        
        # Normalize components
        z_scores_norm = np.minimum(z_scores / z_thresh, 1.0) if z_thresh > 0 else np.zeros_like(z_scores)
        trend_changes_norm = np.minimum(
            trend_changes / trend_thresh, 1.0
        ) if trend_thresh > 0 else np.zeros_like(trend_changes)
        volatility_spike_norm = np.minimum(
            volatility_spike / vol_thresh, 1.0
        ) if vol_thresh > 0 else np.zeros_like(volatility_spike)
        
        # Combine components with weights
        w1, w2, w3 = weights
        anomaly_scores = (
            w1 * z_scores_norm +
            w2 * trend_changes_norm +
            w3 * volatility_spike_norm
        ) / (w1 + w2 + w3)
        
        # Hybrid score is a weighted mean in [0,1]; 0.80 requires at least 2 signals above threshold
        labels = (anomaly_scores >= 0.80).astype(int)

        return AnomalyResult(
            labels=labels,
            scores=anomaly_scores,
            thresholds=threshold_info,
            algorithm=self.name,
            parameters={
                'window_size': self.window_size,
                'trend_window': self.trend_window,
                'volatility_window': self.volatility_window,
                'weights': weights,
                'threshold_method': threshold_method,
                'percentile': percentile,
                'k_std': k_std,
                **threshold_info
            }
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

        features = np.column_stack([data, rolling_mean, rolling_std, rolling_diff])

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

        return AnomalyResult(
            labels=labels,
            scores=anomaly_scores,
            thresholds={'contamination': contamination},
            algorithm=self.name,
            parameters={
                'n_estimators': n_est,
                'contamination': contamination,
                'window_size': ws,
                'features': ['value', 'rolling_mean', 'rolling_std', 'rolling_diff']
            }
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
            'isolation_forest': IsolationForestDetector()
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
