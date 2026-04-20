"""
Custom Anomaly Detection Algorithms

Implements 3 distinct algorithms for time-series anomaly detection:
1. Rolling Statistics Z-Score
2. Prediction Error Based
3. Hybrid Anomaly Score
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
        threshold: float = 2.5,
        min_periods: int = None,
        window_size: int = None
    ) -> AnomalyResult:
        """
        Detect anomalies using rolling statistics.
        
        Args:
            data: Time series data
            threshold: Z-score threshold for anomaly (default: 2.5)
            min_periods: Minimum periods for rolling calculation
            window_size: Override window size if provided
            
        Returns:
            AnomalyResult with labels and scores
        """
        # Use provided window_size or instance value
        window = window_size if window_size is not None else self.window_size
        
        if min_periods is None:
            min_periods = window // 2
        
        self.logger.info(
            f"Running {self.name}: window={self.window_size}, threshold={threshold}"
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
        
        # Normalize scores to [0, 1]
        if z_scores.max() > 0:
            anomaly_scores = np.minimum(z_scores / threshold, 1.0)
        else:
            anomaly_scores = np.zeros_like(z_scores)
        
        # Create binary labels
        labels = (anomaly_scores >= 1.0).astype(int)
        
        return AnomalyResult(
            labels=labels,
            scores=anomaly_scores,
            thresholds={'z_score': threshold},
            algorithm=self.name,
            parameters={
                'window_size': window,
                'min_periods': min_periods,
                'threshold': threshold
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
        threshold: float = 3.0,
        smoothing_window: int = 5,
        forecast_window: int = None
    ) -> AnomalyResult:
        """
        Detect anomalies using prediction errors.
        
        Args:
            data: Time series data
            threshold: Standard deviation multiplier for error threshold
            smoothing_window: Window for smoothing error signal
            forecast_window: Override forecast window if provided
            
        Returns:
            AnomalyResult with labels and scores
        """
        # Use provided forecast_window or instance value
        fw = forecast_window if forecast_window is not None else self.forecast_window
        
        self.logger.info(
            f"Running {self.name}: forecast_window={fw}, "
            f"threshold={threshold}"
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
        
        # Calculate error threshold
        error_std = np.std(errors)
        error_mean = np.mean(errors)
        error_threshold = error_mean + threshold * error_std
        
        # Normalize anomaly scores
        if error_threshold > 0:
            anomaly_scores = np.minimum(error_smooth / error_threshold, 1.0)
        else:
            anomaly_scores = np.zeros_like(error_smooth)
        
        # Create binary labels
        labels = (anomaly_scores >= 1.0).astype(int)
        
        return AnomalyResult(
            labels=labels,
            scores=anomaly_scores,
            thresholds={'error': error_threshold},
            algorithm=self.name,
            parameters={
                'forecast_window': fw,
                'smoothing_window': smoothing_window,
                'threshold': threshold,
                'error_mean': float(error_mean),
                'error_std': float(error_std)
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
        zscore_threshold: float = 2.0,
        trend_threshold: float = 0.5,
        volatility_threshold: float = 2.0,
        weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        window_size: int = None,
        trend_window: int = None,
        volatility_window: int = None
    ) -> AnomalyResult:
        """
        Detect anomalies using hybrid scoring.
        
        Args:
            data: Time series data
            zscore_threshold: Z-score threshold
            trend_threshold: Trend change threshold
            volatility_threshold: Volatility spike threshold
            weights: Weights for [z_score, trend, volatility]
            window_size: Override window size if provided
            trend_window: Override trend window if provided
            volatility_window: Override volatility window if provided
            
        Returns:
            AnomalyResult with labels and scores
        """
        # Use provided windows or instance values
        ws = window_size if window_size is not None else self.window_size
        tw = trend_window if trend_window is not None else self.trend_window
        vw = volatility_window if volatility_window is not None else self.volatility_window
        
        self.logger.info(f"Running {self.name} with weights: {weights}")
        
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
        z_scores_norm = np.minimum(z_scores / zscore_threshold, 1.0)
        
        # 2. Trend deviation component
        trends = calculate_trend(data, window=tw)
        trend_changes = np.abs(np.gradient(trends))
        trend_changes_norm = np.minimum(
            trend_changes / trend_threshold, 1.0
        ) if trend_threshold > 0 else np.zeros_like(trend_changes)
        
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
        volatility_spike_norm = np.minimum(
            volatility_spike / volatility_threshold, 1.0
        )
        
        # Combine components with weights
        w1, w2, w3 = weights
        anomaly_scores = (
            w1 * z_scores_norm +
            w2 * trend_changes_norm +
            w3 * volatility_spike_norm
        ) / (w1 + w2 + w3)
        
        # Create binary labels
        labels = (anomaly_scores >= 0.5).astype(int)
        
        return AnomalyResult(
            labels=labels,
            scores=anomaly_scores,
            thresholds={
                'zscore': zscore_threshold,
                'trend': trend_threshold,
                'volatility': volatility_threshold
            },
            algorithm=self.name,
            parameters={
                'window_size': self.window_size,
                'trend_window': self.trend_window,
                'volatility_window': self.volatility_window,
                'weights': weights,
                'zscore_threshold': zscore_threshold,
                'trend_threshold': trend_threshold,
                'volatility_threshold': volatility_threshold
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
            'hybrid': HybridAnomalyScore()
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
                self.logger.info(f"✓ {algorithm_name} completed")
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
