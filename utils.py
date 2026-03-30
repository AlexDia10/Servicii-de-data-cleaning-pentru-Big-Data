"""
utility functions for anomaly detection system

Provides helper functions for logging, data processing, and visualization.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from datetime import datetime


# Configure logging
def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Set up logger for application modules.
    
    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


logger = setup_logger(__name__)


# Data validation and preprocessing
def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate DataFrame has required columns and structure.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, "DataFrame is empty"
    
    required_cols = {'timestamp', 'value'}
    if not required_cols.issubset(df.columns):
        return False, f"Missing required columns. Need: {required_cols}"
    
    if not pd.api.types.is_numeric_dtype(df['value']):
        return False, "Column 'value' must be numeric"
    
    return True, ""


def safe_division(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Perform safe division handling zeros.
    
    Args:
        numerator: Dividend array
        denominator: Divisor array
        
    Returns:
        Result of division, with zeros where denominator is zero
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(numerator, denominator)
        result[~np.isfinite(result)] = 0
    return result


# Performance metrics
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with precision, recall, F1, etc.
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    
    precision = safe_division(np.array([TP]), np.array([TP + FP]))[0]
    recall = safe_division(np.array([TP]), np.array([TP + FN]))[0]
    fpr = safe_division(np.array([FP]), np.array([FP + TN]))[0]
    f1 = safe_division(
        2 * np.array([precision * recall]),
        np.array([precision + recall])
    )[0]
    
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "false_positive_rate": float(fpr),
        "true_positive": int(TP),
        "false_positive": int(FP),
        "true_negative": int(TN),
        "false_negative": int(FN)
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: Ground truth
        y_pred: Predictions
        
    Returns:
        2x2 confusion matrix
    """
    cm = np.zeros((2, 2), dtype=int)
    cm[0, 0] = np.sum((y_true == 0) & (y_pred == 0))  # TN
    cm[0, 1] = np.sum((y_true == 0) & (y_pred == 1))  # FP
    cm[1, 0] = np.sum((y_true == 1) & (y_pred == 0))  # FN
    cm[1, 1] = np.sum((y_true == 1) & (y_pred == 1))  # TP
    return cm


# Time series utilities
def create_rolling_windows(
    data: np.ndarray,
    window_size: int,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create rolling windows from time series data.
    
    Args:
        data: 1D array of values
        window_size: Size of rolling window
        step: Step size for sliding window
        
    Returns:
        Tuple (windows, indices)
            windows: 2D array of shape (n_windows, window_size)
            indices: Center indices of each window
    """
    n = len(data)
    windows = []
    indices = []
    
    for i in range(0, n - window_size + 1, step):
        windows.append(data[i:i + window_size])
        indices.append(i + window_size // 2)
    
    return np.array(windows), np.array(indices)


def calculate_trend(data: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Calculate local trend using linear regression on windows.
    
    Args:
        data: Time series data
        window: Window size for trend calculation
        
    Returns:
        Array of trend slopes
    """
    trends = np.zeros(len(data))
    half_window = window // 2
    
    for i in range(half_window, len(data) - half_window):
        x = np.arange(window)
        y = data[i - half_window:i + half_window + 1]
        if len(y) == window:
            coeffs = np.polyfit(x, y, 1)
            trends[i] = coeffs[0]
    
    return trends


# Export utilities
def format_timestamp(ts: datetime) -> str:
    """Format datetime to ISO string."""
    return ts.strftime('%Y-%m-%d %H:%M:%S')


def get_memory_usage(df: pd.DataFrame) -> str:
    """Get human-readable memory usage of DataFrame."""
    mb = df.memory_usage(deep=True).sum() / 1024**2
    return f"{mb:.2f} MB"


if __name__ == "__main__":
    logger.info("Utils module loaded successfully")
