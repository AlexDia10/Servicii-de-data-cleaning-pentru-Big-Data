"""
Data Loader Module for Time Series Anomaly Detection

Handles loading, parsing, and preprocessing of CSV time series data.
Supports both NAB datasets and custom uploads.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging
from datetime import datetime

from utils import setup_logger, validate_dataframe, get_memory_usage


logger = setup_logger(__name__)


class DataLoader:
    """
    Load and preprocess time series data from CSV files.
    
    Handles:
    - CSV file loading
    - Timestamp parsing
    - Missing value handling
    - Data normalization/scaling
    - Rolling window generation
    """
    
    def __init__(self, nab_root: Optional[str] = None):
        """
        Initialize DataLoader with optional NAB root directory.
        
        Args:
            nab_root: Path to NAB root directory (default: ./NAB)
        """
        self.nab_root = Path(nab_root or "NAB")
        self.data_dir = self.nab_root / "data"
        self.raw_data = None
        self.processed_data = None
        self.scaler = None
        self.scaler_type = None
        
        logger.info(f"DataLoader initialized with NAB root: {self.nab_root}")
    
    def list_nab_datasets(self) -> List[str]:
        """
        List all available NAB datasets with relative paths.
        
        Returns:
            List of dataset paths: dossier/dataset_name (without .csv)
        """
        if not self.data_dir.exists():
            logger.warning(f"NAB data directory not found: {self.data_dir}")
            return []
        
        datasets = {}
        # Subdirectories
        for subdir in self.data_dir.iterdir():
            if subdir.is_dir():
                for csv_file in subdir.glob("*.csv"):
                    relative_path = f"{subdir.name}/{csv_file.stem}"
                    datasets[relative_path] = relative_path
        
        # Root directory
        for csv_file in self.data_dir.glob("*.csv"):
            datasets[csv_file.stem] = csv_file.stem
        
        logger.info(f"Found {len(datasets)} NAB datasets")
        return sorted(datasets.values())
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load CSV file with automatic type inference.
        
        Args:
            filepath: Path to CSV file (can be relative like "realKnownCause/nyc_taxi")
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV format is invalid
        """
        file_path = Path(filepath)
        
        # If filepath not absolute, try to find in NAB data directory
        if not file_path.is_absolute():
            if "/" in filepath:
                # Relative path with subdirectory
                candidate = self.data_dir / f"{filepath}.csv"
                if candidate.exists():
                    file_path = candidate
            else:
                # Just filename, search subdirectories
                if self.data_dir.exists():
                    for subdir in self.data_dir.iterdir():
                        if subdir.is_dir():
                            candidate = subdir / f"{filepath}.csv"
                            if candidate.exists():
                                file_path = candidate
                                break
        
        # Try root directory if still not found
        if not file_path.exists() and not file_path.is_absolute():
            candidate = self.data_dir / f"{filepath}.csv"
            if candidate.exists():
                file_path = candidate
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        logger.info(f"Loading CSV: {file_path}")
        
        try:
            # Try to load with automatic type inference
            df = pd.read_csv(file_path)
            
            # Validate structure
            is_valid, error_msg = validate_dataframe(df)
            if not is_valid:
                raise ValueError(f"Invalid CSV structure: {error_msg}")
            
            logger.info(f"Loaded {len(df)} rows, memory: {get_memory_usage(df)}")
            self.raw_data = df.copy()
            return self.raw_data
        
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def parse_timestamps(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Parse timestamp column to datetime format.
        
        Args:
            df: DataFrame (uses self.raw_data if None)
            
        Returns:
            DataFrame with parsed timestamps
        """
        if df is None:
            if self.raw_data is None:
                raise ValueError("No data loaded. Call load_csv first.")
            df = self.raw_data.copy()
        
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame missing 'timestamp' column")
        
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info(f"Parsed timestamps: {df['timestamp'].min()} to {df['timestamp'].max()}")
            return df
        except Exception as e:
            logger.error(f"Error parsing timestamps: {e}")
            raise
    
    def sort_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Sort data by timestamp.
        
        Args:
            df: DataFrame (uses self.raw_data if None)
            
        Returns:
            Sorted DataFrame
        """
        if df is None:
            if self.raw_data is None:
                raise ValueError("No data loaded")
            df = self.raw_data.copy()
        
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame missing 'timestamp' column")
        
        df = df.sort_values('timestamp').reset_index(drop=True)
        logger.info("Data sorted by timestamp")
        return df
    
    def handle_missing_values(
        self,
        df: Optional[pd.DataFrame] = None,
        method: str = 'interpolate'
    ) -> pd.DataFrame:
        """
        Handle missing values in data.
        
        Args:
            df: DataFrame (uses self.raw_data if None)
            method: 'interpolate', 'forward_fill', 'drop', or 'mean'
            
        Returns:
            DataFrame with missing values handled
        """
        if df is None:
            if self.raw_data is None:
                raise ValueError("No data loaded")
            df = self.raw_data.copy()
        
        n_missing = df['value'].isna().sum()
        
        if n_missing == 0:
            logger.info("No missing values found")
            return df
        
        logger.info(f"Handling {n_missing} missing values using {method}")
        
        if method == 'interpolate':
            df['value'] = df['value'].interpolate(method='linear')
            df['value'] = df['value'].fillna(method='bfill').fillna(method='ffill')
        
        elif method == 'forward_fill':
            df['value'] = df['value'].fillna(method='ffill').fillna(method='bfill')
        
        elif method == 'mean':
            df['value'] = df['value'].fillna(df['value'].mean())
        
        elif method == 'drop':
            df = df.dropna(subset=['value'])
        
        return df
    
    def normalize_data(
        self,
        df: Optional[pd.DataFrame] = None,
        method: str = 'standard'
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Normalize/scale the data.
        
        Args:
            df: DataFrame (uses self.raw_data if None)
            method: 'standard' (z-score) or 'minmax' (0-1)
            
        Returns:
            Tuple (normalized_df, scaler_object)
        """
        if df is None:
            if self.raw_data is None:
                raise ValueError("No data loaded")
            df = self.raw_data.copy()
        
        logger.info(f"Normalizing data using {method} method")
        
        if method == 'standard':
            self.scaler = StandardScaler()
            self.scaler_type = 'standard'
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
            self.scaler_type = 'minmax'
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        values = df['value'].values.reshape(-1, 1)
        normalized = self.scaler.fit_transform(values)
        
        df['value_normalized'] = normalized.flatten()
        self.processed_data = df
        
        return df, self.scaler
    
    def denormalize_data(self, values: np.ndarray) -> np.ndarray:
        """
        Reverse normalization on predicted values.
        
        Args:
            values: Normalized values
            
        Returns:
            Denormalized values
        """
        if self.scaler is None:
            raise ValueError("No scaler available. Call normalize_data first.")
        
        return self.scaler.inverse_transform(values.reshape(-1, 1)).flatten()
    
    def create_rolling_windows(
        self,
        df: Optional[pd.DataFrame] = None,
        window_size: int = 10,
        step: int = 1,
        use_normalized: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create rolling windows from time series.
        
        Args:
            df: DataFrame (uses self.processed_data if None)
            window_size: Size of each window
            step: Step size for sliding window
            use_normalized: Use normalized values if available
            
        Returns:
            Tuple (windows: shape(n_windows, window_size), indices)
        """
        if df is None:
            df = self.processed_data if self.processed_data is not None else self.raw_data
            if df is None:
                raise ValueError("No data available")
        
        value_col = 'value_normalized' if use_normalized and 'value_normalized' in df.columns else 'value'
        data = df[value_col].values
        
        n = len(data)
        windows = []
        indices = []
        
        for i in range(0, n - window_size + 1, step):
            windows.append(data[i:i + window_size])
            indices.append(i + window_size // 2)
        
        windows = np.array(windows)
        indices = np.array(indices)
        
        logger.info(f"Created {len(windows)} rolling windows of size {window_size}")
        return windows, indices
    
    def preprocess(
        self,
        filepath: str,
        normalize: bool = True,
        normalization_method: str = 'standard',
        missing_value_method: str = 'interpolate'
    ) -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            filepath: Path to CSV file
            normalize: Whether to normalize data
            normalization_method: 'standard' or 'minmax'
            missing_value_method: Method for handling missing values
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting preprocessing pipeline")
        
        # Load and parse
        df = self.load_csv(filepath)
        df = self.parse_timestamps(df)
        df = self.sort_data(df)
        
        # Handle missing values
        df = self.handle_missing_values(df, method=missing_value_method)
        
        # Normalize
        if normalize:
            df, _ = self.normalize_data(df, method=normalization_method)
        
        self.processed_data = df
        logger.info("Preprocessing completed successfully")
        
        return df
    
    def get_statistics(self, df: Optional[pd.DataFrame] = None) -> dict:
        """
        Get basic statistics about the data.
        
        Args:
            df: DataFrame (uses self.processed_data if None)
            
        Returns:
            Dictionary with statistics
        """
        if df is None:
            df = self.processed_data if self.processed_data is not None else self.raw_data
            if df is None:
                raise ValueError("No data available")
        
        values = df['value'].values
        
        return {
            'count': len(df),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'skewness': float(pd.Series(values).skew()),
            'kurtosis': float(pd.Series(values).kurtosis())
        }


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    datasets = loader.list_nab_datasets()
    print(f"Found datasets: {datasets[:5]}")
