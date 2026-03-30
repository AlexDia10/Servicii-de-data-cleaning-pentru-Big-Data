"""
Spark Engine for Distributed Time Series Processing

Provides distributed data loading, processing, and anomaly detection
using Apache Spark and PySpark.
"""

import time
import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, TimestampType, DoubleType
)

from utils import setup_logger


logger = setup_logger(__name__)


class SparkEngine:
    """
    Distributed processing engine for time series anomaly detection.
    
    Features:
    - Distributed CSV loading
    - Parallel data transformations
    - Configurable number of cores
    - Performance benchmarking
    """
    
    def __init__(
        self,
        app_name: str = "AnomalyDetection",
        master: str = "local[*]",
        memory: str = "4g"
    ):
        """
        Initialize Spark session.
        
        Args:
            app_name: Spark application name
            master: Master URL (local[1], local[2], local[4], local[8], etc.)
            memory: Executor memory allocation
        """
        self.app_name = app_name
        self.master = master
        self.memory = memory
        self.spark = None
        self.sc = None
        self.execution_times = {}
        
        self._initialize_spark()
    
    def _initialize_spark(self):
        """Initialize Spark session with configuration."""
        try:
            self.spark = SparkSession.builder \
                .appName(self.app_name) \
                .master(self.master) \
                .config("spark.executor.memory", self.memory) \
                .config("spark.driver.memory", self.memory) \
                .config("spark.sql.shuffle.partitions", "8") \
                .config("spark.default.parallelism", "8") \
                .config("spark.sql.adaptive.enabled", "true") \
                .getOrCreate()
            
            self.sc = self.spark.sparkContext
            
            logger.info(
                f"Spark session initialized: {self.master}, "
                f"memory: {self.memory}"
            )
        except Exception as e:
            logger.error(f"Error initializing Spark: {e}")
            raise
    
    def set_log_level(self, level: str = "WARN"):
        """Set Spark logging level."""
        if self.spark:
            self.spark.sparkContext.setLogLevel(level)
    
    def load_csv(self, filepath: str) -> DataFrame:
        """
        Load CSV file as Spark DataFrame.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Spark DataFrame
        """
        start_time = time.time()
        
        try:
            df = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv(filepath)
            
            elapsed = time.time() - start_time
            self.execution_times['load_csv'] = elapsed
            
            logger.info(
                f"Loaded CSV in {elapsed:.3f}s: {df.count()} rows, "
                f"{len(df.columns)} columns"
            )
            
            return df
        
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def load_pandas_as_spark(self, pdf: pd.DataFrame) -> DataFrame:
        """
        Convert Pandas DataFrame to Spark DataFrame.
        
        Args:
            pdf: Pandas DataFrame
            
        Returns:
            Spark DataFrame
        """
        start_time = time.time()
        
        try:
            df = self.spark.createDataFrame(pdf)
            elapsed = time.time() - start_time
            self.execution_times['load_pandas'] = elapsed
            
            logger.info(f"Converted pandas DataFrame to Spark in {elapsed:.3f}s")
            return df
        
        except Exception as e:
            logger.error(f"Error converting DataFrame: {e}")
            raise
    
    def process_timestamps(self, df: DataFrame) -> DataFrame:
        """
        Parse and process timestamp column.
        
        Args:
            df: Spark DataFrame
            
        Returns:
            DataFrame with parsed timestamps
        """
        start_time = time.time()
        
        df = df.withColumn(
            "timestamp",
            F.to_timestamp(F.col("timestamp"))
        )
        
        df = df.sort("timestamp")
        
        elapsed = time.time() - start_time
        self.execution_times['process_timestamps'] = elapsed
        
        logger.info(f"Processed timestamps in {elapsed:.3f}s")
        return df
    
    def normalize_values(
        self,
        df: DataFrame,
        value_col: str = "value"
    ) -> Tuple[DataFrame, float, float]:
        """
        Normalize values using min-max scaling.
        
        Args:
            df: Spark DataFrame
            value_col: Column name to normalize
            
        Returns:
            Tuple (normalized_df, min_val, max_val)
        """
        start_time = time.time()
        
        # Get min and max
        stats = df.agg(
            F.min(value_col).alias("min_val"),
            F.max(value_col).alias("max_val")
        ).collect()[0]
        
        min_val = float(stats["min_val"])
        max_val = float(stats["max_val"])
        
        # Normalize
        if max_val == min_val:
            df = df.withColumn(f"{value_col}_normalized", F.lit(0.0))
        else:
            df = df.withColumn(
                f"{value_col}_normalized",
                (F.col(value_col) - min_val) / (max_val - min_val)
            )
        
        elapsed = time.time() - start_time
        self.execution_times['normalize'] = elapsed
        
        logger.info(f"Normalized values in {elapsed:.3f}s")
        return df, min_val, max_val
    
    def create_sliding_windows(
        self,
        df: DataFrame,
        window_size: int = 10,
        value_col: str = "value"
    ) -> DataFrame:
        """
        Create sliding windows using Spark SQL window functions.
        
        Args:
            df: Spark DataFrame with timestamp and value columns
            window_size: Size of sliding window
            value_col: Column name for values
            
        Returns:
            DataFrame with window features
        """
        start_time = time.time()
        
        from pyspark.sql.window import Window
        
        # Create row number for indexing
        w = Window.orderBy("timestamp")
        df = df.withColumn("row_num", F.row_number().over(w))
        
        # Calculate window statistics
        window_spec = Window.orderBy("row_num").rangeBetween(
            -(window_size - 1), 0
        )
        
        df = df.withColumn(
            f"window_mean_{window_size}",
            F.avg(value_col).over(window_spec)
        ).withColumn(
            f"window_std_{window_size}",
            F.stddev(value_col).over(window_spec)
        )
        
        elapsed = time.time() - start_time
        self.execution_times['create_windows'] = elapsed
        
        logger.info(f"Created sliding windows in {elapsed:.3f}s")
        return df
    
    def detect_anomalies_zscore(
        self,
        df: DataFrame,
        threshold: float = 2.5,
        value_col: str = "value"
    ) -> DataFrame:
        """
        Detect anomalies using Z-score in Spark.
        
        Args:
            df: Spark DataFrame
            threshold: Z-score threshold
            value_col: Column name for values
            
        Returns:
            DataFrame with anomaly labels and scores
        """
        start_time = time.time()
        
        # Calculate global statistics
        stats = df.agg(
            F.avg(value_col).alias("mean"),
            F.stddev(value_col).alias("std")
        ).collect()[0]
        
        mean = float(stats["mean"])
        std = float(stats["std"])
        
        # Calculate z-scores
        if std > 0:
            df = df.withColumn(
                "zscore",
                F.abs((F.col(value_col) - mean) / std)
            )
        else:
            df = df.withColumn("zscore", F.lit(0.0))
        
        # Create anomaly labels and scores
        df = df.withColumn(
            "anomaly_score",
            F.least(F.col("zscore") / threshold, F.lit(1.0))
        ).withColumn(
            "is_anomaly",
            F.when(F.col("zscore") > threshold, 1).otherwise(0)
        )
        
        elapsed = time.time() - start_time
        self.execution_times['detect_anomalies_zscore'] = elapsed
        
        logger.info(f"Detected anomalies (Z-score) in {elapsed:.3f}s")
        return df
    
    def to_pandas(self, df: DataFrame) -> pd.DataFrame:
        """
        Convert Spark DataFrame to Pandas.
        
        Args:
            df: Spark DataFrame
            
        Returns:
            Pandas DataFrame
        """
        start_time = time.time()
        
        pdf = df.toPandas()
        
        elapsed = time.time() - start_time
        self.execution_times['to_pandas'] = elapsed
        
        logger.info(
            f"Converted Spark DataFrame to Pandas in {elapsed:.3f}s: "
            f"{len(pdf)} rows"
        )
        return pdf
    
    def get_execution_times(self) -> Dict[str, float]:
        """
        Get all recorded execution times.
        
        Returns:
            Dictionary mapping operation names to times in seconds
        """
        return self.execution_times.copy()
    
    def benchmark_scaling(
        self,
        data: pd.DataFrame,
        cores_list: List[int] = None,
        operation: str = "zscore_detection"
    ) -> Dict[int, float]:
        """
        Benchmark performance across different core counts.
        
        Args:
            data: Pandas DataFrame with data
            cores_list: List of core counts to test
            operation: Operation to benchmark
            
        Returns:
            Dictionary mapping core counts to execution times
        """
        if cores_list is None:
            cores_list = [1, 2, 4, 8]
        
        results = {}
        
        for cores in cores_list:
            # Stop current Spark session
            if self.spark:
                self.spark.stop()
            
            # Create new session with different core count
            self.master = f"local[{cores}]"
            self._initialize_spark()
            self.set_log_level("ERROR")
            
            logger.info(f"Benchmarking with {cores} cores")
            
            try:
                # Load and process data
                sdf = self.load_pandas_as_spark(data)
                
                if operation == "zscore_detection":
                    start = time.time()
                    sdf = self.detect_anomalies_zscore(sdf)
                    sdf.collect()  # Force execution
                    elapsed = time.time() - start
                
                results[cores] = elapsed
                logger.info(f"{cores} cores: {elapsed:.3f}s")
            
            except Exception as e:
                logger.error(f"Error benchmarking {cores} cores: {e}")
                results[cores] = None
        
        return results
    
    def stop(self):
        """Stop Spark session."""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session stopped")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


if __name__ == "__main__":
    # Example usage
    engine = SparkEngine(master="local[4]")
    
    # Create sample data
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100),
        'value': np.random.randn(100) * 10 + 50
    })
    
    # Process
    sdf = engine.load_pandas_as_spark(data)
    sdf = engine.process_timestamps(sdf)
    sdf = engine.detect_anomalies_zscore(sdf)
    
    result = engine.to_pandas(sdf)
    print(f"Anomalies found: {result['is_anomaly'].sum()}")
    print(f"Execution times: {engine.get_execution_times()}")
    
    engine.stop()
