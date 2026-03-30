"""
NAB + Spark Integration - Advanced Anomaly Detection Framework
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Configurare JAVA_HOME
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, avg, stddev, abs as spark_abs, 
    window, count, struct, coalesce
)
from pyspark.sql.types import StructType, StructField, TimestampType, DoubleType

# Importă modulul NAB
sys.path.insert(0, str(Path(__file__).parent))
from nab_integration import NABIntegration


class SparkAnomalyDetector:
    """Detector anomalii folosind Spark + NAB"""
    
    def __init__(self, app_name: str = "NAB-SparkDetector"):
        """Inițializează Spark session"""
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .master("local[*]") \
            .config("spark.sql.adaptive.enabled", "true") \
            .getOrCreate()
        
        self.nab = NABIntegration()
    
    def load_nab_data_to_spark(self, dataset_name: str):
        """
        Încarcă datele NAB în Spark DataFrame
        
        Args:
            dataset_name: Numele setului de date
            
        Returns:
            Spark DataFrame
        """
        # Încarcă datele cu pandas
        data = self.nab.load_data(dataset_name)
        
        # Convertește la Spark DataFrame
        spark_df = self.spark.createDataFrame(
            data,
            schema=["timestamp", "value"]
        )
        
        # Convertește timestamp la formato proeper
        spark_df = spark_df.withColumn(
            "timestamp",
            col("timestamp").cast(TimestampType())
        ).withColumn(
            "value",
            col("value").cast(DoubleType())
        )
        
        return spark_df.orderBy("timestamp")
    
    def detect_anomalies_zscore(self, spark_df, threshold: float = 2.0):
        """
        Detectează anomalii folosind Z-score
        
        Args:
            spark_df: Spark DataFrame cu coloane timestamp, value
            threshold: Prag Z-score (default: 2.0)
            
        Returns:
            DataFrame cu anomalii detectate
        """
        # Calculează statistici globale
        stats = spark_df.select(
            avg("value").alias("mean"),
            stddev("value").alias("std")
        ).collect()[0]
        
        mean = stats["mean"]
        std = stats["std"]
        
        # Detectează anomalii
        anomalies = spark_df.withColumn(
            "zscore",
            spark_abs((col("value") - mean) / std)
        ).withColumn(
            "is_anomaly",
            col("zscore") > threshold
        ).filter(
            col("is_anomaly") == True
        )
        
        return anomalies, mean, std
    
    def detect_anomalies_iqr(self, spark_df, multiplier: float = 1.5):
        """
        Detectează anomalii folosind Interquartile Range (IQR)
        
        Args:
            spark_df: Spark DataFrame
            multiplier: Multiplicator pentru IQR (default: 1.5)
            
        Returns:
            DataFrame cu anomalii
        """
        # Calculează quartiles
        quartiles = spark_df.selectExpr(
            "percentile_approx(value, 0.25) as q1",
            "percentile_approx(value, 0.75) as q3"
        ).collect()[0]
        
        q1 = quartiles["q1"]
        q3 = quartiles["q3"]
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        # Detectează anomalii
        anomalies = spark_df.filter(
            (col("value") < lower_bound) | (col("value") > upper_bound)
        )
        
        return anomalies, q1, q3, iqr
    
    def detect_anomalies_windowed(self, spark_df, window_size: str = "5 minutes", 
                                    method: str = "zscore", threshold: float = 2.0):
        """
        Detectează anomalii pe ferestre de timp
        
        Args:
            spark_df: Spark DataFrame
            window_size: Mărimea ferestrei (ex: "5 minutes", "1 hour")
            method: Metodă de detecție ("zscore" sau "iqr")
            threshold: Prag pentru Z-score
            
        Returns:
            DataFrame cu anomalii pe ferestre
        """
        # Aplică window function
        windowed = spark_df.withColumn(
            "time_window",
            window(col("timestamp"), window_size)
        )
        
        if method == "zscore":
            # Calculează statistici pe fereastra
            stats = windowed.groupBy("time_window").agg(
                avg("value").alias("window_mean"),
                stddev("value").alias("window_std"),
                count("value").alias("count")
            )
            
            # Join înapoi pentru a compara
            result = windowed.join(
                stats,
                on="time_window"
            ).withColumn(
                "zscore",
                spark_abs((col("value") - col("window_mean")) / col("window_std"))
            ).filter(
                col("zscore") > threshold
            )
        else:  # IQR
            result = windowed  # Implementare simplificată
        
        return result.select("timestamp", "value", "time_window", "zscore")
    
    def compare_methods(self, dataset_name: str):
        """
        Compară mai multe metode de detecție
        
        Args:
            dataset_name: Numele setului de date
            
        Returns:
            Dicționar cu rezultate
        """
        print(f"\n{'='*60}")
        print(f"Compararea metodelor de detecție: {dataset_name}")
        print(f"{'='*60}\n")
        
        # Încarcă datele
        spark_df = self.load_nab_data_to_spark(dataset_name)
        
        results = {}
        
        # Z-score
        print("📊 Z-Score Method (threshold=2.0):")
        anomalies_zscore, mean, std = self.detect_anomalies_zscore(spark_df, threshold=2.0)
        count_zscore = anomalies_zscore.count()
        results["zscore"] = count_zscore
        print(f"   Anomalii detectate: {count_zscore}")
        print(f"   Media: {mean:.2f}, Std: {std:.2f}")
        
        # IQR
        print("\n📊 IQR Method (multiplier=1.5):")
        anomalies_iqr, q1, q3, iqr = self.detect_anomalies_iqr(spark_df, multiplier=1.5)
        count_iqr = anomalies_iqr.count()
        results["iqr"] = count_iqr
        print(f"   Anomalii detectate: {count_iqr}")
        print(f"   Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
        
        # Windowed Z-score
        print("\n📊 Windowed Z-Score (5-minute windows):")
        anomalies_windowed = self.detect_anomalies_windowed(
            spark_df, 
            window_size="5 minutes", 
            method="zscore",
            threshold=2.0
        )
        count_windowed = anomalies_windowed.count()
        results["windowed"] = count_windowed
        print(f"   Anomalii detectate: {count_windowed}")
        
        print(f"\n{'='*60}\n")
        
        return results
    
    def get_statistics(self, dataset_name: str):
        """Afișează statistici despre dataset"""
        stats = self.nab.get_dataset_statistics(dataset_name)
        print(f"\n📈 Statistici {dataset_name}:")
        for key, value in stats.items():
            print(f"   {key}: {value}")


def main():
    """Funcția principală"""
    print("\n" + "="*60)
    print("NAB + Spark Integration - Anomaly Detection")
    print("="*60)
    
    # Inițializează detectorul
    detector = SparkAnomalyDetector()
    
    # Listează datasetsuri disponibile
    datasets = detector.nab.get_available_datasets()
    
    if not datasets:
        print("❌ No datasets found in NAB!")
        return
    
    print(f"\n✅ Found {len(datasets)} datasets")
    
    # Testează pe primele 2 datasetsuri
    for dataset_name in datasets[:2]:
        try:
            detector.get_statistics(dataset_name)
            detector.compare_methods(dataset_name)
        except Exception as e:
            print(f"⚠️  Error processing {dataset_name}: {e}")
    
    print("\n✅ Detecția anomaliilor s-a completat!")


if __name__ == "__main__":
    main()
