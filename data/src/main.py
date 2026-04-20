"""
Main Anomaly Detection Module - Integrare NAB cu Spark
Detecția anomaliilor în date big data folosind NAB datasets și algoritmi Spark
"""

import os
import sys
from pathlib import Path

# Configurare JAVA_HOME
if not os.environ.get("JAVA_HOME"):
    adoptium_dir = Path(r"C:\Program Files\Eclipse Adoptium")
    jdk_candidates = sorted(adoptium_dir.glob("jdk-*"), reverse=True) if adoptium_dir.exists() else []
    if jdk_candidates:
        os.environ["JAVA_HOME"] = str(jdk_candidates[0])

java_home = os.environ.get("JAVA_HOME")
if java_home:
    java_bin = str(Path(java_home) / "bin")
    if java_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{java_bin}{os.pathsep}{os.environ.get('PATH', '')}"

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, stddev, abs as spark_abs, window

# Adaugă calea pentru importuri locale
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nab_integration import NABIntegration
from spark_nab_detector import SparkAnomalyDetector


def run_simple_test():
    """Test simplu cu date sintetice"""
    print("\n" + "="*60)
    print("TEST 1: Spark Anomaly Detection (Synthetic Data)")
    print("="*60 + "\n")
    
    spark = SparkSession.builder \
        .appName("Licenta-SimpleTest") \
        .master("local[*]") \
        .getOrCreate()
    
    # Date sintetice
    df = spark.createDataFrame([
        (1, 10.0),
        (2, 12.0),
        (3, 100.0),  # Anomalie
        (4, 11.0),
        (5, 13.0),
        (6, 200.0),  # Anomalie
        (7, 14.0)
    ], ["timestamp", "value"])
    
    # Calcule statistici
    stats = df.select(
        avg("value").alias("mean"),
        stddev("value").alias("std")
    ).collect()[0]
    
    mean = stats["mean"]
    std = stats["std"]
    
    print(f" Statistici:")
    print(f"   Media: {mean:.2f}")
    print(f"   Deviație standard: {std:.2f}")
    print(f"   Prag anomalie (mean ± 1*std): {mean-std:.2f} - {mean+std:.2f}")
    
    # Detectare anomalii
    anomalies = df.filter(spark_abs(col("value") - mean) > 1 * std)
    
    print(f"\n Anomalii detectate:")
    anomalies.show()
    
    spark.stop()


def run_nab_test():
    """Test cu datele NAB"""
    print("\n" + "="*60)
    print("TEST 2: NAB Integration")
    print("="*60 + "\n")
    
    try:
        nab = NABIntegration()
        
        # Afișează informații NAB
        datasets = nab.get_available_datasets()
        print(f" NAB Integration Active")
        print(f"   Total datasets: {len(datasets)}")
        print(f"   Data directory: {nab.data_dir}")
        print(f"   Labels directory: {nab.labels_dir}")
        
        if datasets:
            print(f"\n Primele datasetsuri disponibile:")
            for i, dataset in enumerate(datasets[:5], 1):
                try:
                    stats = nab.get_dataset_statistics(dataset)
                    print(f"   {i}. {dataset}")
                    print(f"      - Rânduri: {stats['rows']}")
                    print(f"      - Media: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
                except:
                    print(f"   {i}. {dataset} (eroare la citire)")
        
        return True
    except Exception as e:
        print(f" Error: {e}")
        return False


def run_spark_nab_detector():
    """Rulează detectorul Spark + NAB"""
    print("\n" + "="*60)
    print("TEST 3: Spark + NAB Integrated Detector")
    print("="*60 + "\n")
    
    try:
        detector = SparkAnomalyDetector()
        
        datasets = detector.nab.get_available_datasets()
        if not datasets:
            print(" No datasets found!")
            return False
        
        # Testează pe primul dataset
        dataset_name = datasets[0]
        print(f" Testing on dataset: {dataset_name}\n")
        
        # Afișează statistici
        detector.get_statistics(dataset_name)
        
        print(f"\nAfișare primele anomalii detectate cu Z-score:")
        spark_df = detector.load_nab_data_to_spark(dataset_name)
        anomalies, mean, std = detector.detect_anomalies_zscore(spark_df, threshold=2.0)
        anomalies.show(5)
        
        return True
    except Exception as e:
        print(f"️  Note: {e}")
        print("   (Aceasta este așteptat dacă Spark nu este complet configurate)")
        return False


def main():
    """Funcția principală"""
    print("\n" + "="*70)
    print(" "*15 + "NAB + SPARK INTEGRATED SYSTEM")
    print(" "*10 + "Anomaly Detection for Big Data")
    print("="*70)
    
    # TEST 1: Spark test simplu
    try:
        run_simple_test()
    except Exception as e:
        print(f" Error în TEST 1: {e}")
    
    # TEST 2: NAB Integration test
    nab_ok = run_nab_test()
    
    # TEST 3: Spark + NAB combined (opțional)
    if nab_ok:
        try:
            run_spark_nab_detector()
        except Exception as e:
            print(f"️  Spark+NAB test skipped: {e}")
    
    print("\n" + "="*70)
    print(" Testele s-au completat!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()