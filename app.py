"""
Main Application Entry Point

Provides command-line interface and initialization for the
Anomaly Detection System.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from data_loader import DataLoader
from anomaly_algorithms import AnomalyDetectionEngine
from spark_engine import SparkEngine
from benchmark import Benchmark
from utils import setup_logger


logger = setup_logger(__name__)


def print_banner():
    """Print application banner."""
    banner = """
    ╔═════════════════════════════════════════════════════════════╗
    ║                                                             ║
    ║   ADVANCED ANOMALY DETECTION SYSTEM FOR TIME SERIES DATA   ║
    ║                                                             ║
    ║   Scalable | Distributed | Production-Ready               ║
    ║                                                             ║
    ╚═════════════════════════════════════════════════════════════╝
    """
    print(banner)


def detect_command(args):
    """Run anomaly detection on a dataset."""
    print(f"\n Loading data from: {args.input}")
    
    loader = DataLoader()
    try:
        df = loader.preprocess(
            args.input,
            normalize=args.normalize,
            normalization_method=args.norm_method
        )
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        return
    
    values = df['value'].values
    print(f"Loaded {len(df)} data points")
    
    # Initialize detection engine
    engine = AnomalyDetectionEngine()
    
    print(f"\nRunning {args.algorithm} detection...")
    
    # Run detection
    try:
        if args.algorithm == 'all':
            results = engine.detect_all(values)
        else:
            results = {args.algorithm: engine.detect_single(args.algorithm, values)}
    except Exception as e:
        print(f"Error during detection: {e}")
        return
    
    # Display results
    print("\nRESULTS:")
    print("=" * 70)
    
    for algo_name, result in results.items():
        n_anomalies = int(result.labels.sum())
        pct = 100 * n_anomalies / len(values)
        
        print(f"\n{algo_name}:")
        print(f"  Anomalies detected: {n_anomalies} ({pct:.2f}%)")
        print(f"  Min score: {result.scores.min():.3f}")
        print(f"  Max score: {result.scores.max():.3f}")
        print(f"  Mean score: {result.scores.mean():.3f}")
    
    # Save results if requested
    if args.output:
        print(f"\n Saving results to: {args.output}")
        
        df_results = df.copy()
        for algo_name, result in results.items():
            df_results[f'{algo_name}_anomaly'] = result.labels
            df_results[f'{algo_name}_score'] = result.scores
        
        df_results.to_csv(args.output, index=False)
        print("Results saved")


def benchmark_command(args):
    """Run benchmarking tests."""
    print("\nBENCHMARKING MODE")
    print("=" * 70)
    
    loader = DataLoader()
    
    try:
        df = loader.preprocess(args.input)
    except FileNotFoundError:
        print(f" Error: File not found: {args.input}")
        return
    
    print(f" Loaded {len(df)} data points")
    
    # Run all algorithms
    print("\n Running anomaly detection algorithms...")
    
    engine = AnomalyDetectionEngine()
    values = df['value'].values
    
    start_times = {}
    results = {}
    
    import time
    for algo_name, detector in engine.detectors.items():
        start = time.time()
        result = detector.detect(values)
        elapsed = time.time() - start
        
        results[algo_name] = result
        start_times[algo_name] = elapsed
        
        print(f"   {algo_name}: {elapsed:.3f}s")
    
    # Display summary
    print("\n ALGORITHM COMPARISON:")
    print("-" * 70)
    print(f"{'Algorithm':<25} {'Anomalies':<12} {'Score Mean':<15} {'Time':<10}")
    print("-" * 70)
    
    for algo_name, result in results.items():
        n_anomalies = int(result.labels.sum())
        mean_score = result.scores.mean()
        exec_time = start_times[algo_name]
        print(f"{algo_name:<25} {n_anomalies:<12} {mean_score:<15.3f} {exec_time:<10.3f}s")
    
    # Benchmark Spark if requested
    if args.spark:
        print("\n SPARK DISTRIBUTED PROCESSING")
        print("-" * 70)
        
        spark_engine = SparkEngine(master=f"local[{args.cores}]")
        spark_engine.set_log_level("ERROR")
        
        print(f"Running on {args.cores} cores...")
        
        try:
            sdf = spark_engine.load_pandas_as_spark(df)
            sdf = spark_engine.process_timestamps(sdf)
            
            results_spark = spark_engine.to_pandas(sdf)
            
            exec_times = spark_engine.get_execution_times()
            print("\nSpark execution times:")
            for op, t in exec_times.items():
                print(f"  {op}: {t:.3f}s")
        
        except Exception as e:
            print(f"Error with Spark: {e}")
        finally:
            spark_engine.stop()


def dashboard_command(args):
    """Launch Streamlit dashboard."""
    print("\nLaunching Streamlit Dashboard...")
    print("=" * 70)
    
    import subprocess
    
    dashboard_path = Path(__file__).parent / "dashboard.py"
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--logger.level=error"
        ])
    except Exception as e:
        print(f"Error launching dashboard: {e}")


def list_datasets_command(args):
    """List available NAB datasets."""
    print("\n Available NAB Datasets:")
    print("=" * 70)
    
    loader = DataLoader()
    datasets = loader.list_nab_datasets()
    
    if not datasets:
        print("No datasets found. Ensure NAB directory is in current location.")
        return
    
    for i, dataset in enumerate(datasets, 1):
        print(f"{i:3d}. {dataset}")
    
    print(f"\nTotal: {len(datasets)} datasets")


def info_command(args):
    """Display system information and examples."""
    banner = """
    
    ╔════════════════════════════════════════════════════════════════╗
    ║          ANOMALY DETECTION SYSTEM - USAGE GUIDE               ║
    ╚════════════════════════════════════════════════════════════════╝
    
    QUICK START:
    
    1. LIST AVAILABLE DATASETS:
       python app.py list-datasets
    
    2. RUN ANOMALY DETECTION:
       python app.py detect --input art_daily_small_noise \\
                    --algorithm rolling_stats \\
                    --output results.csv
    
    3. BENCHMARK ALGORITHMS:
       python app.py benchmark --input art_daily_small_noise \\
                      --spark --cores 4
    
    4. LAUNCH INTERACTIVE DASHBOARD:
       python app.py dashboard
    
    ALGORITHMS:
    - rolling_stats      : Rolling mean/std with Z-score
    - prediction_error   : Moving average forecast error
    - hybrid            : Combines multiple signals
    - all               : Run all algorithms
    
    OPTIONS:
    --input             : Input CSV file or NAB dataset name
    --output            : Output CSV file for results
    --algorithm         : Algorithm to use (default: rolling_stats)
    --normalize         : Normalize data (default: True)
    --spark             : Use Spark for processing
    --cores             : Number of cores (1, 2, 4, 8)
    
    EXAMPLES:
    
    # Detect on uploaded CSV
    python app.py detect --input mydata.csv --algorithm hybrid
    
    # Detect on NAB dataset
    python app.py detect --input aws_ec2_cpu_utilization_1 \\
                 --algorithm all
    
    # Benchmark with Spark
    python app.py benchmark --input data.csv --spark --cores 8
    
    # Launch dashboard
    python app.py dashboard
    
    """
    print(banner)


def main():
    """Main application entry point."""
    print_banner()
    
    parser = argparse.ArgumentParser(
        description="Advanced Anomaly Detection System for Time Series Data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Run anomaly detection')
    detect_parser.add_argument('--input', required=True, help='Input CSV file or dataset name')
    detect_parser.add_argument('--output', help='Output CSV file for results')
    detect_parser.add_argument(
        '--algorithm',
        choices=['rolling_stats', 'prediction_error', 'hybrid', 'all'],
        default='rolling_stats',
        help='Algorithm to use'
    )
    detect_parser.add_argument(
        '--normalize',
        action='store_true',
        default=True,
        help='Normalize data'
    )
    detect_parser.add_argument(
        '--norm-method',
        choices=['standard', 'minmax'],
        default='standard',
        help='Normalization method'
    )
    detect_parser.set_defaults(func=detect_command)
    
    # Benchmark command
    bench_parser = subparsers.add_parser('benchmark', help='Run benchmarking')
    bench_parser.add_argument('--input', required=True, help='Input CSV file')
    bench_parser.add_argument(
        '--spark',
        action='store_true',
        help='Use Spark for processing'
    )
    bench_parser.add_argument(
        '--cores',
        type=int,
        choices=[1, 2, 4, 8],
        default=4,
        help='Number of cores'
    )
    bench_parser.set_defaults(func=benchmark_command)
    
    # Dashboard command
    dash_parser = subparsers.add_parser('dashboard', help='Launch interactive dashboard')
    dash_parser.set_defaults(func=dashboard_command)
    
    # List datasets command
    list_parser = subparsers.add_parser('list-datasets', help='List available NAB datasets')
    list_parser.set_defaults(func=list_datasets_command)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display system information')
    info_parser.set_defaults(func=info_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        info_command(args)
    else:
        try:
            args.func(args)
        except Exception as e:
            logger.exception(f"Error executing command: {e}")
            print(f"\nError: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()

