"""
Automated evaluation report: Test anomaly algorithms on key NAB datasets.
Generates comprehensive F1 scores and comparison metrics across multiple datasets.
"""

import pandas as pd
import numpy as np
from data_loader import DataLoader
from nab_integration import NABIntegration
from anomaly_algorithms import AnomalyDetectionEngine
from benchmark import Benchmark
import os
import json
from datetime import datetime
from pathlib import Path


class EvaluationReport:
    """Generate comprehensive evaluation metrics across multiple datasets."""
    
    # KEY DATASETS FOR THESIS: 3 representative NAB datasets
    KEY_DATASETS = [
        {
            'name': 'RDS CPU Utilization',
            'path': 'realAWSCloudwatch/rds_cpu_utilization_cc0c53',
            'type': 'AWS CloudWatch',
            'description': 'CPU spikes in AWS RDS database - production anomalies'
        },
        {
            'name': 'NYC Taxi Volume',
            'path': 'realKnownCause/nyc_taxi',
            'type': 'Real Known Cause',
            'description': 'NYC taxi volume anomalies - known external causes'
        },
        {
            'name': 'EC2 CPU Utilization',
            'path': 'realAWSCloudwatch/ec2_cpu_utilization_24ae8d',
            'type': 'AWS CloudWatch',
            'description': 'EC2 instance CPU spikes - infrastructure anomalies'
        },
    ]
    
    # ALGORITHM PRESETS (for "Simple Mode" - recommended defaults)
    ALGORITHM_PRESETS = {
        'rolling_stats': {
            'name': 'Rolling Statistics Z-Score',
            'window_size': 20,
            'threshold': 4.0,  # Extreme stringency (reduced false positives)
            'description': 'Detects extreme deviations from rolling mean'
        },
        'prediction_error': {
            'name': 'Prediction Error Anomaly',
            'forecast_window': 5,
            'threshold': 4.5,  # High threshold (reduced false positives)
            'description': 'Detects forecast errors above moving average'
        },
        'hybrid': {
            'name': 'Hybrid Score (RECOMMENDED)',
            'zscore_threshold': 3.5,
            'trend_threshold': 1.2,
            'volatility_threshold': 3.5,
            'weights': [0.5, 0.3, 0.2],
            'label_threshold': 0.80,
            'description': 'Weighted combination of 3 detection methods (balanced performance)'
        },
    }
    
    def __init__(self):
        self.engine = AnomalyDetectionEngine()
        self.nab = NABIntegration()
        self.benchmark = Benchmark()
        self.results = []
        self.summary_table = None
        self.encoding_issues = []  # Track encoding issues
        
    def run_full_evaluation(self, output_file='evaluation_report.json'):
        """Run evaluation on all key datasets, save results."""
        print("\n" + "="*80)
        print("ANOMALY DETECTION ALGORITHM EVALUATION REPORT")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Datasets: {len(self.KEY_DATASETS)} key NAB datasets")
        print(f"Algorithms: {len(self.ALGORITHM_PRESETS)} methods")
        print("="*80 + "\n")
        
        # Print algorithm presets
        print("ALGORITHM PRESETS (Simple Mode - Recommended):")
        print("-" * 80)
        for algo_name, preset in self.ALGORITHM_PRESETS.items():
            print(f"\n  {preset['name']}")
            print(f"  Description: {preset['description']}")
            params = {k: v for k, v in preset.items() 
                     if k not in ['name', 'description']}
            for key, val in params.items():
                print(f"    - {key}: {val}")
        
        print("\n" + "="*80)
        print("EVALUATING ON KEY DATASETS")
        print("="*80 + "\n")
        
        # Evaluate each dataset
        for dataset_info in self.KEY_DATASETS:
            self._evaluate_dataset(dataset_info)
        
        # Generate summary table
        self._generate_summary_table()
        
        # Save results
        self._save_results(output_file)
        
        return self.summary_table
    
    def _evaluate_dataset(self, dataset_info):
        """Evaluate all algorithms on a single dataset."""
        path = dataset_info['path']
        name = dataset_info['name']
        
        print(f"\nDataset: {name}")
        print(f"  Path: {path}")
        print(f"  Type: {dataset_info['type']}")
        
        try:
            # Load data using DataLoader (same as diagnostic.py)
            loader = DataLoader()
            data = loader.preprocess(f"NAB/data/{path}.csv", normalize=False)
            
            if data is None or data.empty:
                print(f"  [FAILED] Could not load dataset")
                return
            
            # Get values (no normalization for benchmark consistency)
            values = data['value'].values
            
            # Load NAB ground truth intervals using DataLoader DataFrame
            nab_intervals = self.nab.get_anomaly_intervals(path, data)
            if nab_intervals is None or len(nab_intervals) == 0:
                print(f"  [WARNING] No NAB intervals found")
                return
            
            print(f"  Data points: {len(values)}")
            print(f"  Anomaly intervals: {len(nab_intervals)}")
            
            # Evaluate each algorithm
            dataset_results = {'dataset': name, 'path': path, 'samples': len(values)}
            
            for algo_name, preset in self.ALGORITHM_PRESETS.items():
                print(f"    Testing: {preset['name']}...", end=' ', flush=True)
                
                # Run detection with preset parameters
                try:
                    if algo_name == 'rolling_stats':
                        anomalies = self.engine.detect_single(
                            algo_name, values,
                            window_size=preset['window_size'],
                            threshold=preset['threshold']
                        )
                    elif algo_name == 'prediction_error':
                        anomalies = self.engine.detect_single(
                            algo_name, values,
                            forecast_window=preset['forecast_window'],
                            threshold=preset['threshold']
                        )
                    elif algo_name == 'hybrid':
                        anomalies = self.engine.detect_single(
                            algo_name, values,
                            zscore_threshold=preset['zscore_threshold'],
                            trend_threshold=preset['trend_threshold'],
                            volatility_threshold=preset['volatility_threshold'],
                            weights=preset['weights']
                        )
                    
                    # Get detected indices
                    detected_indices = np.where(anomalies)[0]
                    
                    # Evaluate against NAB using fuzzy matching (±100) via Benchmark class
                    metrics_obj = self.benchmark.evaluate_with_nab_intervals(
                        detected_indices, nab_intervals, len(values),
                        execution_time=0.0, tolerance=100
                    )
                    
                    # Extract metrics from BenchmarkMetrics object (use correct attribute names)
                    metrics = {
                        'tp': metrics_obj.detected_anomalies,
                        'fn': metrics_obj.missed_anomalies,
                        'fp': metrics_obj.false_alarms,
                        'precision': metrics_obj.precision,
                        'recall': metrics_obj.recall,
                        'f1': metrics_obj.f1_score,
                        'nab_score': metrics_obj.nab_score
                    }
                    
                    # Store results
                    algo_result = {
                        'algorithm': preset['name'],
                        'detections': len(detected_indices),
                        'tp': metrics['tp'],
                        'fn': metrics['fn'],
                        'fp': metrics['fp'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'f1': metrics['f1'],
                        'nab_score': metrics['nab_score'],
                    }
                    
                    dataset_results[algo_name] = algo_result
                    
                    # Print inline result
                    f1_str = f"{algo_result['f1']:.4f}"
                    precision_str = f"{algo_result['precision']:.4f}"
                    recall_str = f"{algo_result['recall']:.4f}"
                    print(f"[OK] F1={f1_str} (P={precision_str}, R={recall_str})")
                    
                except Exception as e:
                    print(f"[ERR] {str(e)}")
                    continue
            
            self.results.append(dataset_results)
            
        except Exception as e:
            print(f"  [ERROR] Processing failed: {str(e)}\n")
    
    def _generate_summary_table(self):
        """Generate comparison table of all results."""
        if not self.results:
            print("\nNo results to summarize.")
            return
        
        print("\n" + "="*80)
        print("SUMMARY: F1 SCORES BY ALGORITHM AND DATASET")
        print("="*80 + "\n")
        
        # Create summary table
        summary_data = []
        for dataset_result in self.results:
            row = {'Dataset': dataset_result['dataset'], 'Samples': dataset_result['samples']}
            for algo_name in self.ALGORITHM_PRESETS.keys():
                if algo_name in dataset_result:
                    algo_result = dataset_result[algo_name]
                    row[f"{algo_name}_F1"] = algo_result['f1']
                    row[f"{algo_name}_TP"] = algo_result['tp']
                    row[f"{algo_name}_FP"] = algo_result['fp']
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        
        # Print formatted table
        print(df_summary.to_string(index=False))
        print("\n" + "="*80)
        
        # Print detailed metrics per algorithm
        print("DETAILED METRICS BY ALGORITHM")
        print("="*80 + "\n")
        
        for algo_name, preset in self.ALGORITHM_PRESETS.items():
            print(f"{preset['name']}")
            print("-" * 80)
            
            algo_results = []
            for dataset_result in self.results:
                if algo_name in dataset_result:
                    r = dataset_result[algo_name]
                    algo_results.append({
                        'Dataset': dataset_result['dataset'],
                        'Detections': r['detections'],
                        'TP': r['tp'],
                        'FN': r['fn'],
                        'FP': r['fp'],
                        'Precision': f"{r['precision']:.4f}",
                        'Recall': f"{r['recall']:.4f}",
                        'F1': f"{r['f1']:.4f}",
                        'NAB_Score': f"{r['nab_score']:.2f}"
                    })
            
            df_algo = pd.DataFrame(algo_results)
            print(df_algo.to_string(index=False))
            print()
        
        # Calculate averages
        print("\n[AVERAGE METRICS ACROSS KEY DATASETS]")
        
        avg_metrics = {}
        for algo_name, preset in self.ALGORITHM_PRESETS.items():
            f1_scores = []
            precisions = []
            recalls = []
            
            for dataset_result in self.results:
                if algo_name in dataset_result:
                    r = dataset_result[algo_name]
                    f1_scores.append(r['f1'])
                    precisions.append(r['precision'])
                    recalls.append(r['recall'])
            
            if f1_scores:
                avg_metrics[algo_name] = {
                    'avg_f1': np.mean(f1_scores),
                    'avg_precision': np.mean(precisions),
                    'avg_recall': np.mean(recalls),
                    'std_f1': np.std(f1_scores) if len(f1_scores) > 1 else 0
                }
        
        avg_data = []
        for algo_name, preset in self.ALGORITHM_PRESETS.items():
            if algo_name in avg_metrics:
                m = avg_metrics[algo_name]
                avg_data.append({
                    'Algorithm': preset['name'],
                    'Avg F1': f"{m['avg_f1']:.4f}",
                    'Avg Precision': f"{m['avg_precision']:.4f}",
                    'Avg Recall': f"{m['avg_recall']:.4f}",
                    'Std Dev': f"{m['std_f1']:.4f}"
                })
        
        df_avg = pd.DataFrame(avg_data)
        print(df_avg.to_string(index=False))
        print()
        
        # Recommendation
        print(f"\n[RECOMMENDATION FOR SIMPLE MODE]\n")
        if avg_metrics:
            best_algo = max(avg_metrics.items(), key=lambda x: x[1]['avg_f1'])
            best_preset = self.ALGORITHM_PRESETS[best_algo[0]]
            print(f"RECOMMENDED (for Simple Mode default):")
            print(f"  Algorithm: {best_preset['name']}")
            print(f"  Average F1 Score: {best_algo[1]['avg_f1']:.4f}")
            print(f"  Reason: Balanced precision and recall across all datasets\n")
        else:
            print(f"NOTE: No results available for recommendation\n")
        
        self.summary_table = df_summary
    
    def _save_results(self, output_file):
        """Save complete results to JSON file."""
        output_path = Path(output_file)
        
        output_dict = {
            'generated': datetime.now().isoformat(),
            'datasets': [d['name'] for d in self.KEY_DATASETS],
            'algorithms': list(self.ALGORITHM_PRESETS.keys()),
            'presets': self.ALGORITHM_PRESETS,
            'results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_dict, f, indent=2, default=str)
        
        print(f"Results saved to: {output_path}\n")


def main():
    """Run complete evaluation report."""
    report = EvaluationReport()
    report.run_full_evaluation()


if __name__ == '__main__':
    main()
