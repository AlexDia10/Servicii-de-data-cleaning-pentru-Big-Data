"""
Diagnostic script to evaluate algorithms on nyc_taxi dataset and compare with NAB.
"""

import pandas as pd
import numpy as np
from data_loader import DataLoader
from anomaly_algorithms import AnomalyDetectionEngine
from benchmark import Benchmark
from nab_integration import NABIntegration
from pathlib import Path
import sys


def print_section(title):
    """Print formatted section title."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_diagnostic():
    """Run comprehensive diagnostic on RDS CPU dataset."""
    
    print_section("DIAGNOSTIC: RDS CPU UTILIZATION DATASET + NAB COMPARISON")
    
    # Load data
    print("1. LOADING DATA (RAW - NO NORMALIZATION)...")
    loader = DataLoader()
    try:
        df = loader.preprocess("realAWSCloudwatch/rds_cpu_utilization_cc0c53", normalize=False)
        print(f"   Loaded: {len(df)} rows")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Value range: [{df['value'].min():.2f}, {df['value'].max():.2f}]")
        print(f"   Mean: {df['value'].mean():.2f}, Std: {df['value'].std():.2f}")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Load NAB labels
    print("\n2. LOADING NAB INTERVALS...")
    total_anomaly_points = 0
    try:
        nab = NABIntegration()
        nab_intervals = nab.get_anomaly_intervals("realAWSCloudwatch/rds_cpu_utilization_cc0c53", df)
        if nab_intervals:
            total_anomaly_points = sum(end - start for start, end in nab_intervals)
            print(f"   NAB intervals found: {len(nab_intervals)}")
            print(f"   Total anomaly points in NAB: {total_anomaly_points} ({100*total_anomaly_points/len(df):.2f}%)")
            print(f"   Intervals: {nab_intervals[:3]}... (showing first 3)")
        else:
            print("   WARNING: No NAB intervals found!")
            nab_intervals = None
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        nab_intervals = None
    
    # Run each algorithm
    values = df['value'].values
    engine = AnomalyDetectionEngine()
    benchmark = Benchmark()
    
    algorithms = [
        ('rolling_stats', {'window_size': 20}),
        ('prediction_error', {'forecast_window': 5}),
        ('hybrid', {'window_size': 20})
    ]
    
    results_summary = []
    
    for algo_name, params in algorithms:
        print_section(f"ALGORITHM: {algo_name.upper()}")
        print(f"   Parameters: {params}")
        
        try:
            # Run detection
            result = engine.detect_single(algo_name, values, **params)
            detected_indices = np.where(result.labels == 1)[0]
            n_detected = len(detected_indices)
            
            print(f"\n   Detections: {n_detected} points ({100*n_detected/len(values):.2f}%)")
            
            if n_detected > 0:
                print(f"   First 5 detections at indices: {detected_indices[:5].tolist()}")
                print(f"   Last 5 detections at indices: {detected_indices[-5:].tolist()}")
            
            # Evaluate against NAB if available
            if nab_intervals:
                metrics = benchmark.evaluate_with_nab_intervals(
                    detected_indices,
                    nab_intervals,
                    len(values),
                    0  # execution time not measured
                )
                
                print(f"\n   NAB METRICS:")
                print(f"   - True Positives (TP):  {metrics.detected_anomalies} intervals detected")
                print(f"   - False Negatives (FN): {metrics.missed_anomalies} intervals missed")
                print(f"   - False Positives (FP): {metrics.false_alarms} spurious detections")
                print(f"   - Precision: {metrics.precision:.4f}")
                print(f"   - Recall:    {metrics.recall:.4f}")
                print(f"   - F1 Score:  {metrics.f1_score:.4f}")
                print(f"   - NAB Score: {metrics.nab_score:.4f}")
                
                results_summary.append({
                    'Algorithm': algo_name,
                    'Detections': n_detected,
                    'TP': metrics.detected_anomalies,
                    'FN': metrics.missed_anomalies,
                    'FP': metrics.false_alarms,
                    'Precision': f"{metrics.precision:.4f}",
                    'Recall': f"{metrics.recall:.4f}",
                    'F1': f"{metrics.f1_score:.4f}",
                    'NAB': f"{metrics.nab_score:.4f}"
                })
            else:
                results_summary.append({
                    'Algorithm': algo_name,
                    'Detections': n_detected,
                    'TP': '---',
                    'FN': '---',
                    'FP': '---',
                    'Precision': '---',
                    'Recall': '---',
                    'F1': '---',
                    'NAB': '---'
                })
                
        except Exception as e:
            print(f"   ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table
    print_section("SUMMARY COMPARISON")
    
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.to_string(index=False))
    
    # Analysis
    print_section("ANALYSIS & DIAGNOSTICS")
    
    if nab_intervals is None:
        print("[ERROR] NAB intervals not loaded - cannot evaluate against ground truth")
        print("   Possible causes:")
        print("   - Label file missing or corrupted")
        print("   - Dataset name mismatch")
        print("   - JSON parsing issue")
    else:
        print(f"[OK] NAB intervals loaded: {len(nab_intervals)} anomaly regions")
        
        if not results_summary or all(r['Detections'] == 0 for r in results_summary):
            print("\n[ERROR] NO DETECTIONS MADE - Algorithms may be too conservative")
            print("   Solutions:")
            print("   - Lower thresholds in anomaly_algorithms.py")
            print("   - Reduce window_size parameter")
            print("   - Try different algorithms")
        else:
            best_f1 = max(results_summary, key=lambda x: float(x['F1']) if isinstance(x['F1'], str) and x['F1'] != '---' else 0)
            if best_f1['F1'] != '---':
                print(f"\n[RESULT] Best performing: {best_f1['Algorithm']} (F1={best_f1['F1']})")
                
                f1_val = float(best_f1['F1'])
                if f1_val < 0.3:
                    print("\n   WARNING: Low F1 score - ALGORITHM ISSUES DETECTED:")
                    print("   1. Algorithms detecting noise instead of real anomalies")
                    print("   2. NAB intervals are SPARSE (only 0.05% of data)")
                    print("   3. Thresholds are TOO SENSITIVE")
                    print("\n   ROOT CAUSE ANALYSIS:")
                    for res in results_summary:
                        if res['Detections'] != 0:
                            pct = (res['Detections'] / len(values)) * 100
                            nab_pct = (total_anomaly_points / len(values)) * 100
                            ratio = pct / max(nab_pct, 0.001)
                            print(f"   - {res['Algorithm']}: detects {pct:.2f}% vs NAB {nab_pct:.2f}% ({ratio:.1f}x too many)")
                    print("\n   RECOMMENDATIONS:")
                    print("   1. Debug thresholds (check anomaly_algorithms.py)")
                    print("   2. Increase window_size for smoothing")
                    print("   3. Use ensemble with voting mechanism")
                    print("   4. Apply preprocessing: outlier removal, smoothing")
                elif f1_val < 0.6:
                    print("   MODERATE: Performance could be improved")
                    print("   - Try different parameters")
                    print("   - Combine with other algorithms")
                else:
                    print("   GOOD: Reasonable detection performance")


if __name__ == "__main__":
    run_diagnostic()
