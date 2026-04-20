"""
Benchmark and Performance Evaluation Module

Evaluates anomaly detection algorithms against baselines
and provides comprehensive performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

from utils import setup_logger, calculate_metrics, confusion_matrix
from anomaly_algorithms import AnomalyResult


logger = setup_logger(__name__)


@dataclass
class BenchmarkMetrics:
    """Container for benchmark results."""
    algorithm: str
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    detection_delay_samples: float
    detection_delay_seconds: float
    nab_score: float
    execution_time: float
    total_anomalies: int
    detected_anomalies: int
    missed_anomalies: int
    false_alarms: int


class NABScorer:
    """
    Calculate NAB-like scores for anomaly detection algorithms.
    
    Reference: Numenta Anomaly Benchmark paper
    https://arxiv.org/abs/1510.03336
    """
    
    def __init__(
        self,
        probation_percent: float = 0.15,
        fp_weight: float = 0.11,
        fn_weight: float = 1.0
    ):
        """
        Initialize NAB scorer.
        
        Args:
            probation_percent: Fraction of data at start to ignore
            fp_weight: Weight for false positives
            fn_weight: Weight for false negatives
        """
        self.probation_percent = probation_percent
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight
    
    def calculate_probation_period(self, n_samples: int) -> int:
        """
        Calculate probation period (samples to ignore at start).
        
        Args:
            n_samples: Total number of samples
            
        Returns:
            Probation period in samples
        """
        return max(
            int(np.floor(self.probation_percent * n_samples)),
            int(self.probation_percent * 5000)
        )
    
    def score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict]:
        """
        Calculate NAB-like score.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            timestamps: Optional timestamp array for penalty calculation
            
        Returns:
            Tuple (score, metrics_dict)
        """
        probation = self.calculate_probation_period(len(y_true))
        
        # Ignore probation period
        y_true_eval = y_true[probation:]
        y_pred_eval = y_pred[probation:]
        
        # Count errors
        fp = np.sum((y_true_eval == 0) & (y_pred_eval == 1))
        fn = np.sum((y_true_eval == 1) & (y_pred_eval == 0))
        tp = np.sum((y_true_eval == 1) & (y_pred_eval == 1))
        
        # Calculate raw score (100 for perfect)
        max_score = 100.0
        
        # Penalize false positives and false negatives
        fp_cost = self.fp_weight * fp
        fn_cost = self.fn_weight * fn
        total_cost = fp_cost + fn_cost
        
        # Score = max(0, max_score - penalty)
        score = max(0, max_score - total_cost)
        
        metrics = {
            'probation_period': probation,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'raw_score': float(score),
            'normalized_score': float(score / max_score)
        }
        
        return float(score), metrics


class Benchmark:
    """
    Comprehensive benchmarking suite for anomaly detection algorithms.
    """
    
    def __init__(self, sample_rate: float = 300.0):
        """
        Initialize benchmark.
        
        Args:
            sample_rate: Samples per second (for calculating delays)
        """
        self.sample_rate = sample_rate
        self.scorer = NABScorer()
        self.results = {}
        self.logger = logger
    
    def evaluate_with_nab_intervals(
        self,
        anomaly_indices: np.ndarray,
        nab_intervals: List[Tuple[int, int]],
        total_samples: int,
        execution_time: float
    ) -> BenchmarkMetrics:
        """
        Evaluate detections using NAB interval comparison.
        
        Logica:
        - Interval cu >= 1 detecție în interior → TP
        - Interval fără detecții → FN
        - Detecție în afara tuturor intervalelor → FP
        
        Args:
            anomaly_indices: Indices of detected anomalies (sorted)
            nab_intervals: List of (start, end) tuples for ground truth anomalies
            total_samples: Total number of samples
            execution_time: Algorithm execution time
            
        Returns:
            BenchmarkMetrics with interval-based comparison
        """
        if not isinstance(anomaly_indices, np.ndarray):
            anomaly_indices = np.array(anomaly_indices)
        
        # Compute TP and FN per interval
        tp = 0  # Intervale cu cel puțin o detecție
        fn = 0  # Intervale fără detecții
        
        for start, end in nab_intervals:
            # Verifică dacă există detecții în interval [start, end)
            detections_in_interval = np.sum((anomaly_indices >= start) & (anomaly_indices < end))
            
            if detections_in_interval > 0:
                tp += 1
            else:
                fn += 1
        
        # Compute FP: detecții în afara tuturor intervalelor
        fp = 0
        for idx in anomaly_indices:
            in_any_interval = False
            for start, end in nab_intervals:
                if start <= idx < end:
                    in_any_interval = True
                    break
            if not in_any_interval:
                fp += 1
        
        # Compute TN (pentru completitudine)
        total_anomaly_coverage = sum(end - start for start, end in nab_intervals)
        tn = total_samples - total_anomaly_coverage - fp
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Calculate NAB-like score
        nab_score = max(0, 100 - (0.11 * fp + 1.0 * fn))
        
        bench_metrics = BenchmarkMetrics(
            algorithm="NAB Interval Based",
            precision=precision,
            recall=recall,
            f1_score=f1,
            false_positive_rate=fpr,
            detection_delay_samples=0.0,
            detection_delay_seconds=0.0,
            nab_score=nab_score,
            execution_time=execution_time,
            total_anomalies=len(nab_intervals),
            detected_anomalies=tp,  # TP = intervale detectate
            missed_anomalies=fn,    # FN = intervale ratate
            false_alarms=fp         # FP = detecții false
        )
        
        self.results["NAB_Interval_Based"] = bench_metrics
        
        self.logger.info(
            f"NAB Interval Evaluation: "
            f"TP={tp}, FN={fn}, FP={fp} | "
            f"Precision={precision:.3f}, Recall={recall:.3f}, "
            f"F1={f1:.3f}, NAB={nab_score:.1f}"
        )
        
        return bench_metrics
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        anomaly_result: AnomalyResult,
        execution_time: float,
        timestamps: Optional[np.ndarray] = None,
        algorithm_name: Optional[str] = None
    ) -> BenchmarkMetrics:
        """
        Evaluate single algorithm against ground truth.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            anomaly_result: AnomalyResult object with scores
            execution_time: Algorithm execution time in seconds
            timestamps: Optional timestamps for delay calculation
            algorithm_name: Optional algorithm name to store
            
        Returns:
            BenchmarkMetrics
        """
        # Calculate basic metrics
        metrics = calculate_metrics(y_true, y_pred)
        
        # Count anomalies
        n_true_anomalies = int(np.sum(y_true))
        n_detected = int(np.sum(y_pred & y_true))
        n_false_alarms = int(np.sum(y_pred & ~y_true.astype(bool)))
        
        # Calculate detection delay
        delay_samples = self._calculate_detection_delay(y_true, y_pred)
        delay_seconds = delay_samples / self.sample_rate
        
        # Calculate NAB score
        nab_score, nab_metrics = self.scorer.score(y_true, y_pred, timestamps)
        
        # Create metrics object
        bench_metrics = BenchmarkMetrics(
            algorithm=algorithm_name or anomaly_result.algorithm,
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            false_positive_rate=metrics['false_positive_rate'],
            detection_delay_samples=delay_samples,
            detection_delay_seconds=delay_seconds,
            nab_score=nab_score,
            execution_time=execution_time,
            total_anomalies=n_true_anomalies,
            detected_anomalies=n_detected,
            missed_anomalies=n_true_anomalies - n_detected,
            false_alarms=n_false_alarms
        )
        
        self.results[bench_metrics.algorithm] = bench_metrics
        
        self.logger.info(
            f"Evaluated {bench_metrics.algorithm}: "
            f"F1={bench_metrics.f1_score:.3f}, "
            f"NAB={bench_metrics.nab_score:.1f}"
        )
        
        return bench_metrics
    
    def evaluate_multiple(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, Tuple[np.ndarray, AnomalyResult, float]],
        timestamps: Optional[np.ndarray] = None
    ) -> Dict[str, BenchmarkMetrics]:
        """
        Evaluate multiple algorithms.
        
        Args:
            y_true: Ground truth labels
            predictions: Dict mapping algorithm names to
                        (predictions, AnomalyResult, execution_time)
            timestamps: Optional timestamps
            
        Returns:
            Dictionary of BenchmarkMetrics
        """
        results = {}
        
        for alg_name, (y_pred, anom_result, exec_time) in predictions.items():
            try:
                metrics = self.evaluate(
                    y_true, y_pred, anom_result, exec_time,
                    timestamps, alg_name
                )
                results[alg_name] = metrics
            except Exception as e:
                self.logger.error(f"Error evaluating {alg_name}: {e}")
        
        return results
    
    def _calculate_detection_delay(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate average delay between true and detected anomalies.
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            
        Returns:
            Average delay in samples
        """
        delays = []
        
        # Find anomaly windows in ground truth
        true_events = np.where(y_true == 1)[0]
        if len(true_events) == 0:
            return 0.0
        
        # Group consecutive true anomalies
        event_starts = [true_events[0]]
        for i in range(1, len(true_events)):
            if true_events[i] - true_events[i-1] > 1:
                event_starts.append(true_events[i])
        
        # For each event, find when it was detected
        for start in event_starts:
            # First detection at or after event start
            detections = np.where(y_pred[start:] == 1)[0]
            if len(detections) > 0:
                delay = detections[0]
                delays.append(delay)
        
        return float(np.mean(delays)) if delays else 0.0
    
    def compare_with_baseline(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        baseline_scores: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Compare results with baseline algorithms.
        
        Baseline scores (from NAB paper):
        - HTM: ~70.5
        - Random Cut Forest: ~51.7
        - Skyline: ~35.7
        - Random: ~11.0
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            baseline_scores: Dict of baseline algorithm scores
            
        Returns:
            Comparison dictionary
        """
        if baseline_scores is None:
            baseline_scores = {
                'HTM': 70.5,
                'Random Cut Forest': 51.7,
                'Skyline': 35.7,
                'Random': 11.0
            }
        
        _, nab_metrics = self.scorer.score(y_true, y_pred)
        algo_score = nab_metrics['raw_score']
        
        comparison = {
            'current_score': float(algo_score),
            'baselines': baseline_scores,
            'vs_htm': float(algo_score - baseline_scores['HTM']),
            'vs_rcf': float(algo_score - baseline_scores['Random Cut Forest']),
            'vs_skyline': float(algo_score - baseline_scores['Skyline']),
            'rank': self._calculate_rank(algo_score, baseline_scores)
        }
        
        return comparison
    
    def _calculate_rank(
        self,
        score: float,
        baselines: Dict[str, float]
    ) -> int:
        """Calculate ranking among baselines."""
        rank = 1
        for baseline_score in baselines.values():
            if score < baseline_score:
                rank += 1
        return rank
    
    def scalability_analysis(
        self,
        execution_times: Dict[int, float]
    ) -> Dict:
        """
        Analyze scalability with number of cores.
        
        Args:
            execution_times: Dict mapping core counts to execution times
            
        Returns:
            Scalability analysis with speedup and efficiency
        """
        if 1 not in execution_times or execution_times[1] is None:
            return {}
        
        t1 = execution_times[1]
        analysis = {
            'reference_time_1core': float(t1),
            'speedup': {},
            'efficiency': {}
        }
        
        for cores, t_p in execution_times.items():
            if t_p is not None and t_p > 0:
                speedup = t1 / t_p
                efficiency = speedup / cores
                
                analysis['speedup'][cores] = float(speedup)
                analysis['efficiency'][cores] = float(efficiency)
        
        return analysis
    
    def get_summary(self) -> pd.DataFrame:
        """
        Get summary of all benchmark results.
        
        Returns:
            Pandas DataFrame with all results
        """
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for algorithm, metrics in self.results.items():
            data.append({
                'Algorithm': metrics.algorithm,
                'Precision': f"{metrics.precision:.3f}",
                'Recall': f"{metrics.recall:.3f}",
                'F1-Score': f"{metrics.f1_score:.3f}",
                'FPR': f"{metrics.false_positive_rate:.3f}",
                'NAB Score': f"{metrics.nab_score:.1f}",
                'Exec Time (s)': f"{metrics.execution_time:.3f}",
                'Detected': f"{metrics.detected_anomalies}/{metrics.total_anomalies}"
            })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    y_true = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0])
    
    bench = Benchmark()
    
    from anomaly_algorithms import AnomalyResult
    result = AnomalyResult(
        labels=y_pred,
        scores=np.random.rand(len(y_pred)),
        thresholds={},
        algorithm="Test",
        parameters={}
    )
    
    metrics = bench.evaluate(y_true, y_pred, result, 0.5)
    print(metrics)
