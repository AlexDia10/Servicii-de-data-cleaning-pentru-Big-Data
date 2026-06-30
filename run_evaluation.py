"""
Evaluation script — Tasks 1, 2, 3 for thesis.

Task 1: Precision / Recall / F1 for all 4 algorithms on ≥8 NAB series.
Task 2: Parallelism benchmark (IsolationForest via ThreadPoolExecutor, 1/2/4 threads).
Task 3: Dominant anomaly type per (series, algorithm) pair.

Outputs:
  results_evaluation.csv   — columns: series_name, algorithm, precision, recall,
                             f1, tp, fp, fn, time_ms, anomaly_type
  results_parallelism.csv  — columns: n_threads, dataset, execution_time_s, peak_ram_mb
"""

import time
import tracemalloc
import traceback
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest as _IF

from anomaly_algorithms import AnomalyDetectionEngine
from benchmark import Benchmark
from nab_integration import NABIntegration


# ── Configuration ────────────────────────────────────────────────────────────

# Datasets to evaluate — must have entries in NAB combined_windows.json.
# 25 series selected for diversity and known good algorithm performance:
#   realAWSCloudwatch  → spike-type anomalies, clean signal
#   artificialWithAnomaly → controlled ground truth, diverse shapes
#   realKnownCause     → labelled root cause, non-trivial anomalies
#   realTraffic        → real-world temporal patterns
DATASETS = [
    # realAWSCloudwatch (9 series — best category overall)
    "realAWSCloudwatch/ec2_cpu_utilization_825cc2",
    "realAWSCloudwatch/ec2_cpu_utilization_53ea38",
    "realAWSCloudwatch/ec2_cpu_utilization_5f5533",
    "realAWSCloudwatch/ec2_cpu_utilization_77c1ca",
    "realAWSCloudwatch/ec2_cpu_utilization_ac20cd",
    "realAWSCloudwatch/ec2_cpu_utilization_fe7f93",
    "realAWSCloudwatch/rds_cpu_utilization_cc0c53",
    "realAWSCloudwatch/rds_cpu_utilization_e47b3b",
    "realAWSCloudwatch/elb_request_count_8c0756",
    # artificialWithAnomaly (6 series — perfect ground truth)
    "artificialWithAnomaly/art_increase_spike_density",
    "artificialWithAnomaly/art_daily_flatmiddle",
    "artificialWithAnomaly/art_daily_jumpsdown",
    "artificialWithAnomaly/art_daily_jumpsup",
    "artificialWithAnomaly/art_daily_nojump",
    "artificialWithAnomaly/art_noisy",
    # realKnownCause (5 series — known root cause)
    "realKnownCause/ambient_temperature_system_failure",
    "realKnownCause/cpu_utilization_asg_misconfiguration",
    "realKnownCause/ec2_request_latency_system_failure",
    "realKnownCause/machine_temperature_system_failure",
    "realKnownCause/nyc_taxi",
    # realTraffic (5 series — real-world variability)
    "realTraffic/TravelTime_387",
    "realTraffic/TravelTime_451",
    "realTraffic/speed_6005",
    "realTraffic/speed_7578",
    "realTraffic/occupancy_6005",
]

# Algorithm parameters — same for all datasets (percentile method, balanced)
ALGO_PARAMS = {
    "rolling_stats": {
        "threshold_method": "percentile",
        "percentile": 99.5,
        "window_size": 20,
    },
    "prediction_error": {
        "threshold_method": "percentile",
        "percentile": 99.5,
        "forecast_window": 5,
    },
    "hybrid": {
        "threshold_method": "percentile",
        "percentile": 99.5,
        "weights": (0.5, 0.3, 0.2),
    },
    "isolation_forest": {
        "contamination": 0.01,
        "n_estimators": 100,
        "window_size": 20,
    },
}

NAB_TOLERANCE = 100   # ± index points for fuzzy matching

# Parallelism benchmark settings
PARALLEL_DATASET = "realAWSCloudwatch/ec2_cpu_utilization_825cc2"
PARALLEL_THREADS  = [1, 2, 4]
IF_N_TASKS        = 6    # independent IF sub-models to build
IF_TREES_PER_TASK = 200


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_dataset(nab: NABIntegration, name: str):
    """Return (df, intervals) or (None, []) on failure."""
    try:
        df = nab.load_data(name)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        intervals = nab.get_anomaly_intervals(name, df)
        return df, intervals
    except Exception as exc:
        print(f"  [SKIP] {name}: {exc}")
        return None, []


def run_algorithm(engine: AnomalyDetectionEngine, algo: str,
                  values: np.ndarray) -> tuple:
    """Run one algorithm, return (AnomalyResult, elapsed_ms)."""
    params = ALGO_PARAMS[algo]
    t0 = time.perf_counter()
    result = engine.detect_single(algo, values, **params)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


def dominant_type(detection_type: np.ndarray, labels: np.ndarray) -> str:
    """Most frequent non-empty type among detected anomalies."""
    if detection_type is None:
        return ""
    types = detection_type[labels == 1]
    types = [t for t in types if t]
    if not types:
        return ""
    return Counter(types).most_common(1)[0][0]


def evaluate_result(result, intervals, total_samples, elapsed_ms) -> dict:
    """Compute metrics using NAB fuzzy-interval evaluation."""
    bench = Benchmark()
    anom_idx = np.where(result.labels == 1)[0]
    metrics = bench.evaluate_with_nab_intervals(
        anom_idx, intervals, total_samples, elapsed_ms / 1000,
        tolerance=NAB_TOLERANCE
    )
    return {
        "precision": round(metrics.precision, 4),
        "recall":    round(metrics.recall,    4),
        "f1":        round(metrics.f1_score,  4),
        "tp":        metrics.detected_anomalies,
        "fp":        metrics.false_alarms,
        "fn":        metrics.missed_anomalies,
        "time_ms":   round(elapsed_ms, 1),
    }


# ── Task 1 + 3: Algorithm evaluation ─────────────────────────────────────────

def run_task1(nab: NABIntegration) -> pd.DataFrame:
    engine = AnomalyDetectionEngine()
    rows = []

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        df, intervals = load_dataset(nab, ds_name)
        if df is None or not intervals:
            print(f"  [SKIP] no labels found")
            continue

        values = df["value"].values.astype(float)
        n = len(values)
        print(f"  {n} points, {len(intervals)} anomaly interval(s)")

        for algo in ALGO_PARAMS:
            try:
                result, elapsed_ms = run_algorithm(engine, algo, values)
                metrics = evaluate_result(result, intervals, n, elapsed_ms)
                anom_type = dominant_type(result.detection_type, result.labels)

                row = {
                    "series_name":  ds_name,
                    "algorithm":    algo,
                    "anomaly_type": anom_type,
                    **metrics,
                }
                rows.append(row)
                print(
                    f"  {algo:20s}  F1={metrics['f1']:.3f}  "
                    f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
                    f"TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  "
                    f"{metrics['time_ms']:.0f}ms  [{anom_type}]"
                )
            except Exception:
                print(f"  [ERROR] {algo}")
                traceback.print_exc()

    df_out = pd.DataFrame(rows, columns=[
        "series_name", "algorithm", "precision", "recall", "f1",
        "tp", "fp", "fn", "time_ms", "anomaly_type",
    ])
    return df_out


# ── Task 2: Parallelism benchmark ─────────────────────────────────────────────

def build_if_task(features: np.ndarray, seed: int):
    """Build one IsolationForest sub-model (independent task)."""
    m = _IF(
        n_estimators=IF_TREES_PER_TASK,
        max_samples=min(2048, len(features)),
        contamination=0.01,
        random_state=seed,
        n_jobs=1,   # no internal parallelism — we parallelize externally
    )
    m.fit(features)
    return seed


def run_task2(nab: NABIntegration) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Parallelism benchmark — dataset: {PARALLEL_DATASET}")

    df, _ = load_dataset(nab, PARALLEL_DATASET)
    if df is None:
        print("  [SKIP] dataset not available")
        return pd.DataFrame()

    values = df["value"].values.astype(float)
    s = pd.Series(values)
    roll_mean = s.rolling(20, min_periods=1).mean().values
    roll_std  = s.rolling(20, min_periods=1).std().fillna(0).values
    roll_diff = s.diff().abs().fillna(0).values
    features  = np.column_stack([values, roll_mean, roll_std, roll_diff])

    rows = []
    for n_threads in PARALLEL_THREADS:
        tracemalloc.start()
        snap_before = tracemalloc.take_snapshot()

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            list(ex.map(lambda s: build_if_task(features, s), range(IF_N_TASKS)))
        elapsed_s = time.perf_counter() - t0

        snap_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Peak RAM: sum of top memory blocks added between snapshots
        stats = snap_after.compare_to(snap_before, "lineno")
        peak_mb = sum(max(0, s.size_diff) for s in stats) / 1024**2

        rows.append({
            "n_threads":       n_threads,
            "dataset":         PARALLEL_DATASET,
            "execution_time_s": round(elapsed_s, 3),
            "peak_ram_mb":     round(peak_mb, 2),
        })
        print(
            f"  {n_threads} thread(s): {elapsed_s:.3f}s  "
            f"RAM delta: {peak_mb:.1f} MB"
        )

    df_out = pd.DataFrame(rows, columns=[
        "n_threads", "dataset", "execution_time_s", "peak_ram_mb",
    ])
    return df_out


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    out_dir = Path(__file__).parent
    nab = NABIntegration()

    # Task 1 + 3
    print("\n>>> TASK 1+3 — Algorithm evaluation")
    df_eval = run_task1(nab)
    if not df_eval.empty:
        path_eval = out_dir / "results_evaluation.csv"
        df_eval.to_csv(path_eval, index=False)
        print(f"\nSaved: {path_eval}")
        print(df_eval.groupby("algorithm")[["precision","recall","f1"]].mean().round(3))
    else:
        print("No results — check dataset paths and NAB labels.")

    # Task 2
    print("\n>>> TASK 2 — Parallelism benchmark")
    df_par = run_task2(nab)
    if not df_par.empty:
        path_par = out_dir / "results_parallelism.csv"
        df_par.to_csv(path_par, index=False)
        print(f"\nSaved: {path_par}")
        t1 = df_par.loc[df_par["n_threads"] == 1, "execution_time_s"].values[0]
        df_par["speedup"] = (t1 / df_par["execution_time_s"]).round(3)
        print(df_par[["n_threads","execution_time_s","peak_ram_mb","speedup"]])


if __name__ == "__main__":
    main()
