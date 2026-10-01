"""
Benchmark Isolation Forest: 1 / 2 / 4 / 8 core-uri
Ruleaza pe doua dimensiuni de serie:
  - Originala: machine_temperature_system_failure (~22k pct)
  - Concatenata via np.tile: 1.000.000 pct

Backend testat: loky (procese, default Windows) vs threading (thread-uri numpy)

Masoara: timp mediu (3 rulari) + peak RAM via tracemalloc
Salveaza: benchmark_if_cores.csv
"""

import os
import tracemalloc
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest

# ── Configurare ───────────────────────────────────────────────────────────────

ROOT          = Path(__file__).resolve().parent.parent
CSV_PATH      = str(ROOT / "NAB/data/realKnownCause/machine_temperature_system_failure.csv")
TARGET_SIZE   = 1_000_000
N_REPEATS     = 3
N_JOBS_LIST   = [1, 2, 4, -1]
N_ESTIMATORS  = 100
CONTAMINATION = 0.002
WINDOW_SIZE   = 20
RANDOM_STATE  = 42
N_CPUS        = os.cpu_count()

# ── Feature engineering (identic cu IsolationForestDetector.detect) ───────────

def build_features(data: np.ndarray, ws: int) -> np.ndarray:
    series = pd.Series(data)

    rolling_mean = (
        series.rolling(window=ws, min_periods=ws // 2, center=True)
        .mean().fillna(series.mean()).values
    )
    rolling_std = (
        series.rolling(window=ws, min_periods=ws // 2, center=True)
        .std().fillna(series.std()).values
    )
    rolling_diff = series.diff().abs().fillna(0).values

    p99 = np.percentile(data, 99)
    spike_zone = data[data > p99]
    typical_spike = float(np.median(spike_zone)) if len(spike_zone) >= 2 else float(p99)
    value_ratio = data / (typical_spike + 1e-8)

    std_mean = rolling_std.mean()
    std_std  = rolling_std.std() + 1e-8
    flatline_score = np.maximum(0.0, (std_mean - rolling_std) / std_std)

    return np.column_stack([
        data, rolling_mean, rolling_std, rolling_diff, value_ratio, flatline_score
    ])


def run_once(features: np.ndarray, n_jobs: int, backend: str):
    """Ruleaza IsolationForest si returneaza (timp_ms, peak_ram_mb)."""
    tracemalloc.start()
    t0 = time.perf_counter()

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )
    with joblib.parallel_backend(backend):
        model.fit(features)
        model.decision_function(features)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return elapsed_ms, peak / 1024 / 1024


def bench_size(label: str, data: np.ndarray, rows_out: list):
    n = len(data)
    print(f"\n{'='*70}")
    print(f"  {label}  |  {n:,} puncte  |  n_estimators={N_ESTIMATORS}  |  {N_REPEATS} rulari")
    print(f"{'='*70}")
    print(f"  {'Backend':<12} {'n_jobs':>6}  {'Core-uri':>10}  {'Timp mediu (ms)':>16}  "
          f"{'RAM peak (MB)':>14}  {'Speedup':>8}")
    print(f"  {'-'*70}")

    print("  Construiesc features...", end=" ", flush=True)
    features = build_features(data, WINDOW_SIZE)
    feat_mb = features.nbytes / 1024 / 1024
    print(f"gata  ({feat_mb:.1f} MB matricea de features)\n")

    baseline_t = None

    # Testeaza ambele backend-uri × toate configuratiile de core-uri
    configs = [
        ("sequential", 1,   "loky"),       # referinta: 1 core, fara paralel
        ("loky",       2,   "loky"),
        ("loky",       4,   "loky"),
        ("loky",      -1,   "loky"),
        ("threading",  2,   "threading"),
        ("threading",  4,   "threading"),
        ("threading", -1,   "threading"),
    ]

    for backend_label, n_jobs, backend in configs:
        real_cores = N_CPUS if n_jobs == -1 else n_jobs
        core_label = f"all ({real_cores})" if n_jobs == -1 else str(real_cores)

        times, rams = [], []
        for _ in range(N_REPEATS):
            t, r = run_once(features, n_jobs, backend)
            times.append(t)
            rams.append(r)

        avg_t = sum(times) / N_REPEATS
        avg_r = sum(rams)  / N_REPEATS

        if baseline_t is None:
            baseline_t = avg_t

        speedup = baseline_t / avg_t
        marker = " <-- mai rapid!" if speedup > 1.15 else ""

        print(f"  {backend_label:<12} {n_jobs:>6}  {core_label:>10}  {avg_t:>16.1f}  "
              f"{avg_r:>14.2f}  {speedup:>7.2f}x{marker}")

        rows_out.append({
            "dimensiune"      : label,
            "n_puncte"        : n,
            "backend"         : backend_label,
            "n_jobs"          : n_jobs,
            "n_cores_real"    : real_cores,
            "timp_mediu_ms"   : round(avg_t, 2),
            "ram_peak_mb"     : round(avg_r, 2),
            "t_min_ms"        : round(min(times), 2),
            "t_max_ms"        : round(max(times), 2),
            "speedup_vs_seq"  : round(speedup, 3),
        })


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df_raw   = pd.read_csv(CSV_PATH)
    data_orig = df_raw["value"].values.astype(float)
    n_orig    = len(data_orig)

    # Serie concatenata la 1M pct
    tile_factor = int(np.ceil(TARGET_SIZE / n_orig))
    data_1m     = np.tile(data_orig, tile_factor)[:TARGET_SIZE]

    print(f"Serie sursa: machine_temperature_system_failure")
    print(f"  {n_orig:,} pct  tiled x{tile_factor}  = {len(data_1m):,} pct")
    print(f"  CPU disponibile: {N_CPUS}")

    rows = []

    bench_size(f"Originala (~{n_orig//1000}k pct)", data_orig, rows)
    bench_size(f"Concatenata (1M pct)",              data_1m,   rows)

    df_out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "benchmark_if_cores.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n\nSalvat: {out_path}")

    # Rezumat comparativ
    print("\n=== Speedup real (vs. secvential 1 core) ===")
    print(f"{'Dimensiune':<25} {'Backend':<12} {'n_jobs':>7} {'Core-uri':>10} "
          f"{'Timp (ms)':>11} {'Speedup':>9}")
    print("-" * 80)
    for _, r in df_out.iterrows():
        core_lbl = f"all ({int(r['n_cores_real'])})" if r['n_jobs'] == -1 else str(int(r['n_cores_real']))
        print(f"  {r['dimensiune']:<23} {r['backend']:<12} {int(r['n_jobs']):>7} {core_lbl:>10} "
              f"{r['timp_mediu_ms']:>11.1f} {r['speedup_vs_seq']:>8.2f}x")


if __name__ == "__main__":
    main()
