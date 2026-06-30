"""
Evaluare completa: 5 algoritmi x 2 threshold-uri x toate seriile
din realAWSCloudwatch, realKnownCause, artificialWithAnomaly.

Output:
  results_detailed.csv  — o linie per (serie, algoritm, threshold)
  results_summary.csv   — media per (categorie, algoritm, threshold)
"""

import traceback
import time
import numpy as np
import pandas as pd
from pathlib import Path

from nab_integration import NABIntegration
from anomaly_algorithms import AnomalyDetectionEngine
from benchmark import Benchmark

# ── Configurare ───────────────────────────────────────────────────────────────

CATEGORIES = ['realAWSCloudwatch', 'realKnownCause', 'artificialWithAnomaly']

NAB_DIR = Path('NAB/data')

NAB_TOLERANCE = 100

# Parametri per threshold (adaptive_std)
THRESHOLD_PARAMS = {
    'Low':    {'threshold_method': 'adaptive_std', 'k_std': 4.0, 'contamination': 0.002},
    'Medium': {'threshold_method': 'adaptive_std', 'k_std': 3.0, 'contamination': 0.005},
}

# Parametri fixe per algoritm (independente de threshold)
ALGO_FIXED = {
    'rolling_stats':    {'window_size': 20},
    'prediction_error': {'forecast_window': 5},
    'hybrid':           {'weights': (0.5, 0.3, 0.2)},
    'isolation_forest': {'n_estimators': 100, 'window_size': 20},
    'mean_shift':       {'window': 50},
}

ALGORITHMS = list(ALGO_FIXED.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def list_series(category: str):
    cat_dir = NAB_DIR / category
    if not cat_dir.exists():
        return []
    return sorted([f.stem for f in cat_dir.glob('*.csv')])


def load_series(nab: NABIntegration, category: str, series: str):
    name = f'{category}/{series}'
    try:
        df = nab.load_data(name)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        intervals = nab.get_anomaly_intervals(name, df)
        return df, intervals
    except Exception as exc:
        return None, []


def build_params(algo: str, threshold: str) -> dict:
    tp = THRESHOLD_PARAMS[threshold]
    fixed = ALGO_FIXED[algo].copy()

    if algo == 'isolation_forest':
        fixed['contamination'] = tp['contamination']
    else:
        fixed['threshold_method'] = tp['threshold_method']
        fixed['k_std'] = tp['k_std']

    return fixed


def compute_metrics(engine, algo, values, intervals, n):
    bench = Benchmark()
    result = engine.detect_single(algo, values)
    anom_idx = np.where(result.labels == 1)[0]
    m = bench.evaluate_with_nab_intervals(
        anom_idx, intervals, n, 0.0, tolerance=NAB_TOLERANCE
    )
    return {
        'precision': round(m.precision, 4),
        'recall':    round(m.recall,    4),
        'f1':        round(m.f1_score,  4),
        'tp': m.detected_anomalies,
        'fp': m.false_alarms,
        'fn': m.missed_anomalies,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    nab    = NABIntegration()
    engine = AnomalyDetectionEngine()
    rows   = []

    total_series = sum(len(list_series(c)) for c in CATEGORIES)
    done = 0

    for category in CATEGORIES:
        series_list = list_series(category)
        print(f'\n{"="*65}')
        print(f'Categorie: {category}  ({len(series_list)} serii)')
        print('='*65)

        for series in series_list:
            done += 1
            print(f'\n[{done}/{total_series}] {series}')

            df, intervals = load_series(nab, category, series)
            if df is None or not intervals:
                print('  SKIP — fara labels NAB')
                continue

            values = df['value'].values.astype(float)
            n = len(values)
            print(f'  {n} puncte, {len(intervals)} interval(e)')

            for threshold in ['Low', 'Medium']:
                for algo in ALGORITHMS:
                    params = build_params(algo, threshold)
                    try:
                        t0 = time.perf_counter()
                        result = engine.detect_single(algo, values, **params)
                        elapsed_ms = (time.perf_counter() - t0) * 1000

                        bench = Benchmark()
                        anom_idx = np.where(result.labels == 1)[0]
                        m = bench.evaluate_with_nab_intervals(
                            anom_idx, intervals, n, 0.0,
                            tolerance=NAB_TOLERANCE
                        )

                        rows.append({
                            'category':  category,
                            'series':    series,
                            'algorithm': algo,
                            'threshold': threshold,
                            'precision': round(m.precision, 4),
                            'recall':    round(m.recall,    4),
                            'f1':        round(m.f1_score,  4),
                            'tp': m.detected_anomalies,
                            'fp': m.false_alarms,
                            'fn': m.missed_anomalies,
                            'time_ms': round(elapsed_ms, 1),
                        })
                        print(f'  [{threshold:6s}] {algo:20s} '
                              f'P={m.precision:.3f} R={m.recall:.3f} F1={m.f1_score:.3f}')
                    except Exception:
                        print(f'  [{threshold:6s}] {algo:20s} ERROR')
                        traceback.print_exc()

    # ── Salvare detailed ──────────────────────────────────────────────────────
    df_det = pd.DataFrame(rows, columns=[
        'category','series','algorithm','threshold',
        'precision','recall','f1','tp','fp','fn','time_ms'
    ])
    df_det.to_csv('results_detailed.csv', index=False)
    print(f'\nSalvat: results_detailed.csv  ({len(df_det)} randuri)')

    # ── Salvare summary ───────────────────────────────────────────────────────
    df_sum = (df_det
              .groupby(['category', 'algorithm', 'threshold'])
              [['precision', 'recall', 'f1']]
              .mean()
              .round(4)
              .reset_index()
              .rename(columns={
                  'precision': 'avg_precision',
                  'recall':    'avg_recall',
                  'f1':        'avg_f1',
              }))
    df_sum.to_csv('results_summary.csv', index=False)
    print(f'Salvat: results_summary.csv  ({len(df_sum)} randuri)')

    # ── Preview summary ───────────────────────────────────────────────────────
    print('\n=== SUMMARY (avg F1 per algoritm per threshold) ===')
    pivot = df_sum.pivot_table(
        index=['algorithm'], columns=['threshold'], values='avg_f1'
    ).round(3)
    print(pivot.to_string())


if __name__ == '__main__':
    main()
