"""
Calcul NAB Score oficial pentru algoritmii implementati.

Pipeline:
  1. Rulează fiecare algoritm pe toate seriile NAB oficiale
  2. Scrie fisierele in NAB/results/<algo>/ (format: timestamp, value, anomaly_score, label)
  3. Optimizeaza pragul per algoritm + profil (sweep)
  4. Calculeaza scorul brut cu Sweeper
  5. Normalizeaza: (raw - null_raw) / (perfect_raw - null_raw) * 100
  6. Salveaza nab_scores.csv

Profilele NAB:
  standard         : fpWeight=0.11, fnWeight=1.0, tpWeight=1.0
  reward_low_FP    : fpWeight=0.22
  reward_low_FN    : fnWeight=2.0
"""

import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "NAB"))   # face nab/ importabil
sys.path.insert(0, str(ROOT))           # face anomaly_algorithms importabil

import json
import math
import time
import warnings
import logging
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from nab.sweeper import Sweeper, scaledSigmoid

from anomaly_algorithms import AnomalyDetectionEngine

# ── Configurare ───────────────────────────────────────────────────────────────

NAB_DATA   = ROOT / "NAB/data"
NAB_LABELS = ROOT / "NAB/labels/combined_windows.json"
NAB_RESULTS= ROOT / "NAB/results"

# Parametri fixe per algoritm (identici cu evaluarea anterioara)
ALGO_FIXED = {
    "rollingZScore"   : dict(window_size=20),
    "predictionError" : dict(forecast_window=5),
    "hybridScore"     : dict(weights=(0.5, 0.3, 0.2)),
    "isolationForest" : dict(n_estimators=100, window_size=20),
    "meanShift"       : dict(window=50),
}

# Mapare nume intern → cheie engine.detect_single
ALGO_KEY = {
    "rollingZScore"   : "rolling_stats",
    "predictionError" : "prediction_error",
    "hybridScore"     : "hybrid",
    "isolationForest" : "isolation_forest",
    "meanShift"       : "mean_shift",
}

# Parametri threshold pentru detectia binara (Low = k=4.0 / contamination=0.002)
# Scorurile salvate sunt BINARE (0.0 / 1.0) — corespund exact evaluarii anterioare
ALGO_DETECT_PARAMS = {
    "rollingZScore"   : dict(threshold_method="adaptive_std", k_std=4.0),
    "predictionError" : dict(threshold_method="adaptive_std", k_std=4.0),
    "hybridScore"     : dict(threshold_method="adaptive_std", k_std=4.0),
    "isolationForest" : dict(contamination=0.002),
    "meanShift"       : dict(threshold_method="adaptive_std", k_std=4.0),
}

NAB_PROFILES = {
    "standard"      : {"tpWeight": 1.0, "fnWeight": 1.0, "fpWeight": 0.11},
    "reward_low_FP" : {"tpWeight": 1.0, "fnWeight": 1.0, "fpWeight": 0.22},
    "reward_low_FN" : {"tpWeight": 1.0, "fnWeight": 2.0, "fpWeight": 0.11},
}

PROBATION = 0.15


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_windows():
    with open(NAB_LABELS) as f:
        raw = json.load(f)
    windows = {}
    for rel_path, win_list in raw.items():
        parsed = []
        for w in win_list:
            t0 = pd.Timestamp(w[0])
            t1 = pd.Timestamp(w[1])
            parsed.append((t0, t1))
        windows[rel_path] = parsed
    return windows


def list_nab_series():
    """Returneaza lista de (category, series_stem) pentru toate seriile NAB oficiale."""
    series_list = []
    for cat_dir in sorted(NAB_DATA.iterdir()):
        if not cat_dir.is_dir() or cat_dir.name == "README.md":
            continue
        for csv_file in sorted(cat_dir.glob("*.csv")):
            series_list.append((cat_dir.name, csv_file.stem))
    return series_list


def write_result_file(algo_name, category, series_stem, df_data, scores):
    """Scrie fisierul de rezultate in formatul NAB."""
    out_dir = NAB_RESULTS / algo_name / category
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{algo_name}_{series_stem}.csv"

    df_out = pd.DataFrame({
        "timestamp"     : df_data["timestamp"],
        "value"         : df_data["value"],
        "anomaly_score" : scores,
        "label"         : 0,
    })
    df_out.to_csv(out_path, index=False)
    return out_path


# ── Etapa 1: Generare scoruri si scriere fisiere ──────────────────────────────

def generate_results():
    engine      = AnomalyDetectionEngine()
    series_list = list_nab_series()
    total       = len(series_list)

    print(f"Generare scoruri pentru {total} serii NAB × {len(ALGO_FIXED)} algoritmi\n")

    for idx, (category, series_stem) in enumerate(series_list, 1):
        csv_path = NAB_DATA / category / f"{series_stem}.csv"
        try:
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        except Exception as e:
            print(f"  [SKIP] {category}/{series_stem}: {e}")
            continue

        values = df["value"].values.astype(float)
        print(f"[{idx:3d}/{total}] {category}/{series_stem} ({len(values)} pct)")

        for algo_name, fixed_params in ALGO_FIXED.items():
            detect_params = {**fixed_params, **ALGO_DETECT_PARAMS[algo_name]}
            try:
                result = engine.detect_single(ALGO_KEY[algo_name], values, **detect_params)
                # Scoruri binare: 1.0 = detectat anomalie, 0.0 = normal
                # Consistent cu evaluarea precision/recall anterioara
                scores = result.labels.astype(float)
            except Exception as e:
                print(f"    {algo_name}: ERROR — {e}")
                scores = np.zeros(len(values))

            write_result_file(algo_name, category, series_stem, df, scores)

    print("\nFisiere scrise in NAB/results/<algo>/\n")


# ── Etapa 2: Scoring NAB cu Sweeper ──────────────────────────────────────────

def nab_sweep_algo(algo_name, windows_dict, profile_name, cost_matrix):
    """
    Pentru un algoritm si un profil:
      - Citeste fisierele de rezultate
      - Ruleaza Sweeper.calcSweepScore pe fiecare serie
      - Combina toate AnomalyPoint-urile
      - Optimizeaza pragul (max score)
      - Returneaza (best_threshold, best_score, tp, fp, fn)
    """
    sweeper = Sweeper(
        probationPercent=PROBATION,
        costMatrix=cost_matrix,
    )

    all_points   = []
    missing_keys = []

    result_dir = NAB_RESULTS / algo_name
    for cat_dir in sorted(result_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        for csv_file in sorted(cat_dir.glob(f"{algo_name}_*.csv")):
            series_stem  = csv_file.stem[len(algo_name)+1:]
            rel_key      = f"{category}/{series_stem}.csv"

            if rel_key not in windows_dict:
                missing_keys.append(rel_key)
                continue

            df = pd.read_csv(csv_file, parse_dates=["timestamp"])
            timestamps    = list(df["timestamp"])
            anomaly_scores= list(df["anomaly_score"].astype(float))
            win_limits    = windows_dict[rel_key]

            pts = sweeper.calcSweepScore(
                timestamps, anomaly_scores, win_limits, rel_key
            )
            all_points.extend(pts)

    if not all_points:
        return None

    scores_by_thresh = sweeper.calcScoreByThreshold(all_points)
    best = max(scores_by_thresh, key=lambda s: s.score)
    return best   # ThresholdScore(threshold, score, tp, tn, fp, fn, total)


def compute_null_raw(windows_dict, cost_matrix):
    """
    Scor brut al detectorului null (nu detecteaza nimic):
      = -fnWeight * numar_ferestre_totale
    (fiecare fereastra cu anomalii este un FN)
    """
    fn_weight = cost_matrix["fnWeight"]
    n_windows = sum(len(ws) for ws in windows_dict.values() if ws)
    return -fn_weight * n_windows


def compute_perfect_raw(windows_dict, cost_matrix):
    """
    Scor brut al detectorului perfect (detecteaza primul punct din fiecare fereastra):
      = tpWeight * scaledSigmoid(-1.0) / scaledSigmoid(-1.0) * numar_ferestre
      = tpWeight * numar_ferestre
    Adica fiecare fereastra contribuie cu tpWeight.
    """
    tp_weight = cost_matrix["tpWeight"]
    n_windows = sum(len(ws) for ws in windows_dict.values() if ws)
    return tp_weight * n_windows


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Etapa 1
    generate_results()

    # Incarca ferestrele NAB
    windows_dict = load_windows()
    n_windows_total = sum(len(v) for v in windows_dict.values())
    print(f"Ferestre NAB incarcate: {len(windows_dict)} serii, {n_windows_total} ferestre totale\n")

    rows = []

    for profile_name, cost_matrix in NAB_PROFILES.items():
        null_raw    = compute_null_raw(windows_dict, cost_matrix)
        perfect_raw = compute_perfect_raw(windows_dict, cost_matrix)

        print(f"{'='*60}")
        print(f"Profil: {profile_name}")
        print(f"  null_raw={null_raw:.2f}  perfect_raw={perfect_raw:.2f}")
        print(f"{'='*60}")
        print(f"{'Algoritm':<20} {'Prag':>8} {'Scor brut':>12} {'NAB Score':>11} {'TP':>5} {'FP':>5} {'FN':>5}")
        print("-"*67)

        for algo_name in ALGO_FIXED:
            t0   = time.perf_counter()
            best = nab_sweep_algo(algo_name, windows_dict, profile_name, cost_matrix)
            elapsed = (time.perf_counter() - t0) * 1000

            if best is None:
                print(f"  {algo_name}: NO RESULTS")
                continue

            raw_score  = best.score
            nab_score  = (raw_score - null_raw) / (perfect_raw - null_raw) * 100

            print(f"  {algo_name:<18} {best.threshold:>8.4f} {raw_score:>12.2f} {nab_score:>10.2f}%"
                  f"  tp={best.tp} fp={best.fp} fn={best.fn}  ({elapsed:.0f}ms)")

            rows.append({
                "profile"   : profile_name,
                "algorithm" : algo_name,
                "threshold" : round(best.threshold, 6),
                "raw_score" : round(raw_score,  4),
                "nab_score" : round(nab_score,  4),
                "tp"        : best.tp,
                "fp"        : best.fp,
                "fn"        : best.fn,
                "null_raw"  : round(null_raw,    4),
                "perfect_raw": round(perfect_raw, 4),
            })

        print()

    df_out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "nab_scores.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Salvat: {out_path}")

    # Tabel final compact (Standard profile)
    print("\n=== NAB Score — Standard Profile ===")
    std = df_out[df_out["profile"] == "standard"].sort_values("nab_score", ascending=False)
    print(std[["algorithm","threshold","nab_score","tp","fp","fn"]].to_string(index=False))


if __name__ == "__main__":
    main()
