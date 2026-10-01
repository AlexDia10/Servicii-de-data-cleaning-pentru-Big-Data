"""
NAB Dataset Analysis — Comparative Anomaly Detection
Runs all 4 algorithms with per-dataset tuned parameters on representative
NAB datasets and generates comparative charts for each anomaly type.
"""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from anomaly_algorithms import (
    RollingStatsZScore, PredictionErrorAnomaly,
    HybridAnomalyScore, IsolationForestDetector,
)

NAB_DATA   = ROOT / "NAB" / "data"
NAB_LABELS = ROOT / "NAB" / "labels" / "combined_windows.json"
OUT_DIR    = ROOT / "results" / "nab_analysis_charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(NAB_LABELS) as f:
    WINDOWS = json.load(f)

ALGO_COLORS = {
    "Rolling Z-Score":  "#42A5F5",
    "Prediction Error": "#FFA726",
    "Hybrid Score":     "#CE93D8",
    "Isolation Forest": "#66BB6A",
}

# ─── Per-dataset config ────────────────────────────────────────────────────────
# contamination = expected fraction of anomalous points (= NAB window size / total)
# percentile    = threshold level for statistical methods (top N% are anomalies)
# window        = lookback window; use larger values for mean-shift datasets
# ──────────────────────────────────────────────────────────────────────────────

DATASETS = [
    # ── 1. Spike-uri punctuale izolate ──────────────────────────────────────
    {
        "category": "1. Spike-uri punctuale izolate",
        "key":   "artificialWithAnomaly/art_load_balancer_spikes.csv",
        "short": "art_load_balancer_spikes",
        "desc":  ("Artificial load balancer — burst de vârfuri extreme izolate\n"
                  "într-o fereastră de ~33 ore; seria revine ulterior la normal."),
        # anomaly window ≈ 33h / 14d ≈ 10%; threshold_method=percentile auto-adapts
        "algo_params": {
            "Rolling Z-Score":  {"threshold_method": "percentile", "percentile": 92, "window_size": 30},
            "Prediction Error": {"threshold_method": "percentile", "percentile": 92, "forecast_window": 20},
            "Hybrid Score":     {"threshold_method": "percentile", "percentile": 92, "window_size": 30},
            "Isolation Forest": {"contamination": 0.10, "n_estimators": 200, "window_size": 30},
        },
    },
    {
        "category": "1. Spike-uri punctuale izolate",
        "key":   "realKnownCause/ec2_request_latency_system_failure.csv",
        "short": "ec2_request_latency",
        "desc":  ("EC2 Request Latency (AWS) — 3 ferestre scurte de latență extremă\n"
                  "cauzate de defecte de sistem reale (system failure events)."),
        # 3 windows ≈ 4% of total; percentile=96 → top 4% flagged
        "algo_params": {
            "Rolling Z-Score":  {"threshold_method": "percentile", "percentile": 96, "window_size": 20},
            "Prediction Error": {"threshold_method": "percentile", "percentile": 96, "forecast_window": 15},
            "Hybrid Score":     {"threshold_method": "percentile", "percentile": 96, "window_size": 20},
            "Isolation Forest": {"contamination": 0.04, "n_estimators": 200, "window_size": 20},
        },
    },
    # ── 2. Flatline / varianță anormal de mică ───────────────────────────────
    # art_daily_flatmiddle: daily oscillation → abrupt transition → stuck at 40.0
    # The flat portion begins at the END of the NAB window and continues after it.
    # Key insight: rolling_std → 0 during the flat block. Z-Score score also → 0
    # (division by near-zero std); IF detects the structural change via features.
    # "extended_window" adds a visual annotation over the post-NAB flat period.
    {
        "category": "2. Flatline / varianță anormal de mică",
        "key":   "artificialWithAnomaly/art_daily_flatmiddle.csv",
        "short": "art_daily_flatmiddle",
        "desc":  ("Artificial daily pattern — seria oscilantă se blochează la\n"
                  "valoarea 40.0 exact (rolling_std → 0). Perioada plată începe\n"
                  "la finalul ferestrei NAB și continuă după aceasta."),
        # window ≈ 10%; flat block is at end of window + after window
        # Small IF window (12 = 1h): rolling_std=0 inside flat is very distinct
        "algo_params": {
            "Rolling Z-Score":  {"threshold_method": "percentile", "percentile": 88, "window_size": 20},
            "Prediction Error": {"threshold_method": "percentile", "percentile": 88, "forecast_window": 15},
            "Hybrid Score":     {"threshold_method": "percentile", "percentile": 88, "window_size": 20},
            "Isolation Forest": {"contamination": 0.15, "n_estimators": 200, "window_size": 12},
        },
        # Show extra region after NAB window to make the flat period visible
        "extra_span": ("2014-04-11 16:45:00", "2014-04-12 08:00:00"),
    },
    # ── 3. Outlieri evidenți ────────────────────────────────────────────────
    {
        "category": "3. Outlieri evidenți",
        "key":   "realKnownCause/machine_temperature_system_failure.csv",
        "short": "machine_temperature_failure",
        "desc":  ("Temperatura mașinii industriale — 4 ferestre cu excursii clare\n"
                  "de temperatură cauzate de defecte fizice cunoscute."),
        # 4 × ~2d in ~80d ≈ 10%
        "algo_params": {
            "Rolling Z-Score":  {"threshold_method": "percentile", "percentile": 92, "window_size": 50},
            "Prediction Error": {"threshold_method": "percentile", "percentile": 92, "forecast_window": 30},
            "Hybrid Score":     {"threshold_method": "percentile", "percentile": 92, "window_size": 50},
            "Isolation Forest": {"contamination": 0.10, "n_estimators": 200, "window_size": 50},
        },
    },
    # ── 4. Mean shift / schimbare de regim ──────────────────────────────────
    {
        "category": "4. Mean shift / schimbare de regim",
        "key":   "artificialWithAnomaly/art_daily_jumpsup.csv",
        "short": "art_daily_jumpsup",
        "desc":  ("Artificial daily pattern — media seriei crește brusc cu ~50%\n"
                  "(jump up) și rămâne la noul nivel. Fereastră de anomalie ~33 ore.\n"
                  "Window mare → metodele statistice nu se adaptează rapid."),
        # window ≈ 10%; LARGE windows prevent fast adaptation after the jump
        "algo_params": {
            "Rolling Z-Score":  {"threshold_method": "percentile", "percentile": 92,
                                 "window_size": 200},
            "Prediction Error": {"threshold_method": "percentile", "percentile": 92,
                                 "forecast_window": 150},
            "Hybrid Score":     {"threshold_method": "percentile", "percentile": 92,
                                 "window_size": 200},
            "Isolation Forest": {"contamination": 0.12, "n_estimators": 200, "window_size": 50},
        },
    },
    {
        "category": "4. Mean shift / schimbare de regim",
        "key":   "realKnownCause/cpu_utilization_asg_misconfiguration.csv",
        "short": "cpu_util_misconfiguration",
        "desc":  ("CPU utilization AWS ASG — misconfigurare auto-scaling;\n"
                  "CPU rămâne susținut ridicat timp de ~5 zile (schimbare de regim)."),
        # 5d / ~90d ≈ 6%
        "algo_params": {
            "Rolling Z-Score":  {"threshold_method": "percentile", "percentile": 95,
                                 "window_size": 288},
            "Prediction Error": {"threshold_method": "percentile", "percentile": 95,
                                 "forecast_window": 200},
            "Hybrid Score":     {"threshold_method": "percentile", "percentile": 95,
                                 "window_size": 288},
            "Isolation Forest": {"contamination": 0.06, "n_estimators": 200, "window_size": 144},
        },
    },
    # ── 5. Anomalii contextuale / colective ─────────────────────────────────
    # art_daily_nojump: normal oscillation 18-88, but inside the window the series
    # stays LOW (mean=23 vs 43 outside, std=13 vs 28 outside). The daily PEAK is
    # ABSENT — values never reach 88 during the anomaly period.
    # Z-Score cannot detect this (all values are within the normal range 18-88).
    # IF detects it: rolling_mean and rolling_std are both systematically lower
    # than anything the model has seen during training → structurally anomalous.
    {
        "category": "5. Anomalii contextuale (IF excels)",
        "key":   "artificialWithAnomaly/art_daily_nojump.csv",
        "short": "art_daily_nojump",
        "desc":  ("Artificial daily pattern — vârful zilnic (peak) LIPSESTE\n"
                  "în fereastra de anomalie. Valorile rămân în banda inferioară\n"
                  "(mean=23 vs 43 normal, std=13 vs 28 normal). Z-Score eșuează\n"
                  "(valorile sunt în intervalul normal); IF detectează pattern-ul absent."),
        # window ≈ 10%; values in range but distributional structure is different
        "algo_params": {
            # Stat methods: values not extreme → won't detect → shows limitation
            "Rolling Z-Score":  {"threshold_method": "percentile", "percentile": 92,
                                 "window_size": 30},
            "Prediction Error": {"threshold_method": "percentile", "percentile": 92,
                                 "forecast_window": 50},   # MA history includes peaks → error > 0
            "Hybrid Score":     {"threshold_method": "percentile", "percentile": 92,
                                 "window_size": 30},
            # IF window=200 (17h): anomaly is 33h → center of window has purely anomalous features
            # rolling_mean≈23 (vs 43 normal) and rolling_std≈13 (vs 28) become clearly isolable
            "Isolation Forest": {"contamination": 0.10, "n_estimators": 200, "window_size": 200},
        },
    },
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_dataset(key: str) -> pd.DataFrame:
    path = NAB_DATA / key
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if 'timestamp' not in df.columns:
        df.columns = ['timestamp', 'value'] + list(df.columns[2:])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def nab_point_mask(df: pd.DataFrame, key: str) -> np.ndarray:
    windows = WINDOWS.get(key, [])
    mask = np.zeros(len(df), dtype=bool)
    for (s, e) in windows:
        mask |= (df['timestamp'] >= pd.Timestamp(s)) & (df['timestamp'] <= pd.Timestamp(e))
    return mask


def run_algo(algo, values, params):
    try:
        return algo.detect(values, **params)
    except Exception as ex:
        print(f"    [ERR] {algo.name}: {ex}")
        return None


def score_contrast(scores, nab_mask):
    """
    Return the ratio of mean score inside the NAB window vs outside.
    A ratio > 1 means the algorithm assigns higher anomaly scores in the window.
    Also returns the in/out mean scores for the annotation string.
    """
    if nab_mask.sum() == 0 or (~nab_mask).sum() == 0:
        return "n/a"
    s_in  = np.nanmean(scores[nab_mask])
    s_out = np.nanmean(scores[~nab_mask])
    ratio = s_in / s_out if s_out > 0 else float('inf')
    return f"score IN={s_in:.2f}  OUT={s_out:.2f}  ratio={ratio:.1f}x"


# ─── Plot ─────────────────────────────────────────────────────────────────────

DARK_BG   = '#0F1117'
PANEL_BG  = '#1A1D2E'
GRID_COL  = '#2A2D3E'
TEXT_COL  = '#C8CAD8'
WINDOW_COL= '#FF4444'


def plot_dataset(cfg: dict):
    key   = cfg["key"]
    short = cfg["short"]
    print(f"\n  {short}")

    df      = load_dataset(key)
    values  = df['value'].values
    ts      = df['timestamp']
    nab_mask = nab_point_mask(df, key)
    windows  = WINDOWS.get(key, [])
    extra    = cfg.get("extra_span")

    # Instantiate fresh detectors per dataset so window_size is respected
    algo_instances = {
        "Rolling Z-Score":  RollingStatsZScore(),
        "Prediction Error": PredictionErrorAnomaly(),
        "Hybrid Score":     HybridAnomalyScore(),
        "Isolation Forest": IsolationForestDetector(n_estimators=200),
    }

    results = {}
    for name, algo in algo_instances.items():
        params = cfg["algo_params"].get(name, {})
        results[name] = run_algo(algo, values, params)

    n_algos = len(algo_instances)
    fig = plt.figure(figsize=(18, 4 + 2.4 * n_algos))
    fig.patch.set_facecolor(DARK_BG)

    gs = GridSpec(
        n_algos + 1, 1, figure=fig,
        hspace=0.06, top=0.88, bottom=0.05, left=0.07, right=0.97,
        height_ratios=[2.2] + [1.0] * n_algos,
    )

    # ── Time-series panel ───────────────────────────────────────────────────
    ax_ts = fig.add_subplot(gs[0])
    ax_ts.set_facecolor(PANEL_BG)
    ax_ts.plot(ts, values, color='#E8E8F0', linewidth=0.7, alpha=0.9, zorder=2)

    for (s, e) in windows:
        ax_ts.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                      color=WINDOW_COL, alpha=0.20, zorder=0)

    # Optional extra annotation: extended flat region after NAB window
    if extra:
        ax_ts.axvspan(pd.Timestamp(extra[0]), pd.Timestamp(extra[1]),
                      color='#FFD600', alpha=0.15, zorder=0)
        ax_ts.axvline(pd.Timestamp(extra[0]), color='#FFD600',
                      linewidth=1.0, linestyle=':', alpha=0.7)

    # Scatter detected anomalies per algorithm
    for name, result in results.items():
        if result is None:
            continue
        detected = result.labels == 1
        if detected.any():
            ax_ts.scatter(ts[detected], values[detected],
                          color=ALGO_COLORS[name], s=14, zorder=3, alpha=0.75,
                          label=name)

    y_min, y_max = np.nanmin(values), np.nanmax(values)
    ypad = (y_max - y_min) * 0.08 or 1.0
    ax_ts.set_xlim(ts.iloc[0], ts.iloc[-1])
    ax_ts.set_ylim(y_min - ypad, y_max + ypad * 1.5)
    ax_ts.tick_params(axis='x', labelbottom=False, colors=TEXT_COL)
    ax_ts.tick_params(axis='y', colors=TEXT_COL, labelsize=8)
    ax_ts.set_ylabel("Valoare", color=TEXT_COL, fontsize=9)
    ax_ts.yaxis.grid(True, color=GRID_COL, linewidth=0.4)
    for sp in ax_ts.spines.values():
        sp.set_edgecolor(GRID_COL)

    nab_patch = mpatches.Patch(color=WINDOW_COL, alpha=0.5, label='Fereastră NAB (ground truth)')
    algo_patches = [mpatches.Patch(color=c, label=n) for n, c in ALGO_COLORS.items()]
    ax_ts.legend(
        handles=[nab_patch] + algo_patches,
        loc='upper right', fontsize=7.5, facecolor=PANEL_BG,
        edgecolor=GRID_COL, labelcolor=TEXT_COL,
        ncol=3, handlelength=1.1, columnspacing=1.0,
    )

    # ── Score panels ────────────────────────────────────────────────────────
    axes = []
    for i, (name, result) in enumerate(results.items()):
        ax = fig.add_subplot(gs[i + 1], sharex=ax_ts)
        ax.set_facecolor(PANEL_BG)
        axes.append(ax)

        color = ALGO_COLORS[name]

        for (s, e) in windows:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                       color=WINDOW_COL, alpha=0.13, zorder=0)
        if extra:
            ax.axvspan(pd.Timestamp(extra[0]), pd.Timestamp(extra[1]),
                       color='#FFD600', alpha=0.10, zorder=0)

        if result is None:
            ax.text(0.5, 0.5, 'Eroare', transform=ax.transAxes,
                    ha='center', va='center', color='red', fontsize=9)
        else:
            scores = result.scores
            # Shade detected regions
            detected = result.labels == 1
            ax.fill_between(ts, 0, scores, color=color, alpha=0.40, zorder=1)
            ax.plot(ts, scores, color=color, linewidth=0.65, zorder=2)
            ax.axhline(1.0, color=WINDOW_COL, linewidth=0.9,
                       linestyle='--', alpha=0.75, zorder=3)

            # Score contrast annotation (mean score IN window vs OUT)
            contrast = score_contrast(scores, nab_mask)
            ax.text(0.01, 0.88, contrast,
                    transform=ax.transAxes, fontsize=7,
                    color='#FFFFFF', va='top',
                    bbox=dict(facecolor='#00000055', edgecolor='none', pad=2))

        ax.set_ylim(-0.05, 1.20)
        is_last = (i == n_algos - 1)
        ax.tick_params(axis='x', labelbottom=is_last,
                       labelrotation=30 if is_last else 0,
                       colors=TEXT_COL, labelsize=8)
        ax.tick_params(axis='y', colors=TEXT_COL, labelsize=7)
        ax.set_ylabel(name, color=color, fontsize=8.5, labelpad=4, fontweight='bold')
        ax.yaxis.grid(True, color=GRID_COL, linewidth=0.4)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_COL)

    # ── Title ───────────────────────────────────────────────────────────────
    cat   = cfg["category"]
    desc  = cfg["desc"]
    fig.suptitle(
        f"{cat}  —  {short}\n{desc}",
        fontsize=9.5, color='#FFFFFF', y=0.965, va='top', linespacing=1.6,
        bbox=dict(facecolor=PANEL_BG, edgecolor='none', pad=5),
    )

    out = OUT_DIR / f"{short}.png"
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"    -> {out.name}")
    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Output: {OUT_DIR}\n")
    for cfg in DATASETS:
        try:
            plot_dataset(cfg)
        except Exception as ex:
            import traceback
            print(f"  [ERROR] {cfg['short']}: {ex}")
            traceback.print_exc()
    print("\nDone.")


if __name__ == "__main__":
    main()
