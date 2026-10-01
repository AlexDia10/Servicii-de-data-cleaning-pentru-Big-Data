"""
Generare diagrame arhitectura pentru teza de licenta.
Salvare in output/validation/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "output" / "validation"
OUT.mkdir(parents=True, exist_ok=True)

# ── Paleta de culori ────────────────────────────────────────────────────────
C = {
    "input"     : "#1565C0",   # albastru inchis
    "loader"    : "#0277BD",   # albastru mediu
    "algo"      : "#00695C",   # verde inchis
    "algo_sub"  : "#26A69A",   # verde mediu
    "spark"     : "#E65100",   # portocaliu
    "eval"      : "#6A1B9A",   # violet
    "output"    : "#2E7D32",   # verde
    "dash"      : "#37474F",   # gri inchis
    "arrow"     : "#455A64",
    "bg"        : "#FAFAFA",
    "white"     : "#FFFFFF",
}

def box(ax, x, y, w, h, label, color, fontsize=10, text_color="white",
        sublabel=None, corner_radius=0.04):
    """Deseneaza un dreptunghi rotunjit cu eticheta."""
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad={corner_radius}",
        facecolor=color, edgecolor="white", linewidth=1.5, zorder=3
    )
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + h*0.12, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
        ax.text(x, y - h*0.22, sublabel, ha="center", va="center",
                fontsize=fontsize - 2, color=text_color, alpha=0.88, zorder=4,
                style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)

def arrow(ax, x0, y0, x1, y1, color="#455A64", lw=1.8, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)

def label_arrow(ax, x0, y0, x1, y1, text, color="#455A64"):
    arrow(ax, x0, y0, x1, y1, color=color)
    mx, my = (x0+x1)/2, (y0+y1)/2
    ax.text(mx + 0.02, my, text, ha="left", va="center", fontsize=7.5,
            color="#37474F", style="italic", zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# 1. arhitectura_sistem.png
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")

ax.text(6, 7.6, "Arhitectura Sistemului de Detecție a Anomaliilor",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#1A237E")

# ── Strat 1: INTRARE (y=6.5) ─────────────────────────────────────────────
ax.text(6, 6.95, "STRAT INTRARE", ha="center", va="center",
        fontsize=8, color="#78909C", fontweight="bold", style="italic")
box(ax, 3.2, 6.5, 2.2, 0.7, "CSV Upload\n(Fișier propriu)", C["input"], fontsize=9)
box(ax, 6.0, 6.5, 2.2, 0.7, "NAB Corpus\n(58 serii oficiale)", C["input"], fontsize=9)
box(ax, 8.8, 6.5, 2.2, 0.7, "Dataset Custom\n(data/src/)", C["input"], fontsize=9)

# ── Strat 2: INCARCARE DATE (y=5.4) ──────────────────────────────────────
ax.text(6, 5.85, "STRAT PREPROCESARE", ha="center", va="center",
        fontsize=8, color="#78909C", fontweight="bold", style="italic")
box(ax, 6.0, 5.4, 5.8, 0.7,
    "DataLoader",
    C["loader"], fontsize=10,
    sublabel="load_csv → parse_timestamps → sort → handle_missing → normalize")

# ── Strat 3: DETECȚIE (y=4.0) ────────────────────────────────────────────
ax.text(6, 4.75, "STRAT DETECȚIE", ha="center", va="center",
        fontsize=8, color="#78909C", fontweight="bold", style="italic")

# Motor principal
box(ax, 3.4, 4.1, 2.6, 1.1,
    "AnomalyDetectionEngine", C["algo"], fontsize=9)

# 5 algoritmi sub motor
algo_names = [
    ("Rolling\nZ-Score", 0.85),
    ("Prediction\nError",  2.15),
    ("Hybrid\nScore",      3.45),
    ("Isolation\nForest",  4.75),
    ("Mean\nShift",        6.05),
]
for name, xpos in algo_names:
    box(ax, xpos, 4.05, 1.08, 0.72, name, C["algo_sub"], fontsize=7.5)
    arrow(ax, 3.4 - 1.3 + xpos*0 + 0, 4.55, xpos, 4.42,
          color=C["algo"], lw=1.2)

# Sageti din engine spre algoritmi
arrow(ax, 2.1, 4.1, 0.85, 4.1, color=C["algo"], lw=1.2)
arrow(ax, 2.1, 4.1, 2.15, 4.1, color=C["algo"], lw=1.2)
arrow(ax, 2.1, 4.1, 3.45, 4.1, color=C["algo"], lw=1.2)
arrow(ax, 4.7, 4.1, 4.75, 4.1, color=C["algo"], lw=1.2)
arrow(ax, 4.7, 4.1, 6.05, 4.1, color=C["algo"], lw=1.2)

# Spark Engine (dreapta)
box(ax, 9.8, 4.1, 3.0, 1.1,
    "SparkEngine", C["spark"], fontsize=10,
    sublabel="PySpark local[*]\nRolling distribuit")

# ── Strat 4: EVALUARE (y=2.8) ────────────────────────────────────────────
ax.text(6, 3.2, "STRAT EVALUARE", ha="center", va="center",
        fontsize=8, color="#78909C", fontweight="bold", style="italic")
box(ax, 3.5, 2.7, 3.0, 0.75,
    "Benchmark", C["eval"], fontsize=10,
    sublabel="Precision / Recall / F1")
box(ax, 8.5, 2.7, 3.2, 0.75,
    "NAB Sweeper", C["eval"], fontsize=10,
    sublabel="TP/FP/FN → scor normalizat")

# ── Strat 5: IEȘIRE (y=1.5) ──────────────────────────────────────────────
ax.text(6, 2.05, "STRAT IEȘIRE", ha="center", va="center",
        fontsize=8, color="#78909C", fontweight="bold", style="italic")
box(ax, 2.0, 1.5, 2.8, 0.72, "Dashboard Streamlit\n(5 tab-uri)", C["dash"], fontsize=9)
box(ax, 5.3, 1.5, 2.8, 0.72, "results_*.csv\nnab_scores.csv", C["output"], fontsize=9)
box(ax, 8.5, 1.5, 2.8, 0.72, "output/validation/\n(grafice PNG)", C["output"], fontsize=9)

# ── Săgeți verticale principale ──────────────────────────────────────────
arrow(ax, 3.2, 6.14, 4.0, 5.75)
arrow(ax, 6.0, 6.14, 6.0, 5.75)
arrow(ax, 8.8, 6.14, 8.0, 5.75)
arrow(ax, 6.0, 5.03, 6.0, 4.65)      # DataLoader → Engine
arrow(ax, 6.0, 5.03, 9.8, 4.65)      # DataLoader → Spark
arrow(ax, 3.5, 3.55, 3.5, 3.08)      # Engine → Benchmark
arrow(ax, 9.8, 3.55, 8.5, 3.08)      # Spark → NAB Sweeper
arrow(ax, 3.5, 2.32, 2.0, 1.86)      # Benchmark → Dashboard
arrow(ax, 3.5, 2.32, 5.3, 1.86)      # Benchmark → CSV
arrow(ax, 8.5, 2.32, 8.5, 1.86)      # NAB Sweeper → PNG

# Legenda
legend_items = [
    mpatches.Patch(color=C["input"],    label="Surse de date"),
    mpatches.Patch(color=C["loader"],   label="Preprocesare"),
    mpatches.Patch(color=C["algo"],     label="Motor detecție"),
    mpatches.Patch(color=C["algo_sub"], label="Algoritmi"),
    mpatches.Patch(color=C["spark"],    label="Procesare distribuită"),
    mpatches.Patch(color=C["eval"],     label="Evaluare"),
    mpatches.Patch(color=C["output"],   label="Ieșiri"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=8,
          framealpha=0.85, ncol=4, bbox_to_anchor=(0.0, 0.0))

plt.tight_layout()
plt.savefig(OUT / "arhitectura_sistem.png", dpi=150, bbox_inches="tight",
            facecolor=C["bg"])
plt.close()
print("Salvat: arhitectura_sistem.png")


# ══════════════════════════════════════════════════════════════════════════════
# 2. flux_date.png
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")

ax.text(6, 7.65, "Fluxul de Date — De la CSV la Vizualizare",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#1A237E")

# Culorile pe faze
PH = {
    "in"    : "#1565C0",
    "prep"  : "#0277BD",
    "algo"  : "#00695C",
    "eval"  : "#6A1B9A",
    "out"   : "#2E7D32",
    "dec"   : "#F57F17",   # decizie (romb)
}

BW, BH = 3.2, 0.60   # box width / height standard
BH2 = 0.52

def step(ax, x, y, label, color, w=BW, h=BH, sublabel=None):
    box(ax, x, y, w, h, label, color, fontsize=9, sublabel=sublabel)

def diamond(ax, x, y, w, h, label, color):
    """Romb pentru decizie."""
    pts = [(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)]
    poly = plt.Polygon(pts, facecolor=color, edgecolor="white", lw=1.5, zorder=3)
    ax.add_patch(poly)
    ax.text(x, y, label, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="white", zorder=4)

def arr(ax, x0, y0, x1, y1, txt="", color="#455A64"):
    arrow(ax, x0, y0, x1, y1, color=color)
    if txt:
        mx, my = (x0+x1)/2, (y0+y1)/2
        dx = 0.1 if x1 >= x0 else -0.1
        ax.text(mx + dx, my, txt, ha="left" if dx > 0 else "right",
                va="center", fontsize=7.5, color="#37474F", style="italic", zorder=5)

# ── Coloana stânga: pipeline preprocesare ────────────────────────────────
X_MAIN = 3.0
y = 7.1
step(ax, X_MAIN, y, "CSV Input\n(timestamp, value)", PH["in"], w=3.0, h=0.58)

y -= 0.80
step(ax, X_MAIN, y, "load_csv()", PH["prep"], w=3.0, h=BH2)
arr(ax, X_MAIN, y+0.60, X_MAIN, y+0.32)

y -= 0.72
step(ax, X_MAIN, y, "parse_timestamps()", PH["prep"], w=3.0, h=BH2)
arr(ax, X_MAIN, y+0.60, X_MAIN, y+0.32)

y -= 0.72
step(ax, X_MAIN, y, "sort_data()", PH["prep"], w=3.0, h=BH2)
arr(ax, X_MAIN, y+0.60, X_MAIN, y+0.32)

y -= 0.72
step(ax, X_MAIN, y, "handle_missing_values()\n(interpolate / ffill / mean)", PH["prep"], w=3.2, h=BH2)
arr(ax, X_MAIN, y+0.60, X_MAIN, y+0.32)

# decizie normalizare
y -= 0.80
diamond(ax, X_MAIN, y, 2.8, 0.55, "normalize?", PH["dec"])
arr(ax, X_MAIN, y+0.72, X_MAIN, y+0.28)

# ramura DA
step(ax, 1.1, y, "StandardScaler\n/ MinMaxScaler", PH["prep"], w=2.0, h=BH2)
ax.annotate("", xy=(1.1, y+0.05), xytext=(X_MAIN-1.4, y),
            arrowprops=dict(arrowstyle="-|>", color=PH["dec"], lw=1.5), zorder=2)
ax.text(1.7, y+0.15, "DA", fontsize=7.5, color=PH["dec"], fontweight="bold")

# ramura NU → coboară direct
arr(ax, X_MAIN, y-0.28, X_MAIN, y-0.55, "NU", color=PH["dec"])

# ── Detecție ─────────────────────────────────────────────────────────────
y -= 0.90
step(ax, X_MAIN, y, "detect_single(algo, values, **params)", PH["algo"], w=4.0, h=BH2)
arr(ax, X_MAIN, y+0.60, X_MAIN, y+0.32)

# 5 algoritmi in evantai
y -= 0.90
algo_list = [
    ("Rolling\nZ-Score",  0.9),
    ("Prediction\nError", 2.1),
    ("Hybrid\nScore",     3.3),
    ("Isolation\nForest", 4.5),
    ("Mean\nShift",       5.7),
]
for name, xpos in algo_list:
    box(ax, xpos, y, 1.05, 0.65, name, PH["algo"], fontsize=7.5)
    ax.annotate("", xy=(xpos, y+0.33), xytext=(X_MAIN, y+0.92),
                arrowprops=dict(arrowstyle="-|>", color=PH["algo"],
                                lw=1.1, connectionstyle="arc3,rad=0.0"), zorder=2)

# ── AnomalyResult ─────────────────────────────────────────────────────────
y -= 0.90
step(ax, X_MAIN, y, "AnomalyResult\n(scores, labels, anomaly_type, threshold, exec_time)",
     PH["algo"], w=5.8, h=BH2)
# sageti din algoritmi spre result
for _, xpos in algo_list:
    ax.annotate("", xy=(X_MAIN, y+0.30), xytext=(xpos, y+0.60),
                arrowprops=dict(arrowstyle="-|>", color=PH["algo"],
                                lw=1.1, connectionstyle="arc3,rad=0.0"), zorder=2)

# ── Evaluare ─────────────────────────────────────────────────────────────
y -= 0.82
step(ax, 1.5, y, "Benchmark.evaluate()\nTP / FP / FN → F1", PH["eval"], w=2.8, h=BH2)
step(ax, 4.8, y, "NAB Sweeper\ncalcSweepScore → NAB Score", PH["eval"], w=3.0, h=BH2)
arr(ax, 1.5, y+0.64, 1.5, y+0.32)
arr(ax, 4.0, y+0.64, 4.8, y+0.32)

# ── Ieșire ─────────────────────────────────────────────────────────────
y -= 0.84
step(ax, 1.5, y, "Dashboard Streamlit\nPlotly (tab-uri)", PH["out"], w=2.8, h=BH2)
step(ax, 4.8, y, "results_*.csv\nnab_scores.csv", PH["out"], w=3.0, h=BH2)
step(ax, 8.2, y, "output/PNG\ngrafice validate", PH["out"], w=2.6, h=BH2)
arr(ax, 1.5, y+0.64, 1.5, y+0.32)
arr(ax, 4.8, y+0.64, 4.8, y+0.32)
arr(ax, 4.8, y+0.22, 8.2, y+0.22)

# Coloana dreapta: note contextuale
ax.text(9.6, 6.9, "Format acceptat:\n• timestamp (datetime)\n• value (numeric)",
        ha="left", va="center", fontsize=8, color="#455A64",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#90CAF9"))

ax.text(9.6, 5.4, "Valori lipsă:\n• interpolate (default)\n• ffill / mean / drop",
        ha="left", va="center", fontsize=8, color="#455A64",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#90CAF9"))

ax.text(9.6, 3.8, "Parametri detecție:\n• threshold_method\n• k_std / contamination\n• window_size",
        ha="left", va="center", fontsize=8, color="#455A64",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor="#A5D6A7"))

ax.text(9.6, 2.4, "NAB Profiles:\n• Standard (fpW=0.11)\n• reward_low_FP\n• reward_low_FN",
        ha="left", va="center", fontsize=8, color="#455A64",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F3E5F5", edgecolor="#CE93D8"))

# Legenda faze
legend_items = [
    mpatches.Patch(color=PH["in"],   label="Intrare"),
    mpatches.Patch(color=PH["prep"], label="Preprocesare"),
    mpatches.Patch(color=PH["algo"], label="Detecție"),
    mpatches.Patch(color=PH["eval"], label="Evaluare"),
    mpatches.Patch(color=PH["out"],  label="Ieșire"),
    mpatches.Patch(color=PH["dec"],  label="Decizie"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=8.5,
          framealpha=0.85, ncol=3, bbox_to_anchor=(0.0, 0.0))

plt.tight_layout()
plt.savefig(OUT / "flux_date.png", dpi=150, bbox_inches="tight",
            facecolor=C["bg"])
plt.close()
print("Salvat: flux_date.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3. dependente_module.png
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(12, 8))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")

ax.text(6, 7.65, "Dependențe între Module Python",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#1A237E")

# Culori per rol
MC = {
    "core"   : "#1565C0",   # modul central
    "ui"     : "#37474F",   # interfata
    "algo"   : "#00695C",   # algoritmi
    "infra"  : "#4527A0",   # infrast. (spark, bench)
    "nab"    : "#6A1B9A",   # NAB-specific
    "util"   : "#5D4037",   # utilitare
    "ext"    : "#546E7A",   # librarii externe (gri-albastru)
    "eval"   : "#880E4F",   # evaluare
}

# ── Noduri ───────────────────────────────────────────────────────────────
nodes = {
    # (x, y, label, culoare, latime, inaltime)
    "dashboard.py"         : (6.0,  6.8, "dashboard.py",          MC["ui"],    2.6, 0.60),
    "app.py"               : (9.8,  6.8, "app.py",                MC["ui"],    1.8, 0.60),
    "anomaly_algorithms.py": (6.0,  5.2, "anomaly_algorithms.py", MC["algo"],  2.8, 0.60),
    "data_loader.py"       : (2.2,  5.2, "data_loader.py",        MC["core"],  2.4, 0.60),
    "spark_engine.py"      : (9.8,  5.2, "spark_engine.py",       MC["infra"], 2.4, 0.60),
    "benchmark.py"         : (2.2,  3.6, "benchmark.py",          MC["eval"],  2.4, 0.60),
    "utils.py"             : (6.0,  3.6, "utils.py",              MC["util"],  2.0, 0.60),
    "nab_integration.py"   : (9.8,  3.6, "nab_integration.py",    MC["nab"],   2.4, 0.60),
    "nab_score_compute.py" : (6.0,  2.0, "nab_score_compute.py",  MC["nab"],   2.6, 0.60),
    "run_evaluation.py"    : (2.2,  2.0, "run_evaluation.py",     MC["eval"],  2.4, 0.60),
    "diagnostic.py"        : (9.8,  2.0, "diagnostic.py",         MC["eval"],  2.2, 0.60),

    # Librarii externe
    "sklearn"              : (1.0,  6.5, "scikit-learn",          MC["ext"],   1.8, 0.50),
    "pyspark"              : (11.0, 4.3, "PySpark",               MC["ext"],   1.6, 0.50),
    "plotly"               : (1.0,  7.3, "Plotly",                MC["ext"],   1.6, 0.50),
    "pandas_np"            : (6.0,  4.2, "pandas / numpy",        MC["ext"],   2.0, 0.50),
    "streamlit"            : (3.8,  7.4, "Streamlit",             MC["ext"],   1.8, 0.50),
}

# Deseneaza toate nodurile
for key, (x, y, label, color, w, h) in nodes.items():
    box(ax, x, y, w, h, label, color, fontsize=8.5)

# ── Dependente (sagetile) ─────────────────────────────────────────────
#  (sursa, destinatie, eticheta_optionala)
deps = [
    # dashboard.py importa:
    ("dashboard.py", "data_loader.py",          ""),
    ("dashboard.py", "anomaly_algorithms.py",   ""),
    ("dashboard.py", "spark_engine.py",         ""),
    ("dashboard.py", "benchmark.py",            ""),
    ("dashboard.py", "utils.py",                ""),
    ("dashboard.py", "streamlit",               ""),
    ("dashboard.py", "plotly",                  ""),

    # app.py importa:
    ("app.py", "anomaly_algorithms.py",         ""),
    ("app.py", "data_loader.py",               ""),
    ("app.py", "streamlit",                    ""),

    # anomaly_algorithms.py importa:
    ("anomaly_algorithms.py", "sklearn",        "IsolationForest"),
    ("anomaly_algorithms.py", "pandas_np",      ""),

    # data_loader.py importa:
    ("data_loader.py", "utils.py",             ""),
    ("data_loader.py", "sklearn",              "StandardScaler"),
    ("data_loader.py", "pandas_np",           ""),

    # spark_engine.py importa:
    ("spark_engine.py", "anomaly_algorithms.py", "AnomalyResult"),
    ("spark_engine.py", "utils.py",             ""),
    ("spark_engine.py", "pyspark",             ""),

    # benchmark.py importa:
    ("benchmark.py", "anomaly_algorithms.py",  ""),
    ("benchmark.py", "utils.py",              ""),

    # nab_integration.py importa:
    ("nab_integration.py", "anomaly_algorithms.py", ""),

    # nab_score_compute.py importa:
    ("nab_score_compute.py", "anomaly_algorithms.py", ""),
    ("nab_score_compute.py", "pandas_np",            ""),

    # run_evaluation.py importa:
    ("run_evaluation.py", "anomaly_algorithms.py", ""),
    ("run_evaluation.py", "benchmark.py",          ""),
    ("run_evaluation.py", "data_loader.py",        ""),

    # diagnostic.py importa:
    ("diagnostic.py", "anomaly_algorithms.py", ""),
    ("diagnostic.py", "spark_engine.py",       ""),
]

# Desenare sagetile cu curbe usoare
def get_center(key):
    x, y, *_ = nodes[key]
    return x, y

for src, dst, lbl in deps:
    x0, y0 = get_center(src)
    x1, y1 = get_center(dst)
    # Calcul unghi pentru a evita suprapunere
    dx, dy = x1 - x0, y1 - y0
    dist = (dx**2 + dy**2)**0.5
    # Offset la capete (margine box)
    if dist > 0:
        sx = nodes[src][4] / 2 + 0.05  # latime sursa (index 4)
        sy = nodes[src][5] / 2 + 0.05  # inaltime sursa (index 5)
        dx_n, dy_n = dx/dist, dy/dist
        # estimam offset la sursa si destinatie
        x0e = x0 + dx_n * max(sx, abs(dx_n) * sx + abs(dy_n) * sy) * 0.85
        y0e = y0 + dy_n * max(sy, abs(dx_n) * sx + abs(dy_n) * sy) * 0.85
        dx2, dy2 = x0 - x1, y0 - y1
        dist2 = (dx2**2 + dy2**2)**0.5
        dx2n, dy2n = dx2/dist2, dy2/dist2
        dw2 = nodes[dst][4] / 2 + 0.05  # latime destinatie
        dh2 = nodes[dst][5] / 2 + 0.05  # inaltime destinatie
        x1e = x1 + dx2n * max(dw2, abs(dx2n)*dw2 + abs(dy2n)*dh2) * 0.85
        y1e = y1 + dy2n * max(dh2, abs(dx2n)*dw2 + abs(dy2n)*dh2) * 0.85
    else:
        x0e, y0e, x1e, y1e = x0, y0, x1, y1

    col = "#90A4AE" if nodes[dst][3] == MC["ext"] else "#78909C"
    lw  = 1.0 if nodes[dst][3] == MC["ext"] else 1.4
    style = "->" if nodes[dst][3] == MC["ext"] else "-|>"

    ax.annotate("", xy=(x1e, y1e), xytext=(x0e, y0e),
                arrowprops=dict(arrowstyle=style, color=col, lw=lw,
                                connectionstyle="arc3,rad=0.08"),
                zorder=2)
    if lbl:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx, my, lbl, ha="center", va="bottom", fontsize=6.5,
                color="#546E7A", style="italic", zorder=5)

# Legenda roluri
legend_items = [
    mpatches.Patch(color=MC["ui"],    label="Interfață (dashboard/app)"),
    mpatches.Patch(color=MC["algo"],  label="Algoritmi"),
    mpatches.Patch(color=MC["core"],  label="Date (DataLoader)"),
    mpatches.Patch(color=MC["infra"], label="Infrastructură (Spark)"),
    mpatches.Patch(color=MC["eval"],  label="Evaluare / Benchmark"),
    mpatches.Patch(color=MC["nab"],   label="NAB-specific"),
    mpatches.Patch(color=MC["util"],  label="Utilitare"),
    mpatches.Patch(color=MC["ext"],   label="Librării externe"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=8,
          framealpha=0.88, ncol=4, bbox_to_anchor=(0.0, 0.0))

plt.tight_layout()
plt.savefig(OUT / "dependente_module.png", dpi=150, bbox_inches="tight",
            facecolor=C["bg"])
plt.close()
print("Salvat: dependente_module.png")

print("\nToate cele 3 diagrame salvate in output/validation/")
