"""
Benchmark Hibrid Spark vs Pandas — Teza
=======================================
Part 1: O singura serie sintetica, marimi crescatoare
        (sesiunea Spark deja activa — fara cost JVM startup)
        → arata la ce dimensiune Spark devine competitiv

Part 2: N serii NAB procesate simultan
        Pandas: for-loop secvential
        Spark:  Window.partitionBy(series_id) — toate partitiile in paralel
        → arata avantajul la procesare multipla

Rezultate salvate in benchmark_spark_hybrid.csv
"""

import warnings, logging, os, time
warnings.filterwarnings("ignore")
logging.getLogger("py4j").setLevel(logging.ERROR)
logging.getLogger("pyspark").setLevel(logging.ERROR)
os.environ["PYSPARK_PYTHON"] = "spark-env\\Scripts\\python.exe"

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from nab_integration import NABIntegration

WIN_SIZE  = 20
N_REPEATS = 3   # repetitii per masurare — pastram minimul

NAB_SERIES = [
    "realAWSCloudwatch/ec2_cpu_utilization_825cc2",
    "realAWSCloudwatch/ec2_cpu_utilization_53ea38",
    "realAWSCloudwatch/ec2_cpu_utilization_5f5533",
    "realAWSCloudwatch/ec2_cpu_utilization_77c1ca",
    "realAWSCloudwatch/ec2_cpu_utilization_ac20cd",
    "realAWSCloudwatch/ec2_cpu_utilization_fe7f93",
    "realAWSCloudwatch/rds_cpu_utilization_cc0c53",
    "realAWSCloudwatch/rds_cpu_utilization_e47b3b",
    "realAWSCloudwatch/elb_request_count_8c0756",
    "artificialWithAnomaly/art_increase_spike_density",
    "artificialWithAnomaly/art_daily_flatmiddle",
    "artificialWithAnomaly/art_daily_jumpsdown",
    "artificialWithAnomaly/art_daily_jumpsup",
    "artificialWithAnomaly/art_daily_nojump",
    "artificialWithAnomaly/art_noisy",
    "realKnownCause/ambient_temperature_system_failure",
    "realKnownCause/cpu_utilization_asg_misconfiguration",
    "realKnownCause/ec2_request_latency_system_failure",
    "realKnownCause/machine_temperature_system_failure",
    "realKnownCause/nyc_taxi",
    "realTraffic/TravelTime_387",
    "realTraffic/TravelTime_451",
    "realTraffic/speed_6005",
    "realTraffic/speed_7578",
    "realTraffic/occupancy_6005",
]


def measure(fn, repeats=N_REPEATS):
    """Ruleaza fn de N ori si returneaza minimul (ms)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return min(times)


def pandas_rolling(data: np.ndarray):
    s = pd.Series(data)
    s.rolling(WIN_SIZE, min_periods=1).mean()
    s.rolling(WIN_SIZE, min_periods=2).std()


def spark_single(spark, data: np.ndarray):
    pdf = pd.DataFrame({"idx": np.arange(len(data), dtype=np.int64),
                         "value": data.astype(float)})
    sdf = spark.createDataFrame(pdf)
    w   = Window.orderBy("idx").rowsBetween(-(WIN_SIZE - 1), 0)
    (sdf
     .withColumn("rolling_mean", F.avg("value").over(w))
     .withColumn("rolling_std",  F.coalesce(F.stddev("value").over(w), F.lit(0.0)))
     .select("rolling_mean", "rolling_std")
     .collect())


def pandas_multi(dfs):
    for df in dfs:
        v = df["value"]
        v.rolling(WIN_SIZE, min_periods=1).mean()
        v.rolling(WIN_SIZE, min_periods=2).std()


def spark_multi(spark, dfs):
    parts = []
    for sid, df in enumerate(dfs):
        parts.append(pd.DataFrame({
            "series_id": sid,
            "idx":       np.arange(len(df), dtype=np.int64),
            "value":     df["value"].values.astype(float),
        }))
    combined = pd.concat(parts, ignore_index=True)
    sdf = spark.createDataFrame(combined)
    w   = (Window.partitionBy("series_id")
                 .orderBy("idx")
                 .rowsBetween(-(WIN_SIZE - 1), 0))
    (sdf
     .withColumn("rolling_mean", F.avg("value").over(w))
     .withColumn("rolling_std",  F.coalesce(F.stddev("value").over(w), F.lit(0.0)))
     .select("rolling_mean", "rolling_std")
     .collect())


def main():
    # ── Initializare Spark ────────────────────────────────────────────────────
    print("Initializing Spark session...")
    t_jvm_start = time.perf_counter()
    spark = (SparkSession.builder
             .appName("BenchmarkHybrid")
             .master("local[8]")
             .config("spark.sql.shuffle.partitions", "16")
             .config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    t_jvm = (time.perf_counter() - t_jvm_start) * 1000

    print(f"Spark {spark.version} — 8 cores")
    print(f"JVM startup time: {t_jvm:.0f} ms  ({t_jvm/1000:.1f}s)\n")

    # Warm-up: prima operatie Spark e intotdeauna mai lenta
    _wdf = spark.createDataFrame(pd.DataFrame({"a": range(500), "b": range(500)}))
    _wdf.count()
    print("Spark warmed up.\n")

    rows = []

    # ── PART 1: O singura serie, marimi crescatoare ───────────────────────────
    print("=" * 65)
    print("PART 1 — O singura serie sintetica, marimi crescatoare")
    print(f"  (JVM deja pornit; masuram doar calculul)")
    print("=" * 65)
    print(f"{'Puncte':>10}  {'Pandas (ms)':>12}  {'Spark (ms)':>11}  {'Spark/Pandas':>13}")
    print("-" * 52)

    rng   = np.random.default_rng(42)
    sizes = [5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]

    for size in sizes:
        data = rng.normal(50, 10, size)

        t_pd = measure(lambda: pandas_rolling(data))
        t_sp = measure(lambda: spark_single(spark, data))
        ratio = t_sp / t_pd

        print(f"{size:>10,}  {t_pd:>12.1f}  {t_sp:>11.1f}  {ratio:>12.1f}x")
        rows.append({"part": 1, "n_series": 1, "total_points": size,
                     "pandas_ms": round(t_pd, 2), "spark_ms": round(t_sp, 2),
                     "speedup": round(t_pd / t_sp, 3), "jvm_startup_ms": round(t_jvm, 0)})

    # ── PART 2: N serii NAB in paralel ───────────────────────────────────────
    print()
    print("=" * 65)
    print("PART 2 — N serii NAB: Pandas secvential vs Spark paralel")
    print(f"  (Window.partitionBy — fiecare serie pe alt core)")
    print("=" * 65)
    print(f"{'N serii':>8}  {'Total pct':>10}  {'Pandas (ms)':>12}  "
          f"{'Spark (ms)':>11}  {'Speedup':>9}")
    print("-" * 57)

    nab = NABIntegration()
    loaded = {}
    for name in NAB_SERIES:
        try:
            df = nab.load_data(name)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            loaded[name] = df
        except Exception:
            pass

    n_series_list = [1, 3, 5, 10, 15, 25]

    for n in n_series_list:
        dfs         = list(loaded.values())[:n]
        total_pts   = sum(len(d) for d in dfs)

        t_pd = measure(lambda: pandas_multi(dfs))
        t_sp = measure(lambda: spark_multi(spark, dfs))
        speedup = t_pd / t_sp

        marker = " ← Spark castiga" if speedup > 1 else ""
        print(f"{n:>8}  {total_pts:>10,}  {t_pd:>12.1f}  {t_sp:>11.1f}  "
              f"{speedup:>8.2f}x{marker}")
        rows.append({"part": 2, "n_series": n, "total_points": total_pts,
                     "pandas_ms": round(t_pd, 2), "spark_ms": round(t_sp, 2),
                     "speedup": round(speedup, 3), "jvm_startup_ms": round(t_jvm, 0)})

    # ── Rezumat ───────────────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print(f"JVM startup (cost fix, o singura data): {t_jvm:.0f} ms")
    df_out = pd.DataFrame(rows)
    df_out.to_csv("benchmark_spark_hybrid.csv", index=False)
    print("Salvat: benchmark_spark_hybrid.csv")

    spark.stop()


if __name__ == "__main__":
    main()
