"""PySpark pipeline: same logic as ``pipeline_pandas`` but distributed-ready.

Cleaning and feature engineering use window functions partitioned by ``test_id`` so the
job scales to millions of rows. Results are written as Parquet and read back with
Pandas for the (in-memory) Isolation Forest stage.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .config import (
    FILL_LIMIT,
    HARD_BRAKE_THRESHOLD_MS2,
    PROCESSED_DIR,
    RAW_DIR,
    ROLL_WINDOW,
    SENSOR_RANGES,
    SENSORS,
)


def get_spark(app_name: str = "vehicle-test-analyzer"):
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.master(os.environ.get("VTA_SPARK_MASTER", "local[*]"))
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", os.environ.get("VTA_SPARK_MEM", "2g"))
        .getOrCreate()
    )


def read_measurements(spark, path: Path | str):
    from pyspark.sql import types as T

    schema = T.StructType(
        [T.StructField("test_id", T.StringType()), T.StructField("timestamp", T.StringType())]
        + [T.StructField(s, T.DoubleType()) for s in SENSORS]
    )
    return spark.read.csv(str(path), header=True, schema=schema)


def clean_and_featurize(sdf):
    """Spark equivalent of ``pipeline_pandas.clean_and_featurize``. Returns (features, quality)."""
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    sdf = sdf.withColumn("timestamp", F.to_timestamp("timestamp"))

    missing_expr = sum(F.col(s).isNull().cast("int") for s in SENSORS)
    raw_stats = sdf.groupBy("test_id").agg(
        F.count("*").alias("n_rows_raw"), F.sum(missing_expr).alias("n_cells_missing_raw")
    )

    sdf = sdf.dropDuplicates(["test_id", "timestamp"])

    # Out-of-range -> NULL
    oor_expr = sum(
        F.when((F.col(s) < lo) | (F.col(s) > hi), 1).otherwise(0) for s, (lo, hi) in SENSOR_RANGES.items()
    )
    sdf = sdf.withColumn("_oor", oor_expr)
    for s, (lo, hi) in SENSOR_RANGES.items():
        sdf = sdf.withColumn(s, F.when((F.col(s) < lo) | (F.col(s) > hi), F.lit(None)).otherwise(F.col(s)))

    w = Window.partitionBy("test_id").orderBy("timestamp")
    w_fill = w.rowsBetween(-FILL_LIMIT, 0)
    w_roll = w.rowsBetween(-(ROLL_WINDOW - 1), 0)
    w_part = Window.partitionBy("test_id")

    # Forward fill of short gaps (<= FILL_LIMIT consecutive samples)
    for s in SENSORS:
        sdf = sdf.withColumn(s, F.last(s, ignorenulls=True).over(w_fill))

    ts = F.unix_timestamp("timestamp")
    sdf = sdf.withColumn("t_s", (ts - F.min(ts).over(w_part)).cast("double"))
    dt = F.col("t_s") - F.lag("t_s", 1).over(w)
    sdf = sdf.withColumn(
        "accel_ms2", F.coalesce((F.col("speed_kmh") - F.lag("speed_kmh", 1).over(w)) / 3.6 / dt, F.lit(0.0))
    )
    sdf = sdf.withColumn(
        "engine_temp_slope",
        F.coalesce(
            (F.col("engine_temp_c") - F.lag("engine_temp_c", ROLL_WINDOW).over(w))
            / (F.col("t_s") - F.lag("t_s", ROLL_WINDOW).over(w)),
            F.lit(0.0),
        ),
    )
    sdf = sdf.withColumn("rpm_roll_std", F.coalesce(F.stddev_samp("rpm").over(w_roll), F.lit(0.0)))
    sdf = sdf.withColumn("vibration_roll_mean", F.avg("vibration_g").over(w_roll))
    sdf = sdf.withColumn(
        "fuel_l100km", F.when(F.col("speed_kmh") >= 5, F.col("fuel_rate_lph") / F.col("speed_kmh") * 100.0)
    )
    valid = F.lit(True)
    for s in SENSORS:
        valid = valid & F.col(s).isNotNull()
    sdf = sdf.withColumn("is_valid", valid)

    missing_after = sum(F.col(s).isNull().cast("int") for s in SENSORS)
    qual = sdf.groupBy("test_id").agg(
        F.count("*").alias("n_rows_dedup"),
        F.sum("_oor").alias("n_cells_out_of_range"),
        F.sum(missing_after).alias("n_cells_missing_after_fill"),
        F.sum(F.col("is_valid").cast("int")).alias("n_rows_valid"),
    )
    quality = (
        raw_stats.join(qual, "test_id")
        .withColumn("n_duplicates_removed", F.col("n_rows_raw") - F.col("n_rows_dedup"))
        .withColumn("valid_pct", F.round(F.col("n_rows_valid") / F.col("n_rows_dedup") * 100.0, 2))
        .select(
            "test_id",
            "n_rows_raw",
            "n_duplicates_removed",
            "n_cells_missing_raw",
            "n_cells_out_of_range",
            "n_cells_missing_after_fill",
            "n_rows_valid",
            "valid_pct",
        )
    )
    features = sdf.drop("_oor").orderBy("test_id", "t_s")
    return features, quality


def compute_kpis(features):
    from pyspark.sql import functions as F

    d = features.filter(F.col("is_valid"))
    agg = d.groupBy("test_id").agg(
        F.count("*").alias("n_rows"),
        (F.sum("speed_kmh") / 3600.0).alias("distance_km"),
        (F.sum("fuel_rate_lph") / 3600.0).alias("_fuel_l"),
        F.avg("speed_kmh").alias("avg_speed_kmh"),
        F.max("speed_kmh").alias("max_speed_kmh"),
        F.avg("rpm").alias("avg_rpm"),
        F.max("rpm").alias("max_rpm"),
        F.max("engine_temp_c").alias("max_engine_temp_c"),
        (F.avg((F.col("speed_kmh") < 1).cast("double")) * 100.0).alias("idle_pct"),
        F.max("accel_ms2").alias("max_accel_ms2"),
        F.sum((F.col("accel_ms2") < HARD_BRAKE_THRESHOLD_MS2).cast("int")).alias("hard_brake_count"),
        F.avg("vibration_g").alias("avg_vibration_g"),
        F.min("oil_pressure_bar").alias("min_oil_pressure_bar"),
        F.min("battery_v").alias("min_battery_v"),
    )
    return agg.withColumn(
        "avg_fuel_l100km", F.when(F.col("distance_km") > 0, F.col("_fuel_l") / F.col("distance_km") * 100.0)
    ).drop("_fuel_l")


def run_spark(raw_dir: Path | str = RAW_DIR, out_dir: Path | str = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Run the Spark pipeline end to end; writes Parquet and returns Pandas frames."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        sdf = read_measurements(spark, raw_dir / "measurements.csv")
        features, quality = clean_and_featurize(sdf)
        features = features.cache()
        kpis = compute_kpis(features)
        results = {}
        for name, df in {"features": features, "quality": quality, "kpis": kpis}.items():
            path = out_dir / f"{name}_spark.parquet"
            df.coalesce(1).write.mode("overwrite").parquet(str(path))
            results[name] = pd.read_parquet(path)
    finally:
        spark.stop()

    results["features"] = results["features"].sort_values(["test_id", "t_s"]).reset_index(drop=True)
    results["quality"] = results["quality"].sort_values("test_id").reset_index(drop=True)
    results["kpis"] = results["kpis"].sort_values("test_id").reset_index(drop=True)
    for name in ("tests", "vehicles", "anomaly_labels"):
        f = raw_dir / f"{name}.csv"
        results[name] = pd.read_csv(f) if f.exists() else pd.DataFrame()
    return results
