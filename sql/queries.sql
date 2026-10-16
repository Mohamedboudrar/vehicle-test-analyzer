-- Analytical queries. Each block starts with "-- name: <id>" followed by "-- desc: <text>".
-- They are used by the dashboard's SQL tab and by `python -m vta.sql_report`.

-- name: fleet_fuel_ranking
-- desc: Vehicles ranked by average fuel consumption within their fuel type (window RANK)
SELECT v.vehicle_id, v.model, v.fuel_type,
       COUNT(*)                                   AS n_tests,
       ROUND(AVG(k.avg_fuel_l100km), 2)           AS avg_l100km,
       RANK() OVER (PARTITION BY v.fuel_type ORDER BY AVG(k.avg_fuel_l100km)) AS rank_in_fuel_type
FROM test_kpis k
JOIN tests t    ON t.test_id = k.test_id
JOIN vehicles v ON v.vehicle_id = t.vehicle_id
GROUP BY v.vehicle_id, v.model, v.fuel_type
ORDER BY v.fuel_type, rank_in_fuel_type;

-- name: anomaly_rate_by_vehicle
-- desc: Anomaly events and share of anomalous seconds per vehicle
SELECT v.vehicle_id, v.model,
       SUM(k.n_anomaly_events)                         AS events,
       SUM(k.anomaly_points)                           AS anomalous_seconds,
       SUM(k.n_rows)                                   AS total_seconds,
       ROUND(100.0 * SUM(k.anomaly_points) / SUM(k.n_rows), 2) AS anomaly_pct
FROM test_kpis k
JOIN tests t    ON t.test_id = k.test_id
JOIN vehicles v ON v.vehicle_id = t.vehicle_id
GROUP BY v.vehicle_id, v.model
ORDER BY anomaly_pct DESC;

-- name: worst_tests
-- desc: Ten tests with the most severe anomaly activity
SELECT e.test_id, t.vehicle_id, t.test_type,
       COUNT(*)                                        AS events,
       SUM(CASE WHEN e.severity = 'high' THEN 1 ELSE 0 END) AS high_severity,
       ROUND(MAX(e.max_score), 3)                      AS peak_score
FROM anomaly_events e
JOIN tests t ON t.test_id = e.test_id
GROUP BY e.test_id, t.vehicle_id, t.test_type
ORDER BY high_severity DESC, events DESC
LIMIT 10;

-- name: events_by_sensor_severity
-- desc: Event counts by suspected sensor and severity
SELECT top_sensor,
       SUM(severity = 'high')   AS high,
       SUM(severity = 'medium') AS medium,
       SUM(severity = 'low')    AS low,
       COUNT(*)                 AS total
FROM anomaly_events
GROUP BY top_sensor
ORDER BY total DESC;

-- name: kpi_vs_test_type_average
-- desc: Each test compared with the average of its test type (window AVG)
SELECT t.test_id, t.vehicle_id, t.test_type,
       ROUND(k.avg_fuel_l100km, 2)                                              AS fuel_l100km,
       ROUND(AVG(k.avg_fuel_l100km) OVER (PARTITION BY t.test_type), 2)         AS type_avg,
       ROUND(100.0 * (k.avg_fuel_l100km / AVG(k.avg_fuel_l100km) OVER (PARTITION BY t.test_type) - 1), 1) AS delta_pct
FROM test_kpis k
JOIN tests t ON t.test_id = k.test_id
ORDER BY delta_pct DESC
LIMIT 15;

-- name: rolling_max_temperature
-- desc: Rolling 1-minute max temperature around the peak of the hottest test (window frame, every 10 s)
WITH hottest AS (
    SELECT test_id FROM test_kpis ORDER BY max_engine_temp_c DESC LIMIT 1
),
peak AS (
    SELECT m.t_s AS peak_t
    FROM measurements m
    WHERE m.test_id = (SELECT test_id FROM hottest)
    ORDER BY m.engine_temp_c DESC
    LIMIT 1
),
w AS (
    SELECT m.test_id, m.t_s, m.engine_temp_c,
           MAX(m.engine_temp_c) OVER (ORDER BY m.t_s ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS rolling_max_1min
    FROM measurements m
    WHERE m.test_id = (SELECT test_id FROM hottest) AND m.is_valid = 1
)
SELECT w.test_id, CAST(w.t_s AS INTEGER) AS t_s, ROUND(w.engine_temp_c, 1) AS engine_temp_c,
       ROUND(w.rolling_max_1min, 1) AS rolling_max_1min
FROM w, peak
WHERE w.t_s BETWEEN peak.peak_t - 120 AND peak.peak_t + 60
  AND CAST(w.t_s AS INTEGER) % 10 = 0
ORDER BY w.t_s;

-- name: urban_fuel_consumption
-- desc: City-driving (urban test_type) fuel consumption per vehicle, vs. their overall average
SELECT v.vehicle_id, v.model, v.fuel_type,
       ROUND(AVG(CASE WHEN t.test_type = 'urban' THEN k.avg_fuel_l100km END), 2) AS urban_l100km,
       ROUND(AVG(k.avg_fuel_l100km), 2)                                          AS overall_l100km,
       COUNT(CASE WHEN t.test_type = 'urban' THEN 1 END)                         AS n_urban_tests
FROM test_kpis k
JOIN tests t    ON t.test_id = k.test_id
JOIN vehicles v ON v.vehicle_id = t.vehicle_id
GROUP BY v.vehicle_id, v.model, v.fuel_type
HAVING n_urban_tests > 0
ORDER BY urban_l100km DESC;

-- name: braking_events_by_vehicle
-- desc: Hard-braking events and severity breakdown per vehicle
SELECT v.vehicle_id, v.model,
       COUNT(*)                             AS n_events,
       SUM(b.severity = 'severe')           AS severe,
       SUM(b.severity = 'moderate')         AS moderate,
       SUM(b.severity = 'light')            AS light,
       ROUND(AVG(b.speed_drop_kmh), 1)      AS avg_speed_drop_kmh
FROM braking_events b
JOIN tests t    ON t.test_id = b.test_id
JOIN vehicles v ON v.vehicle_id = t.vehicle_id
GROUP BY v.vehicle_id, v.model
ORDER BY n_events DESC;

-- name: data_quality_summary
-- desc: Data-quality issues repaired by the cleaning step, per vehicle
SELECT t.vehicle_id,
       SUM(q.n_duplicates_removed)       AS duplicates_removed,
       SUM(q.n_cells_out_of_range)       AS out_of_range_cells,
       SUM(q.n_cells_missing_raw)        AS missing_cells_raw,
       SUM(q.n_cells_missing_after_fill) AS missing_cells_after_fill,
       ROUND(AVG(q.valid_pct), 2)        AS avg_valid_pct
FROM data_quality q
JOIN tests t ON t.test_id = q.test_id
GROUP BY t.vehicle_id
ORDER BY out_of_range_cells DESC, duplicates_removed DESC;
