"""Vehicle Test Data Analyzer — interactive dashboard (Streamlit + Plotly).

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from vta import anomaly as an
from vta.config import DB_PATH, DETECTOR_FEATURES, HARD_BRAKE_THRESHOLD_MS2, SENSORS
from vta.storage import is_read_only_query, load_named_queries, read_sql

ANOMALY_COLOR = "#d62728"
SENSOR_LABELS = {
    "speed_kmh": "Vitesse (km/h)",
    "rpm": "Régime (tr/min)",
    "throttle_pct": "Accélérateur (%)",
    "engine_temp_c": "Température moteur (°C)",
    "oil_pressure_bar": "Pression d'huile (bar)",
    "fuel_rate_lph": "Débit carburant (L/h)",
    "battery_v": "Batterie (V)",
    "vibration_g": "Vibrations (g)",
}
KPI_LABELS = {
    "distance_km": "Distance (km)",
    "avg_speed_kmh": "Vitesse moy. (km/h)",
    "max_speed_kmh": "Vitesse max (km/h)",
    "avg_rpm": "Régime moy. (tr/min)",
    "max_engine_temp_c": "Temp. moteur max (°C)",
    "avg_fuel_l100km": "Conso moy. (L/100 km)",
    "idle_pct": "Temps au ralenti (%)",
    "hard_brake_count": "Freinages brusques",
    "avg_vibration_g": "Vibrations moy. (g)",
    "min_oil_pressure_bar": "Pression huile min (bar)",
    "min_battery_v": "Batterie min (V)",
    "n_anomaly_events": "Événements anormaux",
    "anomaly_pct": "Secondes anormales (%)",
    "n_braking_events": "Événements de freinage",
}
BRAKE_SEVERITY_COLORS = {"light": "#ffbf00", "moderate": "#ff7f0e", "severe": ANOMALY_COLOR}
BRAKE_SEVERITY_ORDER = ["light", "moderate", "severe"]

st.set_page_config(page_title="Vehicle Test Data Analyzer", page_icon="🚗", layout="wide")


# ----------------------------------------------------------------------------- data access
def db_stamp() -> float:
    return DB_PATH.stat().st_mtime if DB_PATH.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_tables(stamp: float) -> dict[str, pd.DataFrame]:
    return {
        "vehicles": read_sql("SELECT * FROM vehicles"),
        "tests": read_sql("SELECT * FROM tests"),
        "kpis": read_sql("SELECT * FROM test_kpis"),
        "events": read_sql("SELECT * FROM anomaly_events"),
        "quality": read_sql("SELECT * FROM data_quality"),
        "labels": read_sql("SELECT * FROM injected_anomalies"),
        "braking": read_sql("SELECT * FROM braking_events"),
    }


@st.cache_data(show_spinner=False)
def load_test(test_id: str, stamp: float) -> pd.DataFrame:
    return read_sql("SELECT * FROM measurements WHERE test_id = ? ORDER BY t_s", params=(test_id,))


@st.cache_data(show_spinner="Chargement des mesures…")
def load_features(stamp: float) -> pd.DataFrame:
    cols = ["test_id", "timestamp", "t_s", "is_valid", "speed_kmh", "rpm"] + DETECTOR_FEATURES
    cols = list(dict.fromkeys(cols))
    df = read_sql(f"SELECT {', '.join(cols)} FROM measurements")
    df["is_valid"] = df["is_valid"].astype(bool)
    return df


@st.cache_data(show_spinner="Isolation Forest en cours…")
def run_detection(contamination: float, n_estimators: int, min_points: int, stamp: float):
    tests = load_tables(stamp)["tests"][["test_id", "vehicle_id", "test_type"]]
    df = load_features(stamp).merge(tests, on="test_id", how="left")
    cfg = an.AnomalyConfig(contamination=contamination, n_estimators=n_estimators, min_event_points=min_points)
    scored = an.VehicleAnomalyDetector(cfg).fit_score(df)
    events = an.extract_events(scored, cfg.min_event_points, cfg.max_gap_s)
    metrics, by_type = an.evaluate_against_labels(scored, events, load_tables(stamp)["labels"])
    return scored, events, metrics, by_type


# ----------------------------------------------------------------------------- empty state
if not DB_PATH.exists():
    st.title("🚗 Vehicle Test Data Analyzer")
    st.warning("Aucune base de données trouvée. Lance le pipeline pour générer les données et détecter les anomalies.")
    if st.button("Générer les données et exécuter le pipeline (Pandas)", type="primary"):
        from vta.run_pipeline import run

        with st.spinner("Pipeline en cours (≈ 10 s)…"):
            run(engine="pandas", regenerate=True, quiet=True)
        st.rerun()
    st.code("python -m vta.run_pipeline --engine pandas --regenerate", language="bash")
    st.stop()

stamp = db_stamp()
T = load_tables(stamp)
tests = T["tests"].merge(T["vehicles"], on="vehicle_id").merge(T["kpis"], on="test_id")

# ----------------------------------------------------------------------------- sidebar
st.sidebar.header("Filtres")
sel_veh = st.sidebar.multiselect("Véhicules", sorted(tests["vehicle_id"].unique()), default=sorted(tests["vehicle_id"].unique()))
sel_type = st.sidebar.multiselect("Type d'essai", sorted(tests["test_type"].unique()), default=sorted(tests["test_type"].unique()))
ftests = tests[tests["vehicle_id"].isin(sel_veh) & tests["test_type"].isin(sel_type)].copy()
st.sidebar.caption(f"{len(ftests)} essais sélectionnés sur {len(tests)}")
st.sidebar.divider()
st.sidebar.caption("Base : " + str(DB_PATH.name))

st.title("🚗 Vehicle Test Data Analyzer")
st.caption("Analyse de données d'essais automobiles et détection d'anomalies (Isolation Forest)")

if ftests.empty:
    st.info("Aucun essai ne correspond aux filtres.")
    st.stop()

tab_overview, tab_test, tab_compare, tab_brake, tab_anom, tab_sql = st.tabs(
    ["Vue d'ensemble", "Analyse d'un essai", "Comparer des essais", "Freinage", "Détection d'anomalies", "SQL"]
)

# ----------------------------------------------------------------------------- overview
with tab_overview:
    fev = T["events"][T["events"]["test_id"].isin(ftests["test_id"])]
    c = st.columns(5)
    c[0].metric("Véhicules", ftests["vehicle_id"].nunique())
    c[1].metric("Essais", len(ftests))
    c[2].metric("Données", f"{ftests['n_rows'].sum() / 3600:.1f} h")
    c[3].metric("Événements anormaux", len(fev), f"{(fev['severity'] == 'high').sum()} sévères", delta_color="inverse")
    c[4].metric("Secondes anormales", f"{ftests['anomaly_points'].sum() / ftests['n_rows'].sum() * 100:.2f} %")

    left, right = st.columns(2)
    by_veh = (
        ftests.groupby(["vehicle_id", "model", "fuel_type"], as_index=False)
        .agg(
            conso=("avg_fuel_l100km", "mean"),
            vibrations=("avg_vibration_g", "mean"),
            points=("anomaly_points", "sum"),
            rows=("n_rows", "sum"),
            events=("n_anomaly_events", "sum"),
        )
        .assign(anomaly_pct=lambda d: d["points"] / d["rows"] * 100)
    )
    fig = px.bar(
        by_veh.sort_values("conso"), x="vehicle_id", y="conso", color="fuel_type",
        labels={"conso": "Conso moyenne (L/100 km)", "vehicle_id": "", "fuel_type": "Carburant"},
        title="Consommation moyenne par véhicule", hover_data=["model"],
    )
    left.plotly_chart(fig, width="stretch")
    fig = px.bar(
        by_veh.sort_values("anomaly_pct", ascending=False), x="vehicle_id", y="anomaly_pct",
        color_discrete_sequence=[ANOMALY_COLOR], hover_data=["model", "events"],
        labels={"anomaly_pct": "Secondes anormales (%)", "vehicle_id": ""},
        title="Taux d'anomalies par véhicule",
    )
    right.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    fig = px.box(
        ftests, x="vehicle_id", y="avg_vibration_g", color="fuel_type", points="all",
        labels={"avg_vibration_g": "Vibrations moyennes (g)", "vehicle_id": "", "fuel_type": "Carburant"},
        title="Vibrations moyennes par véhicule (dérive d'usure)",
    )
    left.plotly_chart(fig, width="stretch")
    fig = px.scatter(
        ftests, x="avg_speed_kmh", y="avg_fuel_l100km", color="test_type", symbol="fuel_type",
        hover_data=["test_id", "vehicle_id"], size="distance_km",
        labels={"avg_speed_kmh": "Vitesse moyenne (km/h)", "avg_fuel_l100km": "Conso (L/100 km)", "test_type": "Type d'essai"},
        title="Consommation vs vitesse moyenne",
    )
    right.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    urban = ftests[ftests["test_type"] == "urban"]
    if urban.empty:
        left.info("Aucun essai de type « urbain » dans la sélection.")
    else:
        by_urban = (
            urban.groupby(["vehicle_id", "model", "fuel_type"], as_index=False)["avg_fuel_l100km"].mean()
        )
        fig = px.bar(
            by_urban.sort_values("avg_fuel_l100km"), x="vehicle_id", y="avg_fuel_l100km", color="fuel_type",
            hover_data=["model"],
            labels={"avg_fuel_l100km": "Conso en ville (L/100 km)", "vehicle_id": "", "fuel_type": "Carburant"},
            title="Circulation en ville (L/100 km)",
        )
        left.plotly_chart(fig, width="stretch")

    fbrk = T["braking"][T["braking"]["test_id"].isin(ftests["test_id"])]
    if fbrk.empty:
        right.info("Aucun freinage brusque détecté dans la sélection.")
    else:
        by_brake = (
            fbrk.merge(ftests[["test_id", "vehicle_id"]], on="test_id")
            .groupby("vehicle_id", as_index=False)
            .size()
            .rename(columns={"size": "events"})
        )
        fig = px.bar(
            by_brake.sort_values("events", ascending=False), x="vehicle_id", y="events",
            color_discrete_sequence=[ANOMALY_COLOR],
            labels={"events": "Événements de freinage", "vehicle_id": ""},
            title="Freinages brusques par véhicule",
        )
        right.plotly_chart(fig, width="stretch")

    with st.expander("Qualité des données (nettoyage appliqué par le pipeline)"):
        q = T["quality"][T["quality"]["test_id"].isin(ftests["test_id"])]
        k = st.columns(4)
        k[0].metric("Doublons supprimés", int(q["n_duplicates_removed"].sum()))
        k[1].metric("Valeurs hors plage (capteur en erreur)", int(q["n_cells_out_of_range"].sum()))
        k[2].metric("Valeurs manquantes (brut)", int(q["n_cells_missing_raw"].sum()))
        k[3].metric("Manquantes après imputation", int(q["n_cells_missing_after_fill"].sum()))
        st.dataframe(q.sort_values("valid_pct").head(10), width="stretch", hide_index=True)

# ----------------------------------------------------------------------------- one test
with tab_test:
    order = ftests.sort_values(["n_anomaly_events", "test_id"], ascending=[False, True])
    labels = {
        r.test_id: f"{r.test_id} · {r.vehicle_id} · {r.test_type} · {r.n_anomaly_events} évt"
        for r in order.itertuples()
    }
    test_id = st.selectbox("Essai", list(labels), format_func=labels.get)
    shown = st.multiselect(
        "Capteurs affichés", SENSORS, default=["speed_kmh", "rpm", "engine_temp_c", "oil_pressure_bar", "fuel_rate_lph", "vibration_g"],
        format_func=SENSOR_LABELS.get,
    )
    df = load_test(test_id, stamp)
    ev = T["events"][T["events"]["test_id"] == test_id]
    tb = T["braking"][T["braking"]["test_id"] == test_id]
    row = tests[tests["test_id"] == test_id].iloc[0]

    c = st.columns(6)
    c[0].metric("Véhicule", f"{row.vehicle_id}")
    c[1].metric("Distance", f"{row.distance_km:.1f} km")
    c[2].metric("Conso", f"{row.avg_fuel_l100km:.2f} L/100")
    c[3].metric("Temp. max", f"{row.max_engine_temp_c:.0f} °C")
    c[4].metric("Freinages brusques", int(row.hard_brake_count), f"{len(tb)} évt")
    c[5].metric("Événements", len(ev))

    if shown:
        fig = make_subplots(rows=len(shown), cols=1, shared_xaxes=True, vertical_spacing=0.025,
                            subplot_titles=[SENSOR_LABELS[s] for s in shown])
        flagged = df[df["is_anomaly"] == 1]
        for i, s in enumerate(shown, 1):
            fig.add_trace(go.Scattergl(x=df["t_s"], y=df[s], mode="lines", line=dict(width=1.1), name=SENSOR_LABELS[s], showlegend=False), row=i, col=1)
            fig.add_trace(
                go.Scattergl(x=flagged["t_s"], y=flagged[s], mode="markers", marker=dict(color=ANOMALY_COLOR, size=4),
                             name="Point anormal", showlegend=(i == 1), hovertext=flagged["top_sensor"]),
                row=i, col=1,
            )
        for e in ev.itertuples():
            fig.add_vrect(x0=e.start_t_s, x1=e.end_t_s, fillcolor=ANOMALY_COLOR, opacity=0.12, line_width=0, row="all", col=1)
        for b in tb.itertuples():
            fig.add_vrect(x0=b.start_t_s, x1=b.end_t_s, fillcolor="#1f77b4", opacity=0.12, line_width=0, row="all", col=1)
        fig.update_xaxes(title_text="Temps depuis le début de l'essai (s)", row=len(shown), col=1)
        fig.update_layout(height=190 * len(shown) + 90, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.04))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Événements détectés")
    if ev.empty:
        st.success("Aucun événement anormal sur cet essai.")
    else:
        st.dataframe(
            ev.drop(columns=["event_id", "test_id"]).round(3), width="stretch", hide_index=True,
        )

    st.subheader("Freinages brusques détectés")
    if tb.empty:
        st.success("Aucun freinage brusque sur cet essai.")
    else:
        st.dataframe(
            tb.drop(columns=["event_id", "test_id"]).round(2), width="stretch", hide_index=True,
        )

# ----------------------------------------------------------------------------- compare
with tab_compare:
    ids = list(ftests["test_id"])
    default = ids[: min(3, len(ids))]
    picked = st.multiselect("Essais à comparer (2 à 4)", ids, default=default, max_selections=4)
    if len(picked) < 2:
        st.info("Sélectionne au moins deux essais.")
    else:
        names = {i: f"{i} ({tests.loc[tests.test_id == i, 'vehicle_id'].iat[0]}, {tests.loc[tests.test_id == i, 'test_type'].iat[0]})" for i in picked}
        sensor = st.selectbox("Capteur à superposer", SENSORS, index=SENSORS.index("engine_temp_c"), format_func=SENSOR_LABELS.get)
        smooth = st.slider("Lissage (moyenne glissante, secondes)", 1, 60, 10)
        fig = go.Figure()
        for i in picked:
            d = load_test(i, stamp)
            y = d[sensor].rolling(smooth, min_periods=1).mean()
            fig.add_trace(go.Scattergl(x=d["t_s"], y=y, mode="lines", name=names[i]))
        fig.update_layout(height=420, xaxis_title="Temps (s)", yaxis_title=SENSOR_LABELS[sensor], margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, width="stretch")

        cols = ["distance_km", "avg_speed_kmh", "avg_fuel_l100km", "max_engine_temp_c", "avg_vibration_g",
                "min_oil_pressure_bar", "min_battery_v", "hard_brake_count", "n_braking_events",
                "n_anomaly_events", "anomaly_pct"]
        sub = tests[tests["test_id"].isin(picked)].set_index("test_id").loc[picked, cols]
        table = sub.T.rename(index=KPI_LABELS)
        table.columns = [names[c] for c in table.columns]
        left, right = st.columns([3, 2])
        left.subheader("KPIs côte à côte")
        left.dataframe(table.round(2), width="stretch")

        radar_cols = ["avg_speed_kmh", "avg_rpm", "avg_fuel_l100km", "max_engine_temp_c", "avg_vibration_g", "hard_brake_count"]
        allk = tests[radar_cols]
        norm = (tests.set_index("test_id")[radar_cols] - allk.min().values) / (allk.max() - allk.min()).replace(0, 1).values
        rf = go.Figure()
        for i in picked:
            vals = norm.loc[i].tolist()
            rf.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=[KPI_LABELS.get(c, c) for c in radar_cols] + [KPI_LABELS.get(radar_cols[0], radar_cols[0])],
                                         fill="toself", opacity=0.45, name=names[i]))
        rf.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)), height=380,
                         margin=dict(l=30, r=30, t=30, b=10), legend=dict(orientation="h", y=-0.15))
        right.subheader("Profil normalisé (0 = min flotte, 1 = max)")
        right.plotly_chart(rf, width="stretch")

# ----------------------------------------------------------------------------- braking
with tab_brake:
    st.markdown(
        f"Un freinage brusque est détecté quand la décélération dépasse **{abs(HARD_BRAKE_THRESHOLD_MS2):.0f} m/s²** "
        "pendant au moins une seconde ; les secondes consécutives (écart ≤ 2 s) sont regroupées en un seul événement."
    )
    fbrk = T["braking"][T["braking"]["test_id"].isin(ftests["test_id"])]
    m = st.columns(4)
    m[0].metric("Événements de freinage", len(fbrk))
    m[1].metric("Sévères", int((fbrk["severity"] == "severe").sum()) if len(fbrk) else 0)
    m[2].metric("Modérés", int((fbrk["severity"] == "moderate").sum()) if len(fbrk) else 0)
    m[3].metric("Essais touchés", fbrk["test_id"].nunique() if len(fbrk) else 0)

    if fbrk.empty:
        st.success("Aucun freinage brusque détecté sur la sélection.")
    else:
        left, right = st.columns(2)
        by_veh = (
            fbrk.merge(tests[["test_id", "vehicle_id", "model"]], on="test_id")
            .groupby(["vehicle_id", "model"], as_index=False)
            .size()
            .rename(columns={"size": "events"})
        )
        fig = px.bar(
            by_veh.sort_values("events", ascending=False), x="vehicle_id", y="events", hover_data=["model"],
            labels={"events": "Événements de freinage", "vehicle_id": ""},
            title="Freinages brusques par véhicule",
        )
        left.plotly_chart(fig, width="stretch")

        fig = px.histogram(
            fbrk, x="severity", color="severity", category_orders={"severity": BRAKE_SEVERITY_ORDER},
            color_discrete_map=BRAKE_SEVERITY_COLORS,
            labels={"severity": "Sévérité", "count": "Événements"},
            title="Répartition par sévérité",
        )
        right.plotly_chart(fig, width="stretch")

        fig = px.scatter(
            fbrk, x="speed_before_kmh", y="speed_drop_kmh", color="severity",
            color_discrete_map=BRAKE_SEVERITY_COLORS, category_orders={"severity": BRAKE_SEVERITY_ORDER},
            hover_data=["test_id", "duration_s"],
            labels={"speed_before_kmh": "Vitesse avant freinage (km/h)", "speed_drop_kmh": "Chute de vitesse (km/h)", "severity": "Sévérité"},
            title="Chute de vitesse vs vitesse initiale",
        )
        st.plotly_chart(fig, width="stretch")

        st.subheader("Événements de freinage")
        view = fbrk.merge(tests[["test_id", "vehicle_id", "test_type"]], on="test_id")
        view = view[[
            "event_id", "test_id", "vehicle_id", "test_type", "start_time", "duration_s",
            "min_accel_ms2", "speed_before_kmh", "speed_after_kmh", "speed_drop_kmh", "severity",
        ]]
        st.dataframe(view.sort_values("min_accel_ms2").round(2), width="stretch", hide_index=True)
        st.download_button(
            "Télécharger les événements de freinage (CSV)", view.to_csv(index=False).encode(),
            "braking_events.csv", "text/csv",
        )

# ----------------------------------------------------------------------------- anomalies
with tab_anom:
    st.markdown(
        "Les résidus contextuels (valeur mesurée − valeur attendue selon régime, charge, vitesse…) sont donnés à un "
        "**Isolation Forest** entraîné séparément pour chaque type d'essai. Ajuste les paramètres : la détection est recalculée."
    )
    c = st.columns(3)
    contamination = c[0].slider("Contamination (part attendue de secondes anormales)", 0.005, 0.06, 0.02, 0.005, format="%.3f")
    n_est = c[1].select_slider("Nombre d'arbres", options=[50, 100, 200, 400], value=100)
    min_pts = c[2].slider("Durée minimale d'un événement (s)", 1, 10, 3)
    scored, events, metrics, by_type = run_detection(contamination, n_est, min_pts, stamp)
    scored = scored[scored["test_id"].isin(ftests["test_id"])]
    events = events[events["test_id"].isin(ftests["test_id"])]

    m = st.columns(4)
    m[0].metric("Secondes signalées", int(scored["is_anomaly"].sum()))
    m[1].metric("Événements", len(events))
    m[2].metric("Sévérité élevée", int((events["severity"] == "high").sum()) if len(events) else 0)
    m[3].metric("Essais touchés", events["test_id"].nunique() if len(events) else 0)

    if metrics:
        st.subheader("Évaluation (données synthétiques : anomalies injectées connues)")
        e = st.columns(4)
        e[0].metric("Rappel (événements)", f"{metrics['event_recall'] * 100:.0f} %")
        e[1].metric("Précision (événements)", f"{metrics['event_precision'] * 100:.0f} %")
        e[2].metric("Rappel (secondes)", f"{metrics['point_recall'] * 100:.0f} %")
        e[3].metric("Précision (secondes)", f"{metrics['point_precision'] * 100:.0f} %")
        if len(by_type):
            fig = px.bar(by_type, x="anomaly_type", y="recall", text=by_type["recall"].map("{:.0%}".format),
                         labels={"anomaly_type": "Type de défaut injecté", "recall": "Taux de détection"}, range_y=[0, 1.05])
            fig.update_layout(height=320, margin=dict(t=20))
            st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    valid = scored[scored["is_valid"]]
    fig = px.histogram(valid, x="anomaly_score", color="is_anomaly", nbins=80, barmode="overlay", opacity=0.75,
                       color_discrete_map={False: "#7f7f7f", True: ANOMALY_COLOR},
                       labels={"anomaly_score": "Score d'anomalie (plus haut = plus anormal)", "is_anomaly": "Signalé"},
                       title="Distribution des scores")
    fig.update_yaxes(type="log")
    left.plotly_chart(fig, width="stretch")

    if len(events):
        ct = events.groupby(["top_sensor", "severity"]).size().reset_index(name="n")
        fig = px.bar(ct, x="top_sensor", y="n", color="severity", title="Événements par capteur suspect",
                     color_discrete_map={"low": "#ffbf00", "medium": "#ff7f0e", "high": ANOMALY_COLOR},
                     category_orders={"severity": ["low", "medium", "high"]},
                     labels={"top_sensor": "Capteur suspect", "n": "Événements", "severity": "Sévérité"})
        right.plotly_chart(fig, width="stretch")

    samp = valid.sample(min(len(valid), 15000), random_state=0)
    fig = px.scatter(samp.sort_values("is_anomaly"), x="speed_kmh", y="rpm", color="is_anomaly", opacity=0.6,
                     color_discrete_map={False: "#c7c7c7", True: ANOMALY_COLOR},
                     labels={"speed_kmh": "Vitesse (km/h)", "rpm": "Régime (tr/min)", "is_anomaly": "Signalé"},
                     title="Régime vs vitesse : les points rouges sortent du schéma de rapports")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Événements")
    if events.empty:
        st.success("Aucun événement avec ces paramètres.")
    else:
        view = events.merge(tests[["test_id", "vehicle_id", "test_type"]], on="test_id")
        view = view[["event_id", "test_id", "vehicle_id", "test_type", "start_time", "duration_s", "severity", "top_sensor", "max_score"]]
        st.dataframe(view.sort_values("max_score", ascending=False).round(3), width="stretch", hide_index=True)
        st.download_button("Télécharger les événements (CSV)", view.to_csv(index=False).encode(), "anomaly_events.csv", "text/csv")

# ----------------------------------------------------------------------------- SQL
with tab_sql:
    st.markdown("Explore la base SQLite (lecture seule). Tables : `vehicles`, `tests`, `measurements`, `test_kpis`, `data_quality`, `anomaly_events`, `braking_events`, vue `v_test_summary`.")
    named = load_named_queries()
    choice = st.selectbox("Requête prédéfinie", list(named), format_func=lambda k: f"{k} — {named[k]['desc']}")
    sql = st.text_area("SQL", named[choice]["sql"], height=240, key=f"sql_{choice}")
    if st.button("Exécuter", type="primary"):
        if not is_read_only_query(sql):
            st.error("Seule une requête SELECT / WITH unique est autorisée.")
        else:
            try:
                res = read_sql(sql)
                st.caption(f"{len(res)} lignes")
                st.dataframe(res, width="stretch", hide_index=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erreur SQL : {exc}")
