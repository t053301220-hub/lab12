# streamlit_app.py
"""
Simulación completa del Sistema de Gestión de Fatiga
- Genera datos de dispositivos (SMARTWATCH, BANDA_ANTIFATIGA, TELEMATICA)
- Fusión, predicción de fatiga y detección de anomalías localmente
- Dashboard interactivo con alertas y reportes
Listo para subir a GitHub y desplegar en streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import uuid
import random
import time

st.set_page_config(page_title="Simulación - Gestión de Fatiga", layout="wide")

# -----------------------------
# Utilidades: simulador de dispositivos
# -----------------------------
DEVICE_TYPES = ["SMARTWATCH", "BANDA_ANTIFATIGA", "TELEMATICA"]

def gen_operator(i):
    return {
        "id": str(uuid.uuid4()),
        "codigo_operador": f"OP{i:03}",
        "nombre_completo": f"Operador {i}",
        "turno_asignado": random.choice(["DIA","NOCHE","ROTATIVO"]),
    }

def simulate_smartwatch():
    # simula sleep + vitals
    sleep_hours = max(3, min(9, np.random.normal(7,1.5)))
    return {
        "device_type":"SMARTWATCH",
        "timestamp": datetime.utcnow().isoformat(),
        "sleep": {
            "duration_hours": round(sleep_hours,2),
            "quality_score": int(np.clip(np.random.normal(75,10),30,100)),
            "deep_minutes": int(max(0, np.random.normal(80,30))),
            "rem_minutes": int(max(0, np.random.normal(90,30))),
            "efficiency": round(np.clip(np.random.normal(0.85,0.08),0.4,1.0),2)
        },
        "vitals": {
            "heart_rate": int(np.clip(np.random.normal(72,8),40,140)),
            "hrv_rmssd": float(np.clip(np.random.normal(50,20),5,120)),
            "hrv_sdnn": float(np.clip(np.random.normal(55,20),5,120)),
            "spo2": round(np.clip(np.random.normal(97,1.5),85,100),1),
            "skin_temp": round(np.clip(np.random.normal(36.2,0.5),34,39),2),
            "stress_level": int(np.clip(np.random.normal(30,20),0,100))
        },
        "activity": {
            "steps": int(max(0, np.random.normal(4000,2000))),
            "calories": int(max(1200, np.random.normal(2200,300))),
            "active_minutes": int(max(0, np.random.normal(90,40)))
        }
    }

def simulate_fatigue_band():
    return {
        "device_type":"BANDA_ANTIFATIGA",
        "timestamp": datetime.utcnow().isoformat(),
        "posture": {
            "trunk_angle": float(np.clip(np.random.normal(5,8),-45,45)),
            "head_nods": int(np.clip(np.random.poisson(0.3),0,50)),
            "micro_sleeps": int(np.clip(np.random.poisson(0.1),0,20))
        },
        "emg": {
            "neck_activity": int(np.clip(np.random.normal(40,20),0,200)),
            "back_activity": int(np.clip(np.random.normal(50,20),0,200))
        },
        "movement": {
            "inactivity_minutes": int(np.clip(np.random.normal(10,8),0,180)),
            "erratic_count": int(np.clip(np.random.poisson(0.2),0,50))
        }
    }

def simulate_telematic():
    return {
        "device_type":"TELEMATICA",
        "timestamp": datetime.utcnow().isoformat(),
        "machinery": {"type": random.choice(["EXCAVADORA","BULDOZER","CAMION"]), "engine_hours": round(np.random.uniform(0,5000),1)},
        "shift": {"type": random.choice(["DIA","NOCHE"]), "hours_elapsed": round(np.clip(np.random.normal(4,3),0,16),2)},
        "environment": {"temperature": round(np.clip(np.random.normal(22,6),-10,45),1), "humidity": round(np.clip(np.random.normal(50,20),5,100),1)},
        "noise_db": int(np.clip(np.random.normal(75,10),30,120))
    }

# -----------------------------
# Pipeline: limpieza/estandarización, fusión, ML (simulado), detección anomalias
# -----------------------------
def standardize(raw):
    # transforma cada payload simulado a formato unificado (similar al nodo 'Limpieza y Estandarización')
    base = {
        "timestamp": raw["timestamp"],
        "device_type": raw["device_type"]
    }
    if raw["device_type"]=="SMARTWATCH":
        base["hrv_rmssd"] = raw["vitals"]["hrv_rmssd"]
        base["heart_rate"] = raw["vitals"]["heart_rate"]
        base["spo2"] = raw["vitals"]["spo2"]
        base["sleep_duration_hours"] = raw["sleep"]["duration_hours"]
        base["sleep_quality"] = raw["sleep"]["quality_score"]
        base["stress_level"] = raw["vitals"]["stress_level"]
    elif raw["device_type"]=="BANDA_ANTIFATIGA":
        base["trunk_angle"] = raw["posture"]["trunk_angle"]
        base["head_nods"] = raw["posture"]["head_nods"]
        base["micro_sleeps"] = raw["posture"]["micro_sleeps"]
        base["inactivity_minutes"] = raw["movement"]["inactivity_minutes"]
    else:
        base["machinery_type"] = raw["machinery"]["type"]
        base["hours_in_shift"] = raw["shift"]["hours_elapsed"]
        base["ambient_temperature"] = raw["environment"]["temperature"]
        base["ambient_humidity"] = raw["environment"]["humidity"]
    return base

def fuse_data(items):
    # recibe lista de standardized items y los combina por operador (aquí usamos simulación por operador index)
    fused = {}
    for it in items:
        key = it.get("operator_id","sim-op")  # en simulación, podemos usar 'sim-op'
        if key not in fused:
            fused[key] = {"merged_data":{}, "sources":set(), "timestamp":it["timestamp"]}
        for k,v in it.items():
            if k not in ["timestamp","device_type","operator_id"]:
                fused[key]["merged_data"][k]=v
        fused[key]["sources"].add(it["device_type"])
    # devolver lista
    return [ {"operator_id":k, "timestamp":v["timestamp"], "merged_data":v["merged_data"], "data_completeness": int(len(v["sources"])/3*100)} for k,v in fused.items() ]

# Modelo de fatiga (misma lógica simplificada del documento)
def calculate_fatigue_index(merged):
    total = 0.0
    confs = []
    # HRV: inverso
    hrv = merged.get("hrv_rmssd")
    if hrv is not None:
        hrv_norm = (hrv - 10) / (100 - 10) if (100-10)!=0 else 0.5
        hrv_score = (1 - hrv_norm)*100
        total += hrv_score * 0.15
        confs.append(0.9)
    else:
        confs.append(0.3)
    # SpO2
    spo2 = merged.get("spo2")
    if spo2 is not None:
        spo2_norm = (spo2 - 85) / (100 - 85) if (100-85)!=0 else 0.5
        total += (1 - spo2_norm)*100 * 0.10
        confs.append(0.85)
    else:
        confs.append(0.3)
    # HR
    hr = merged.get("heart_rate")
    if hr is not None:
        dev = abs(hr - 70)/70
        hrScore = min(dev*200, 100)
        total += hrScore * 0.08
        confs.append(0.8)
    else:
        confs.append(0.3)
    # Sleep duration
    sd = merged.get("sleep_duration_hours")
    if sd is not None:
        sleepScore = max(0, (7 - sd)/6 * 100)  # simplificado
        total += sleepScore * 0.12
        confs.append(0.85)
    else:
        total += 50 * 0.12
        confs.append(0.4)
    # micro-sleeps (crítico)
    ms = merged.get("micro_sleeps")
    if ms is not None:
        total += min(ms*25,100) * 0.05
        confs.append(0.95)
    else:
        confs.append(0.2)
    # posture
    ta = merged.get("trunk_angle")
    if ta is not None:
        total += min(abs(ta)*2,100) * 0.08
        confs.append(0.7)
    else:
        confs.append(0.2)
    # shift hours
    hrs = merged.get("hours_in_shift")
    if hrs is not None:
        total += (hrs/12*100) * 0.05
        confs.append(0.9)
    else:
        confs.append(0.3)
    final = min(max(total,0),100)
    avg_conf = round(sum(confs)/len(confs),2)
    return {"fatigue_index": round(final,2), "confidence": avg_conf}

def classify_risk(fi):
    if fi < 40: return "BAJO"
    if fi < 70: return "MEDIO"
    if fi < 85: return "ALTO"
    return "CRITICO"

def detect_anomalies(merged, fatigue_index):
    anomalies = []
    # thresholds similar al doc
    if merged.get("hrv_rmssd") is not None and merged.get("hrv_rmssd") < 20:
        anomalies.append({"type":"HRV_CRITICAL_LOW","severity":"HIGH","value":merged.get("hrv_rmssd")})
    if merged.get("spo2") is not None and merged.get("spo2") < 92:
        anomalies.append({"type":"SPO2_LOW","severity":"CRITICAL","value":merged.get("spo2")})
    if merged.get("micro_sleeps") is not None and merged.get("micro_sleeps") >= 3:
        anomalies.append({"type":"MICRO_SLEEPS","severity":"CRITICAL","value":merged.get("micro_sleeps")})
    if merged.get("head_nods") is not None and merged.get("head_nods") > 5:
        anomalies.append({"type":"HEAD_NODS","severity":"HIGH","value":merged.get("head_nods")})
    if merged.get("hours_in_shift") is not None and merged.get("hours_in_shift") > 10 and fatigue_index>75:
        anomalies.append({"type":"EXTENDED_SHIFT_HIGH_FATIGUE","severity":"HIGH","value":{"hours":merged.get("hours_in_shift"),"fatigue":fatigue_index}})
    return anomalies

# -----------------------------
# Construcción de UI Streamlit
# -----------------------------
st.title("Simulación Sistema de Gestión de Fatiga — Demo Local")

# Sidebar: parámetros de simulación
st.sidebar.header("Parámetros de Simulación")
n_ops = st.sidebar.slider("Número de Operadores", 1, 10, 4)
gen_batch = st.sidebar.number_input("Generar registros por operador", 1, 10, 3)
freq_sec = st.sidebar.slider("Intervalo simulación (s)", 1, 10, 2)
start_sim = st.sidebar.button("▶️ Ejecutar Simulación (genera datos)")

# crear operadores simulados
operators = [gen_operator(i+1) for i in range(n_ops)]
df_ops = pd.DataFrame(operators)

# panel principal
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Operadores simulados")
    st.dataframe(df_ops[["codigo_operador","nombre_completo","turno_asignado"]], use_container_width=True)
with col2:
    st.subheader("Controles")
    if st.button("Generar instantáneamente un batch"):
        start_sim = True

# area de logs y tabla de métricas procesadas en memoria
if 'metrics' not in st.session_state:
    st.session_state['metrics'] = []

if start_sim:
    # generar datos
    records = []
    for op in operators:
        for _ in range(gen_batch):
            # generar un payload por cada tipo (para aumentar probabilidad de tener todas las fuentes)
            raw_sw = simulate_smartwatch()
            raw_sw["operator_id"] = op["id"]
            std_sw = standardize(raw_sw)
            std_sw["operator_id"] = op["id"]
            raw_band = simulate_fatigue_band()
            raw_band["operator_id"] = op["id"]
            std_band = standardize(raw_band)
            std_band["operator_id"] = op["id"]
            raw_tel = simulate_telematic()
            raw_tel["operator_id"] = op["id"]
            std_tel = standardize(raw_tel)
            std_tel["operator_id"] = op["id"]
            # agregarlos
            records.extend([std_sw, std_band, std_tel])
    # fusionar por operador
    fused = fuse_data(records)
    # correr modelo y detección
    new_metrics = []
    for f in fused:
        merged = f["merged_data"]
        pred = calculate_fatigue_index(merged)
        risk = classify_risk(pred["fatigue_index"])
        anomalies = detect_anomalies(merged, pred["fatigue_index"])
        metric = {
            "id": str(uuid.uuid4()),
            "timestamp": f["timestamp"],
            "operator_id": f["operator_id"],
            "indice_fatiga": pred["fatigue_index"],
            "clasificacion_riesgo": risk,
            "confianza": pred["confidence"],
            "merged": merged,
            "anomalies": anomalies
        }
        new_metrics.append(metric)
    # guardar en session
    st.session_state['metrics'] = new_metrics + st.session_state['metrics']
    st.success(f"{len(new_metrics)} métricas procesadas y agregadas")

    # small pause to simulate time
    time.sleep(0.2)

# mostrar tabla de métricas
st.markdown("---")
st.subheader("Métricas Procesadas (últimas 100)")
df_metrics = pd.DataFrame(st.session_state['metrics']) if st.session_state['metrics'] else pd.DataFrame()
if not df_metrics.empty:
    display_cols = ["timestamp","operator_id","indice_fatiga","clasificacion_riesgo","confianza"]
    st.dataframe(df_metrics[display_cols].head(100), use_container_width=True)
else:
    st.info("No hay métricas generadas. Presiona 'Ejecutar Simulación' o 'Generar instantáneamente'.")

# alertas simples: filtrar métricas con riesgo ALTO/CRITICO o anomalías
st.markdown("---")
st.subheader("Alertas detectadas")
alerts = []
for m in st.session_state['metrics']:
    if m["clasificacion_riesgo"] in ["ALTO","CRITICO"] or len(m["anomalies"])>0:
        alerts.append(m)
if alerts:
    for a in alerts[:20]:
        st.markdown(f"**Operador:** {a['operator_id']}  —  **Fatiga:** {a['indice_fatiga']}  —  **Riesgo:** {a['clasificacion_riesgo']}")
        if a['anomalies']:
            for an in a['anomalies']:
                st.markdown(f"- ⚠️ {an['type']} (severity: {an['severity']}) — value: {an.get('value')}")
        st.markdown("---")
else:
    st.info("No hay alertas activas.")

# Visualizaciones — gauge y serie temporal por operador
st.markdown("---")
st.subheader("Visualizaciones")
if not df_metrics.empty:
    # seleccionar operador para graficar
    op_list = df_ops["id"].tolist()
    selected_op = st.selectbox("Selecciona un operador para ver su evolución", op_list)
    df_op = df_metrics[df_metrics["operator_id"]==selected_op]
    if not df_op.empty:
        # gauge último
        last = df_op.iloc[0]
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=last["indice_fatiga"],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Índice de Fatiga"},
            gauge={'axis': {'range': [0,100]},
                   'bar': {'color': "darkred" if last["indice_fatiga"]>85 else "orange" if last["indice_fatiga"]>70 else "green"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

        # serie temporal
        df_op_sorted = df_op.sort_values("timestamp")
        fig2 = px.line(df_op_sorted, x="timestamp", y="indice_fatiga", title="Evolución índice de fatiga")
        st.plotly_chart(fig2, use_container_width=True)

# sección de reporte rápido (PDF simulado)
st.markdown("---")
st.subheader("Generar reporte rápido (simulado)")
if st.button("Generar reporte PDF (simulado)"):
    st.info("Generando reporte... (simulación, no se guarda en Supabase).")
    # armar resumen
    if not df_metrics.empty:
        avg_fatigue = df_metrics["indice_fatiga"].mean()
        total_alerts = len([m for m in st.session_state['metrics'] if m['clasificacion_riesgo'] in ["ALTO","CRITICO"] or m['anomalies']])
        st.success(f"Reporte generado: promedio fatiga={avg_fatigue:.2f}, alertas={total_alerts}")
    else:
        st.warning("No hay datos para generar reporte.")

st.markdown("----")
st.caption("Simulación basada en el documento HERRAMIENTA_CLAUDE — lista para subir a GitHub / Streamlit Cloud.")
