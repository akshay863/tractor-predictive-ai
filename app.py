import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time
import numpy as np
from collections import deque

st.set_page_config(page_title="High-Dim Telemetry OS", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .premium-header { background: linear-gradient(90deg, #1f2937 0%, #111827 100%); padding: 24px; border-radius: 12px; border-left: 6px solid #eab308; margin-bottom: 24px; }
    .premium-header h1 { margin: 0; color: #ffffff; font-size: 26px; font-weight: 600; font-family: 'Segoe UI', sans-serif;}
    .premium-header p { margin: 4px 0 0 0; color: #9ca3af; font-size: 13px; letter-spacing: 1px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="premium-header">
        <h1>🚜 Industrial Fleet & Telemetry OS</h1>
        <p>SCALED ANOMALY DETECTION • RUL ESTIMATION • DYNAMIC NLP • FLEET COMMAND</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models(): 
    return joblib.load('tractor_health_model.pkl'), joblib.load('anomaly_model.pkl'), joblib.load('scaler.pkl')

try: 
    rf_model, anomaly_model, scaler = load_models()
except FileNotFoundError: 
    st.error("⚠️ AI Cores Offline: Please run predictive_model.py first.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=50)

with st.sidebar:
    st.markdown("### 📡 Telemetry Link")
    streaming = st.toggle("🟢 Auto-Streaming", value=False)
    st.divider()

    if 'sensors' not in st.session_state:
        st.session_state.sensors = { 
            'rpm': 1800, 'load': 45, 'temp': 85, 'egt': 400, 'fuel': 1500, 'intake': 40,
            'trans': 75, 'pressure': 200, 'flow': 80, 'pto': 540, 'draft': 15,
            'slip': 10, 'radar': 12, 'steer': 0, 'battery': 13.8 
        }

    st.markdown("#### Primary Controls")
    rpm = st.slider("Engine RPM", 800, 2500, st.session_state.sensors['rpm'])
    load = st.slider("Engine Load (%)", 0, 100, st.session_state.sensors['load'])
    temp = st.slider("Coolant Temp (°C)", 70, 125, int(st.session_state.sensors['temp']))
    egt = st.slider("Exhaust Gas Temp (°C)", 200, 800, st.session_state.sensors['egt'])
    fuel = st.slider("Fuel Rail Pressure (bar)", 500, 2500, st.session_state.sensors['fuel'])
    intake = st.slider("Intake Air Temp (°C)", 20, 90, st.session_state.sensors['intake'])
    trans = st.slider("Transmission Temp (°C)", 50, 130, st.session_state.sensors['trans'])
    pressure = st.slider("Hydraulic Pressure (bar)", 100, 250, int(st.session_state.sensors['pressure']))
    flow = st.slider("Hydraulic Flow (L/min)", 10, 120, st.session_state.sensors['flow'])
    pto = st.slider("PTO Speed (RPM)", 0, 600, st.session_state.sensors['pto'])
    draft = st.slider("Draft Load (kN)", 0, 60, st.session_state.sensors['draft'])
    slip = st.slider("Wheel Slip (%)", 0, 50, st.session_state.sensors['slip'])
    radar = st.slider("Radar Speed (km/h)", 0.0, 40.0, float(st.session_state.sensors['radar']))
    steer = st.slider("Steering Angle (°)", -45, 45, st.session_state.sensors['steer'])
    battery = st.slider("Battery Voltage (V)", 10.0, 15.0, round(st.session_state.sensors['battery'], 1))

input_dict = {
    'Engine_RPM': rpm, 'Engine_Load_pct': load, 'Coolant_Temp_C': temp,
    'Exhaust_Gas_Temp_C': egt, 'Fuel_Rail_Pressure_bar': fuel, 'Intake_Air_Temp_C': intake,
    'Transmission_Temp_C': trans, 'Hydraulic_Pressure_bar': pressure, 'Hydraulic_Flow_Lpm': flow,
    'PTO_Speed_RPM': pto, 'Draft_Load_kN': draft, 'Wheel_Slip_pct': slip,
    'Radar_Speed_kmh': radar, 'Steering_Angle_deg': steer, 'Battery_Voltage_V': battery
}

# 1. Unlock the Random Seed for Realistic Live Jitter
np.random.seed(int(time.time())) 
for i in range(1, 22): input_dict[f'Engine_Micro_{i}'] = rpm * np.random.uniform(0.01, 0.05) + np.random.normal(0, 2)
for i in range(1, 22): input_dict[f'Hyd_Micro_{i}'] = pressure * np.random.uniform(0.8, 1.2) + np.random.normal(0, 1)
for i in range(1, 22): input_dict[f'Trans_Micro_{i}'] = trans * np.random.uniform(0.9, 1.1) + np.random.normal(0, 0.5)
for i in range(1, 23): input_dict[f'Elec_Micro_{i}'] = battery * np.random.uniform(0.95, 1.05) + np.random.normal(0, 0.05)

display_dict = input_dict.copy()
display_dict['Time'] = time.strftime('%H:%M:%S')
st.session_state.history.append(display_dict)

input_df = pd.DataFrame([input_dict])

# 2. Scale the Live Data before feeding to AI
input_scaled = scaler.transform(input_df)

prediction = rf_model.predict(input_scaled)[0]
confidence = max(rf_model.predict_proba(input_scaled)[0]) * 100
importances = rf_model.feature_importances_
anomaly = anomaly_model.predict(input_scaled)[0]

# --- Dynamic Health Score & Remaining Useful Life (RUL) ---
health_score = 100.0
stress_penalty = ((load / 100.0) * 8) + (((temp - 70) / 55.0) * 7)
health_score -= stress_penalty
if prediction != 0: health_score -= (confidence * 0.4) 
if anomaly == -1: health_score -= 25
health_score = max(0.0, min(100.0, health_score))

# RUL Estimation (Base 500 hours, degrades rapidly under fault/stress)
rul_hours = max(0, 500 - ((100 - health_score) * 4.5))

def generate_dynamic_diagnostics(p_rpm, p_load, p_temp, p_egt, p_fuel, p_intake, p_trans, p_press, p_flow, p_pto, p_draft, p_slip, p_batt):
    conditions, fixes = [], []
    if p_load > 85: conditions.append(f"sustaining severe mechanical load ({p_load}%)"); fixes.append("Shift to a lower gear.")
    if p_temp > 105: conditions.append(f"experiencing critical coolant thermal stress ({p_temp}°C)"); fixes.append("Inspect radiator fins.")
    if p_egt > 650: conditions.append(f"pushing dangerous exhaust gas temperatures ({p_egt}°C)"); fixes.append("Decrease engine load immediately.")
    if p_fuel < 1000 and p_load > 50: conditions.append(f"losing common-rail fuel pressure ({p_fuel} bar)"); fixes.append("Replace primary diesel filters.")
    if p_press < 140 or p_flow < 40: conditions.append(f"losing hydraulic flow and pressure"); fixes.append("Check main hydraulic pump seals.")
    if p_draft > 40: conditions.append(f"experiencing severe implement draft resistance ({p_draft} kN)"); fixes.append("Raise 3-point hitch slightly.")
    if p_trans > 110: conditions.append(f"overheating transmission clutch packs ({p_trans}°C)"); fixes.append("Disengage implement to allow cooling.")
    if p_slip > 20 and p_draft > 30: conditions.append(f"spinning out under heavy pulling load ({p_slip}% slip)"); fixes.append("Engage differential lock.")
    if p_batt < 12.0: conditions.append(f"detecting dangerous voltage drops ({p_batt}V)"); fixes.append("Test alternator output.")

    if prediction == 0:
        if not conditions: return "✅ **STATUS:** Operating flawlessly.", "✔️ Continue standard operations."
        else: return f"⚠️ **STATUS:** No critical failures yet, but the system is " + ", and ".join(conditions) + ".", "**🔧 Preventative Actions:**\n" + "\n".join([f"- {f}" for f in fixes])
    
    dtc_names = {1: "HYD-001", 2: "ENG-002", 3: "TRN-003", 4: "ELE-004", 5: "PTO-005"}
    return f"❌ **CRITICAL AI FAULT [{dtc_names.get(prediction)}]:** Failure signature identified because the machine is " + ", and ".join(conditions) + ".", "**🛠️ IMMEDIATE ACTIONS:**\n" + "\n".join([f"- {f}" for f in fixes])

status_msg, fix_msg = generate_dynamic_diagnostics(rpm, load, temp, egt, fuel, intake, trans, pressure, flow, pto, draft, slip, battery)

# --- UI Layout: Top Tier ---
col_health, col_conf, col_rul = st.columns(3)
with col_health: st.metric(label="Overall Health Score", value=f"{health_score:.1f}%", delta="Stable" if health_score > 80 else "Degrading", delta_color="normal" if health_score > 80 else "inverse")
with col_conf: st.progress(confidence/100, text=f"AI Diagnostic Confidence: {confidence:.1f}%")
with col_rul: st.metric("Estimated Remaining Useful Life", f"{rul_hours:.0f} hrs", delta="- Accelerated Wear" if rul_hours < 350 else "Normal Wear", delta_color="inverse" if rul_hours < 350 else "normal")

st.divider()

col_alert, col_xai = st.columns([1.3, 1])
with col_alert:
    st.markdown("### Prescriptive Diagnostics")
    if anomaly == -1: st.error("🚨 **UNKNOWN ANOMALY DETECTED:** Sensor patterns fall outside scaled distribution!")
    if prediction == 0 and "flawlessly" in status_msg: st.success(status_msg); st.info(fix_msg)
    elif prediction == 0: st.warning(status_msg); st.info(fix_msg)
    else: st.error(status_msg); st.error(fix_msg)

with col_xai:
    st.markdown("### Explainable AI (Stable Feature Importance)")
    importance_df = pd.DataFrame({"feature": input_df.columns, "importance": importances}).sort_values("importance", ascending=False).head(10)
    fig_xai = px.bar(importance_df, x="importance", y="feature", orientation='h')
    fig_xai.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(tickfont=dict(size=10)))
    fig_xai.update_traces(marker_color='#eab308')
    st.plotly_chart(fig_xai, use_container_width=True, config={'displayModeBar': False})

# --- UI Layout: Middle Tier (Gauges & Radar) ---
with st.container(border=True):
    col_g, col_rad = st.columns([1.5, 1])
    with col_g:
        st.markdown("### Live ISOBUS Telemetry")
        def sleek_gauge(val, title, min_v, max_v, color):
            fig = go.Figure(go.Indicator(mode="gauge+number", value=val, title={'text': title, 'font': {'size': 14}}, number={'font': {'color': color}}, gauge={'axis': {'range': [min_v, max_v]}, 'bar': {'color': color, 'thickness': 0.8}, 'bgcolor': "rgba(0,0,0,0.05)"}))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=200, margin=dict(l=15, r=15, t=30, b=10))
            return fig
        g1, g2 = st.columns(2); g3, g4 = st.columns(2)
        with g1: st.plotly_chart(sleek_gauge(rpm, "Engine (RPM)", 800, 2500, "#3b82f6"), use_container_width=True, config={'displayModeBar': False})
        with g2: st.plotly_chart(sleek_gauge(temp, "Coolant (°C)", 70, 125, "#ef4444"), use_container_width=True, config={'displayModeBar': False})
        with g3: st.plotly_chart(sleek_gauge(pressure, "Hydraulic (bar)", 100, 250, "#10b981"), use_container_width=True, config={'displayModeBar': False})
        with g4: st.plotly_chart(sleek_gauge(slip, "Wheel Slip (%)", 0, 50, "#f59e0b"), use_container_width=True, config={'displayModeBar': False})
    
    with col_rad:
        st.markdown("### Stress Distribution")
        radar_df = pd.DataFrame(dict(r=[load, min((temp/125)*100, 100), min((pressure/250)*100, 100), min((draft/60)*100, 100), min((slip/50)*100, 100)], theta=['Engine Load', 'Thermal Stress', 'Hydraulic Load', 'Draft Pull', 'Traction Loss']))
        fig_radar = px.line_polar(radar_df, r='r', theta='theta', line_close=True)
        fig_radar.update_traces(fill='toself', line_color='#eab308')
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=30, r=30, t=20, b=20))
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})

# --- UI Layout: Bottom Tier (Fleet & History) ---
col_fleet, col_hist = st.columns([1, 1.5])
with col_fleet:
    with st.container(border=True):
        st.markdown("### 🚜 Fleet Monitoring Matrix")
        # Live unit is first, followed by simulated fleet members
        fleet = pd.DataFrame({
            "Unit ID": ["TRX-01 (Live)", "TRX-02", "TRX-03", "TRX-04"],
            "Location": ["Agra Sec-A", "Agra Sec-B", "Mathura Hub", "Noida Field"],
            "Health Score": [f"{health_score:.1f}%", "98.2%", "81.5%", "64.0%"],
            "Status": [
                "CRITICAL" if prediction != 0 or anomaly == -1 else "Nominal", 
                "Nominal", "Warning", "CRITICAL: ERR-HYD-001"
            ]
        })
        st.dataframe(fleet, use_container_width=True, hide_index=True)

with col_hist:
    with st.container(border=True):
        st.markdown("### 📈 Time-Series History Buffer")
        history_df = pd.DataFrame(st.session_state.history)
        fig_history = px.line(history_df, x='Time', y=['Engine_RPM', 'Coolant_Temp_C', 'Hydraulic_Pressure_bar', 'Exhaust_Gas_Temp_C'], markers=True)
        fig_history.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_history, use_container_width=True, config={'displayModeBar': False})

if streaming:
    st.session_state.sensors['rpm'] = int(np.clip(st.session_state.sensors['rpm'] + np.random.normal(0, 15), 800, 2500))
    st.session_state.sensors['pressure'] = np.clip(st.session_state.sensors['pressure'] + np.random.normal(0, 1.5), 100, 250)
    time.sleep(0.5)
    st.rerun()