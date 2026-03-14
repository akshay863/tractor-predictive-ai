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
        <h1>🚜 Industrial Digital Twin & Telemetry OS</h1>
        <p>TIME-SERIES BUFFER • UNSUPERVISED ANOMALY DETECTION • DYNAMIC PRESCRIPTIVE MAINTENANCE</p>
    </div>
""", unsafe_allow_html=True)

# Load Both AI Models
@st.cache_resource
def load_models(): 
    return joblib.load('tractor_health_model.pkl'), joblib.load('anomaly_model.pkl')

try: 
    rf_model, anomaly_model = load_models()
except FileNotFoundError: 
    st.error("⚠️ AI Cores Offline: Please run predictive_model.py first.")
    st.stop()

# Initialize Time-Series History Buffer (stores last 50 readings)
if "history" not in st.session_state:
    st.session_state.history = deque(maxlen=50)

# --- Sidebar Inputs ---
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

# Add 85 Micro-Sensors with Noise
np.random.seed(int(time.time())) 
for i in range(1, 22): input_dict[f'Engine_Micro_{i}'] = rpm * np.random.uniform(0.01, 0.05) + np.random.normal(0, 2)
for i in range(1, 22): input_dict[f'Hyd_Micro_{i}'] = pressure * np.random.uniform(0.8, 1.2) + np.random.normal(0, 1)
for i in range(1, 22): input_dict[f'Trans_Micro_{i}'] = trans * np.random.uniform(0.9, 1.1) + np.random.normal(0, 0.5)
for i in range(1, 23): input_dict[f'Elec_Micro_{i}'] = battery * np.random.uniform(0.95, 1.05) + np.random.normal(0, 0.05)

# --- Time-Series State Management ---
# Add a timestamp to the input dictionary and append it to the buffer
display_dict = input_dict.copy()
display_dict['Time'] = time.strftime('%H:%M:%S')
st.session_state.history.append(display_dict)

input_df = pd.DataFrame([input_dict])

# --- AI Inferences ---
prediction = rf_model.predict(input_df)[0]
confidence = max(rf_model.predict_proba(input_df)[0]) * 100
importances = rf_model.feature_importances_

# Unsupervised Anomaly Detection (-1 means anomaly, 1 means normal)
anomaly = anomaly_model.predict(input_df)[0]

# --- Dynamic Industry Health Score ---
health_score = 100 - confidence if prediction != 0 else 95 + np.random.uniform(0, 5)
if anomaly == -1:
    health_score -= 20 # Massive penalty for unknown physics

# --- DYNAMIC NLP HEURISTIC ENGINE ---
def generate_dynamic_diagnostics(p_rpm, p_load, p_temp, p_egt, p_fuel, p_intake, p_trans, p_press, p_flow, p_pto, p_draft, p_slip, p_batt):
    conditions, fixes = [], []
    if p_load > 85: conditions.append(f"sustaining severe mechanical load ({p_load}%)"); fixes.append("Shift to a lower gear.")
    if p_temp > 105: conditions.append(f"experiencing critical coolant thermal stress ({p_temp}°C)"); fixes.append("Inspect radiator fins.")
    if p_egt > 650: conditions.append(f"pushing dangerous exhaust gas temperatures ({p_egt}°C)"); fixes.append("Decrease engine load immediately to save turbocharger.")
    if p_fuel < 1000 and p_load > 50: conditions.append(f"losing common-rail fuel pressure ({p_fuel} bar)"); fixes.append("Replace primary diesel filters.")
    if p_press < 140 or p_flow < 40: conditions.append(f"losing hydraulic flow and pressure"); fixes.append("Check main hydraulic pump seals.")
    if p_draft > 40: conditions.append(f"experiencing severe implement draft resistance ({p_draft} kN)"); fixes.append("Raise 3-point hitch slightly.")
    if p_trans > 110: conditions.append(f"overheating transmission clutch packs ({p_trans}°C)"); fixes.append("Disengage implement to allow cooling.")
    if p_slip > 20 and p_draft > 30: conditions.append(f"spinning out under heavy pulling load ({p_slip}% slip)"); fixes.append("Engage MFWD or differential lock.")
    if p_batt < 12.0: conditions.append(f"detecting dangerous voltage drops ({p_batt}V)"); fixes.append("Test alternator output.")

    if prediction == 0:
        if not conditions: return "✅ **STATUS:** The vehicle is operating flawlessly.", "✔️ Continue standard operations."
        else: return f"⚠️ **STATUS:** AI confirms no critical failures yet, but the system is currently " + ", and ".join(conditions) + ".", "**🔧 Preventative Actions:**\n" + "\n".join([f"- {f}" for f in fixes])
    
    dtc_names = {1: "HYD-001", 2: "ENG-002", 3: "TRN-003", 4: "ELE-004", 5: "PTO-005"}
    return f"❌ **CRITICAL AI FAULT [{dtc_names.get(prediction)}]:** Failure signature identified because the machine is " + ", and ".join(conditions) + ".", "**🛠️ IMMEDIATE ACTIONS:**\n" + "\n".join([f"- {f}" for f in fixes])

status_msg, fix_msg = generate_dynamic_diagnostics(rpm, load, temp, egt, fuel, intake, trans, pressure, flow, pto, draft, slip, battery)

# --- UI Layout: Health Metrics & Diagnostics ---
col_health, col_conf = st.columns(2)
with col_health:
    st.metric(label="Overall Machine Health Score", value=f"{health_score:.1f}%", delta="Stable" if health_score > 80 else "Degrading", delta_color="normal" if health_score > 80 else "inverse")
with col_conf:
    st.markdown("### AI Diagnostic Confidence")
    st.progress(confidence/100, text=f"{confidence:.1f}% Confidence Level")

st.divider()
col_alert, col_xai = st.columns([1.3, 1])
with col_alert:
    st.markdown("### Prescriptive Diagnostics")
    
    # Trigger Unsupervised Anomaly Alert
    if anomaly == -1:
        st.error("🚨 **UNKNOWN ANOMALY DETECTED:** Sensor patterns are falling outside the training distribution! Possible zero-day mechanical failure occurring.")

    if prediction == 0 and "flawlessly" in status_msg:
        st.success(status_msg); st.info(fix_msg)
    elif prediction == 0:
        st.warning(status_msg); st.info(fix_msg)
    else:
        st.error(status_msg); st.error(fix_msg)

with col_xai:
    st.markdown("### Explainable AI (Stable Feature Importance)")
    
    # Stabilize the XAI output
    importance_df = pd.DataFrame({"feature": input_df.columns, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False).head(10)
    
    fig_xai = px.bar(importance_df, x="importance", y="feature", orientation='h')
    fig_xai.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=250, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(tickfont=dict(size=10)))
    fig_xai.update_traces(marker_color='#eab308')
    st.plotly_chart(fig_xai, use_container_width=True, config={'displayModeBar': False})

# --- UI Layout: Time-Series Visuals ---
with st.container(border=True):
    st.markdown("### 📈 Live Telemetry History Buffer")
    history_df = pd.DataFrame(st.session_state.history)
    
    # Plotly Line Chart for the rolling buffer
    fig_history = px.line(history_df, x='Time', y=['Engine_RPM', 'Coolant_Temp_C', 'Hydraulic_Pressure_bar', 'Exhaust_Gas_Temp_C'], markers=True, template="plotly_dark")
    fig_history.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title="Time", yaxis_title="Sensor Value")
    st.plotly_chart(fig_history, use_container_width=True, config={'displayModeBar': False})

if streaming:
    st.session_state.sensors['rpm'] = int(np.clip(st.session_state.sensors['rpm'] + np.random.normal(0, 15), 800, 2500))
    st.session_state.sensors['pressure'] = np.clip(st.session_state.sensors['pressure'] + np.random.normal(0, 1.5), 100, 250)
    time.sleep(0.5)
    st.rerun()