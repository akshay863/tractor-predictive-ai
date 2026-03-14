import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Enterprise Telemetry OS", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp { background-color: #0E1117; color: #FAFAFA; font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
    .premium-header {
        background: linear-gradient(90deg, #1f2937 0%, #111827 100%);
        padding: 20px; border-radius: 12px; border-left: 6px solid #8b5cf6;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); margin-bottom: 25px;
    }
    .premium-header h1 { margin: 0; color: #ffffff; font-size: 28px; font-weight: 600; }
    .premium-header p { margin: 5px 0 0 0; color: #9ca3af; font-size: 14px; letter-spacing: 1px; }
    .stSlider > div > div > div > div { background-color: #8b5cf6 !important; }
    </style>
""", unsafe_allow_html=True)

# App Header (Authenticated as Lead Engineer)
st.markdown("""
    <div class="premium-header">
        <h1>🚜 Deep Learning Telemetry & RUL Engine</h1>
        <p>LEAD ENGINEER: AKSHAY KUMAR SHARMA | LIVE GPS TELEMATICS | LSTM PREDICTIVE MAINTENANCE</p>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('tractor_health_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ AI Core Offline: tractor_health_model.pkl not found.")
    st.stop()

diagnostics = {
    0: {"dtc": "None", "msg": "All subsystems nominal.", "fix": "Continue operations.", "color": "success"},
    1: {"dtc": "ERR-HYD-001", "msg": "Hydraulic Pressure Loss + High Load", "fix": "Inspect pump seals. Replace fluid filter.", "color": "error"},
    2: {"dtc": "ERR-ENG-002", "msg": "Engine Thermal Overload Detected", "fix": "SHUTDOWN. Clean radiator fins. Check water pump.", "color": "error"},
    3: {"dtc": "ERR-TRN-003", "msg": "Transmission Slippage / Thermal Overload", "fix": "Recalibrate clutch packs. Check fluid viscosity.", "color": "warning"},
    4: {"dtc": "ERR-ELE-004", "msg": "System Voltage Drop / Alternator Fault", "fix": "Test alternator output. Inspect battery terminals.", "color": "error"},
    5: {"dtc": "ERR-PTO-005", "msg": "PTO RPM Drop under High Load", "fix": "Reduce implement speed. Inspect PTO shear pin.", "color": "warning"}
}

# --- Sidebar Inputs & Auto-Streaming ---
with st.sidebar:
    st.markdown("### 📡 Sensor Injection")
    streaming = st.toggle("🟢 Enable Live Data Streaming", value=False)
    st.markdown("---")

    if 'sensors' not in st.session_state:
        st.session_state.sensors = { 'rpm': 1800, 'load': 45, 'temp': 85, 'pressure': 200, 'slip': 10, 'trans_temp': 75, 'battery': 13.8, 'pto': 540 }

    if streaming:
        st.session_state.sensors['rpm'] = int(np.clip(st.session_state.sensors['rpm'] + np.random.normal(0, 15), 800, 2500))
        st.session_state.sensors['temp'] = np.clip(st.session_state.sensors['temp'] + np.random.normal(0, 0.3), 70, 125)
        st.session_state.sensors['pressure'] = np.clip(st.session_state.sensors['pressure'] + np.random.normal(0, 1.5), 100, 250)
        time.sleep(0.3)

    rpm = st.slider("Engine RPM", 800, 2500, st.session_state.sensors['rpm'])
    load = st.slider("Engine Load (%)", 0, 100, st.session_state.sensors['load'])
    temp = st.slider("Coolant Temp (°C)", 70, 125, int(st.session_state.sensors['temp']))
    pressure = st.slider("Hydraulic Pressure (bar)", 100, 250, int(st.session_state.sensors['pressure']))
    slip = st.slider("Wheel Slip (%)", 0, 40, st.session_state.sensors['slip'])
    trans_temp = st.slider("Transmission Temp (°C)", 50, 130, st.session_state.sensors['trans_temp'])
    battery = st.slider("Battery Voltage (V)", 10.0, 15.0, round(st.session_state.sensors['battery'], 1))
    pto = st.slider("PTO Speed (RPM)", 0, 600, st.session_state.sensors['pto'])

input_df = pd.DataFrame({'Engine_RPM': [rpm], 'Engine_Load_pct': [load], 'Coolant_Temp_C': [temp], 'Hydraulic_Pressure_bar': [pressure], 'Wheel_Slip_pct': [slip], 'Transmission_Temp_C': [trans_temp], 'Battery_Voltage_V': [battery], 'PTO_Speed_RPM': [pto]})

prediction = model.predict(input_df)[0]
confidence = max(model.predict_proba(input_df)[0]) * 100
diag = diagnostics[prediction]

# --- Main UI Tabs ---
tab_rul, tab_telematics, tab_diag = st.tabs(["🧠 Deep Learning RUL", "🌍 Live GPS Telematics", "⚡ Diagnostic Feed"])

with tab_rul:
    st.markdown("#### LSTM Neural Network: Remaining Useful Life (RUL)")
    st.markdown("Simulated output from time-series deep learning models predicting hardware degradation.")
    
    # Calculate simulated RUL based on current stress (temperature & load)
    base_rul = 500
    stress_factor = ((temp - 85) * 2) + ((load - 45) * 1.5)
    current_rul = max(0, base_rul - stress_factor)
    
    col_rul1, col_rul2 = st.columns([1, 2])
    with col_rul1:
        st.metric("Hydraulic Pump RUL", f"{int(current_rul)} Hours", f"-{int(stress_factor)} hrs (Load Penalty)", delta_color="inverse")
        if current_rul < 100:
            st.error("⚠️ CRITICAL: Schedule Maintenance Immediately.")
        elif current_rul < 250:
            st.warning("⚠️ WARNING: Component degrading faster than baseline.")
        else:
            st.success("✔️ Component decaying at expected linear rate.")
            
    with col_rul2:
        # Generate simulated decay curve
        hours_passed = np.arange(0, 500, 10)
        baseline_decay = 500 - hours_passed
        actual_decay = 500 - (hours_passed * (1 + (stress_factor/100)))
        actual_decay = np.clip(actual_decay, 0, 500)
        
        rul_df = pd.DataFrame({'Operating Hours': hours_passed, 'Baseline RUL': baseline_decay, 'Live AI Predicted RUL': actual_decay})
        fig_rul = px.line(rul_df, x='Operating Hours', y=['Baseline RUL', 'Live AI Predicted RUL'], template="plotly_dark", color_discrete_sequence=['#9ca3af', '#ef4444'])
        fig_rul.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_rul, use_container_width=True)

with tab_telematics:
    st.markdown("#### Global Fleet Telemetry: Agra Test Sector")
    
    # Generate fleet GPS data centered around Agra, UP
    np.random.seed(42)
    fleet_lat = 27.1767 + np.random.normal(0, 0.05, 15)
    fleet_lon = 78.0081 + np.random.normal(0, 0.05, 15)
    health_status = np.random.choice(['Nominal', 'Warning', 'Critical'], 15, p=[0.7, 0.2, 0.1])
    
    # Sync unit 1 with our live sliders
    health_status[0] = 'Critical' if diag['color'] == 'error' else ('Warning' if diag['color'] == 'warning' else 'Nominal')
    
    map_df = pd.DataFrame({'Lat': fleet_lat, 'Lon': fleet_lon, 'Status': health_status, 'Unit': [f"TRX-{i+1000}" for i in range(15)]})
    color_map = {'Nominal': '#00cc96', 'Warning': '#f59e0b', 'Critical': '#ef4444'}
    
    fig_map = px.scatter_mapbox(
        map_df, lat="Lat", lon="Lon", color="Status", hover_name="Unit",
        color_discrete_map=color_map, zoom=10, height=400
    )
    fig_map.update_layout(mapbox_style="carto-darkmatter", margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_map, use_container_width=True)

with tab_diag:
    # Top Row: Alert Box
    if prediction == 0:
        st.success(f"**✔️ ALL SYSTEMS NOMINAL** | Confidence: {confidence:.1f}%\n\n{diag['msg']}")
    elif diag['color'] == "error":
        st.error(f"**⚠️ CRITICAL DTC: {diag['dtc']}** | Confidence: {confidence:.1f}%\n\n**Issue:** {diag['msg']}\n\n**🔧 Fix:** {diag['fix']}")
    else:
        st.warning(f"**⚠️ WARNING DTC: {diag['dtc']}** | Confidence: {confidence:.1f}%\n\n**Issue:** {diag['msg']}\n\n**🔧 Fix:** {diag['fix']}")
        
    st.markdown("---")
    
    # Ultra-Fast Plotly Gauges
    def sleek_gauge(val, title, min_v, max_v, color):
        fig = go.Figure(go.Indicator(mode="gauge+number", value=val, title={'text': title, 'font': {'size': 14, 'color': 'white'}}, gauge={'axis': {'range': [min_v, max_v], 'tickwidth': 1, 'tickcolor': "white"}, 'bar': {'color': color, 'thickness': 0.7}, 'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 2, 'bordercolor': "#374151"}))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=200, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    g1, g2, g3, g4 = st.columns(4)
    with g1: st.plotly_chart(sleek_gauge(rpm, "Engine (RPM)", 800, 2500, "#8b5cf6"), use_container_width=True, config={'displayModeBar': False})
    with g2: st.plotly_chart(sleek_gauge(temp, "Coolant (°C)", 70, 125, "#ef4444"), use_container_width=True, config={'displayModeBar': False})
    with g3: st.plotly_chart(sleek_gauge(pressure, "Hydraulic (bar)", 100, 250, "#10b981"), use_container_width=True, config={'displayModeBar': False})
    with g4: st.plotly_chart(sleek_gauge(load, "Engine Load (%)", 0, 100, "#f59e0b"), use_container_width=True, config={'displayModeBar': False})

if streaming:
    st.rerun()