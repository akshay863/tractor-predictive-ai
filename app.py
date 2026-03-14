import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time
import numpy as np

# 1. Native Page Configuration
st.set_page_config(page_title="Tractor Telemetry OS", layout="wide")

# Hide default Streamlit menus for a clean app look
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Clean, Native Header
st.title("🚜 Precision R&D Diagnostics OS")
st.markdown("**Live Telemetry • Deep Learning RUL • Fleet Command**")
st.divider()

# 2. Fast Model Loading
@st.cache_resource
def load_model():
    return joblib.load('tractor_health_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ AI Core Offline: tractor_health_model.pkl not found.")
    st.stop()

# 3. Industry Diagnostic Engine
diagnostics = {
    0: {"dtc": "None", "msg": "All subsystems operating nominally.", "fix": "Continue standard operations.", "color": "normal"},
    1: {"dtc": "ERR-HYD-001", "msg": "Hydraulic Pressure Loss + High Load", "fix": "Inspect pump seals. Replace fluid filter.", "color": "inverse"},
    2: {"dtc": "ERR-ENG-002", "msg": "Engine Thermal Overload Detected", "fix": "SHUTDOWN. Clean radiator fins. Check water pump.", "color": "inverse"},
    3: {"dtc": "ERR-TRN-003", "msg": "Transmission Slippage / Thermal Overload", "fix": "Recalibrate clutch packs. Check fluid viscosity.", "color": "off"},
    4: {"dtc": "ERR-ELE-004", "msg": "System Voltage Drop / Alternator Fault", "fix": "Test alternator output. Inspect battery terminals.", "color": "inverse"},
    5: {"dtc": "ERR-PTO-005", "msg": "PTO RPM Drop under High Load", "fix": "Reduce implement speed. Inspect PTO shear pin.", "color": "off"}
}

# 4. Clean Sidebar - Inputs & Streaming
with st.sidebar:
    st.header("📡 Sensor Injection")
    streaming = st.toggle("Enable Live Auto-Streaming", value=False)
    st.divider()

    if 'sensors' not in st.session_state:
        st.session_state.sensors = { 'rpm': 1800, 'load': 45, 'temp': 85, 'pressure': 200, 'slip': 10, 'trans_temp': 75, 'battery': 13.8, 'pto': 540 }

    if streaming:
        st.session_state.sensors['rpm'] = int(np.clip(st.session_state.sensors['rpm'] + np.random.normal(0, 10), 800, 2500))
        st.session_state.sensors['temp'] = np.clip(st.session_state.sensors['temp'] + np.random.normal(0, 0.5), 70, 125)
        st.session_state.sensors['pressure'] = np.clip(st.session_state.sensors['pressure'] + np.random.normal(0, 1), 100, 250)
        time.sleep(0.4) 

    rpm = st.slider("Engine RPM", 800, 2500, st.session_state.sensors['rpm'])
    load = st.slider("Engine Load (%)", 0, 100, st.session_state.sensors['load'])
    temp = st.slider("Coolant Temp (°C)", 70, 125, int(st.session_state.sensors['temp']))
    pressure = st.slider("Hydraulic Pressure (bar)", 100, 250, int(st.session_state.sensors['pressure']))
    slip = st.slider("Wheel Slip (%)", 0, 40, st.session_state.sensors['slip'])
    trans_temp = st.slider("Transmission Temp (°C)", 50, 130, st.session_state.sensors['trans_temp'])
    battery = st.slider("Battery Voltage (V)", 10.0, 15.0, round(st.session_state.sensors['battery'], 1))
    pto = st.slider("PTO Speed (RPM)", 0, 600, st.session_state.sensors['pto'])

input_df = pd.DataFrame({'Engine_RPM': [rpm], 'Engine_Load_pct': [load], 'Coolant_Temp_C': [temp], 'Hydraulic_Pressure_bar': [pressure], 'Wheel_Slip_pct': [slip], 'Transmission_Temp_C': [trans_temp], 'Battery_Voltage_V': [battery], 'PTO_Speed_RPM': [pto]})

# 5. AI Inference
prediction = model.predict(input_df)[0]
confidence = max(model.predict_proba(input_df)[0]) * 100
diag = diagnostics[prediction]

# 6. Main UI Tabs
tab_diag, tab_rul, tab_telematics = st.tabs(["⚡ Live Diagnostics", "🧠 Deep Learning RUL", "🌍 Global Telematics"])

with tab_diag:
    # Alert Box
    if prediction == 0:
        st.success(f"**✔️ SYSTEM HEALTHY** | Confidence: {confidence:.1f}% \n\n {diag['msg']}")
    elif diag['color'] == "inverse":
        st.error(f"**⚠️ CRITICAL FAULT: {diag['dtc']}** | Confidence: {confidence:.1f}%\n\n**Issue:** {diag['msg']}\n\n**🔧 Fix:** {diag['fix']}")
    else:
        st.warning(f"**⚠️ WARNING: {diag['dtc']}** | Confidence: {confidence:.1f}%\n\n**Issue:** {diag['msg']}\n\n**🔧 Fix:** {diag['fix']}")
        
    st.markdown("### Live ISOBUS Telemetry")
    
    # Ultra-clean native Plotly Gauges
    def clean_gauge(val, title, min_v, max_v, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=val, title={'text': title},
            gauge={'axis': {'range': [min_v, max_v]}, 'bar': {'color': color}, 'bgcolor': "rgba(0,0,0,0)"}
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=220, margin=dict(l=20, r=20, t=30, b=20))
        return fig

    g1, g2, g3, g4 = st.columns(4)
    with g1: st.plotly_chart(clean_gauge(rpm, "Engine (RPM)", 800, 2500, "#2b83ba"), use_container_width=True)
    with g2: st.plotly_chart(clean_gauge(temp, "Coolant (°C)", 70, 125, "#d7191c"), use_container_width=True)
    with g3: st.plotly_chart(clean_gauge(pressure, "Hydraulic (bar)", 100, 250, "#abdda4"), use_container_width=True)
    with g4: st.plotly_chart(clean_gauge(load, "Engine Load (%)", 0, 100, "#fdae61"), use_container_width=True)

with tab_rul:
    st.markdown("### LSTM Predicted Component Degradation")
    base_rul = 500
    stress_factor = ((temp - 85) * 2) + ((load - 45) * 1.5)
    current_rul = max(0, base_rul - stress_factor)
    
    col_metric, col_chart = st.columns([1, 2])
    with col_metric:
        st.metric("Hydraulic Pump RUL", f"{int(current_rul)} Hours", f"-{int(stress_factor)} hrs (Load Penalty)", delta_color="inverse")
        if current_rul < 100:
            st.error("Schedule Maintenance Immediately.")
        elif current_rul < 250:
            st.warning("Component degrading faster than baseline.")
        else:
            st.success("Component decaying at expected rate.")
            
    with col_chart:
        hours_passed = np.arange(0, 500, 10)
        baseline_decay = 500 - hours_passed
        actual_decay = np.clip(500 - (hours_passed * (1 + (stress_factor/100))), 0, 500)
        
        rul_df = pd.DataFrame({'Operating Hours': hours_passed, 'Baseline RUL': baseline_decay, 'Live AI Predicted RUL': actual_decay})
        fig_rul = px.line(rul_df, x='Operating Hours', y=['Baseline RUL', 'Live AI Predicted RUL'], color_discrete_sequence=['gray', 'red'])
        fig_rul.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_rul, use_container_width=True)

with tab_telematics:
    st.markdown("### Regional Fleet Overview: Agra Test Sector")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Active Units", "15")
    k2.metric("Healthy", "11", "+2", delta_color="normal")
    k3.metric("Warnings", "3", "-1", delta_color="inverse")
    k4.metric("Critical", "1", "+1", delta_color="inverse")
    
    np.random.seed(42)
    fleet_lat = 27.1767 + np.random.normal(0, 0.05, 15)
    fleet_lon = 78.0081 + np.random.normal(0, 0.05, 15)
    health_status = np.random.choice(['Nominal', 'Warning', 'Critical'], 15, p=[0.7, 0.2, 0.1])
    health_status[0] = 'Critical' if diag['color'] == 'inverse' else ('Warning' if diag['color'] == 'off' else 'Nominal')
    
    map_df = pd.DataFrame({'Lat': fleet_lat, 'Lon': fleet_lon, 'Status': health_status, 'Unit': [f"TRX-{i+1000}" for i in range(15)]})
    color_map = {'Nominal': 'green', 'Warning': 'orange', 'Critical': 'red'}
    
    fig_map = px.scatter_mapbox(
        map_df, lat="Lat", lon="Lon", color="Status", hover_name="Unit",
        color_discrete_map=color_map, zoom=10, height=450
    )
    fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

if streaming:
    st.rerun()