import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Enterprise Fleet Diagnostics", layout="wide")

st.markdown("""
    <div style='background-color: #0e1117; padding: 20px; border-radius: 10px; border-left: 10px solid #ff4b4b;'>
        <h1 style='margin:0; color: white;'>🌐 Global Fleet Operations Center</h1>
        <p style='margin:0; color: #a8a8a8;'>Real-Time Telemetry | Explainable AI | Prescriptive Maintenance</p>
    </div>
    <br>
""", unsafe_allow_html=True)

# 2. Load the Master AI Model
@st.cache_resource
def load_model():
    return joblib.load('tractor_health_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file not found. Please upload tractor_health_model.pkl")
    st.stop()

# 3. Industry Diagnostic Dictionary
diagnostics = {
    0: {"dtc": "None", "msg": "All subsystems operating nominally.", "fix": "Continue standard operations."},
    1: {"dtc": "ERR-HYD-001", "msg": "Hydraulic Pressure Loss under Heavy Engine Load.", "fix": "Inspect main hydraulic pump seals. Replace fluid filter."},
    2: {"dtc": "ERR-ENG-002", "msg": "Engine Thermal Overload / Coolant Spike.", "fix": "IMMEDIATE SHUTDOWN. Clean radiator fins. Check water pump belt."},
    3: {"dtc": "ERR-TRN-003", "msg": "Transmission Slippage / Thermal Overload.", "fix": "Recalibrate clutch packs. Check transmission fluid viscosity."},
    4: {"dtc": "ERR-ELE-004", "msg": "System Voltage Drop / Alternator Fault.", "fix": "Test alternator output. Inspect battery terminals."},
    5: {"dtc": "ERR-PTO-005", "msg": "PTO RPM Drop under High Engine Load.", "fix": "Reduce implement speed. Inspect PTO shear pin."}
}

# 4. Navigation Tabs
tab_twin, tab_fleet = st.tabs(["🚜 Single Unit Digital Twin", "🌍 Fleet Command Center"])

with tab_twin:
    # --- SIDEBAR: SIMULATED STREAMING & INPUTS ---
    st.sidebar.header("📡 Live CAN Bus Feed")
    streaming = st.sidebar.checkbox("🟢 Enable Live Auto-Streaming")
    
    # Initialize session state for sensor values so they can update dynamically
    if 'sensors' not in st.session_state:
        st.session_state.sensors = {
            'rpm': 1800, 'load': 45, 'temp': 85, 'pressure': 200,
            'slip': 10, 'trans_temp': 75, 'battery': 13.8, 'pto': 540
        }

    # If streaming is on, add slight random fluctuations to simulate a live tractor
    if streaming:
        st.session_state.sensors['rpm'] = int(np.clip(st.session_state.sensors['rpm'] + np.random.normal(0, 10), 800, 2500))
        st.session_state.sensors['temp'] = np.clip(st.session_state.sensors['temp'] + np.random.normal(0, 0.5), 70, 125)
        st.session_state.sensors['pressure'] = np.clip(st.session_state.sensors['pressure'] + np.random.normal(0, 1), 100, 250)
        time.sleep(0.5) # Refresh rate

    # Manual Sliders (Updated by session state or user input)
    rpm = st.sidebar.slider("Engine RPM", 800, 2500, st.session_state.sensors['rpm'])
    load = st.sidebar.slider("Engine Load (%)", 0, 100, st.session_state.sensors['load'])
    temp = st.sidebar.slider("Coolant Temp (°C)", 70, 125, int(st.session_state.sensors['temp']))
    pressure = st.sidebar.slider("Hydraulic Pressure (bar)", 100, 250, int(st.session_state.sensors['pressure']))
    slip = st.sidebar.slider("Wheel Slip (%)", 0, 40, st.session_state.sensors['slip'])
    trans_temp = st.sidebar.slider("Transmission Temp (°C)", 50, 130, st.session_state.sensors['trans_temp'])
    battery = st.sidebar.slider("Battery Voltage (V)", 10.0, 15.0, st.session_state.sensors['battery'])
    pto = st.sidebar.slider("PTO Speed (RPM)", 0, 600, st.session_state.sensors['pto'])

    # Format input
    input_df = pd.DataFrame({
        'Engine_RPM': [rpm], 'Engine_Load_pct': [load], 'Coolant_Temp_C': [temp],
        'Hydraulic_Pressure_bar': [pressure], 'Wheel_Slip_pct': [slip],
        'Transmission_Temp_C': [trans_temp], 'Battery_Voltage_V': [battery],
        'PTO_Speed_RPM': [pto]
    })

    # --- ADVANCED AI INFERENCE ---
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    confidence = max(probabilities) * 100

    # Get feature importances for this specific model architecture
    importances = model.feature_importances_
    features = input_df.columns

    # --- MAIN DASHBOARD UI ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Diagnostic Status")
        diag = diagnostics[prediction]
        if prediction == 0:
            st.success(f"✔️ **NOMINAL** | Confidence: {confidence:.1f}% | {diag['msg']}")
        else:
            st.error(f"⚠️ **DTC: {diag['dtc']}** | Confidence: {confidence:.1f}%")
            st.warning(f"**Issue:** {diag['msg']}")
            st.info(f"**🔧 Prescriptive Fix:** {diag['fix']}")

        st.markdown("---")
        st.subheader("Live Subsystem Telemetry")
        
        # Helper function for gauges
        def build_gauge(val, title, min_val, max_val, color):
            return go.Figure(go.Indicator(
                mode="gauge+number", value=val, title={'text': title},
                gauge={'axis': {'range': [min_val, max_val]}, 'bar': {'color': color}}
            )).update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))

        g1, g2, g3 = st.columns(3)
        with g1: st.plotly_chart(build_gauge(rpm, "RPM", 800, 2500, "#1f77b4"), use_container_width=True)
        with g2: st.plotly_chart(build_gauge(temp, "Coolant (°C)", 70, 125, "#d62728"), use_container_width=True)
        with g3: st.plotly_chart(build_gauge(pressure, "Hydraulic (bar)", 100, 250, "#2ca02c"), use_container_width=True)

    with col2:
        st.subheader("Explainable AI (XAI)")
        st.markdown("Metrics driving the current prediction:")
        
        # Explainable AI Bar Chart
        fig_xai = px.bar(
            x=importances, y=features, orientation='h', 
            labels={'x': 'Impact Factor', 'y': 'Sensor'},
            color=importances, color_continuous_scale='Reds'
        )
        fig_xai.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_xai, use_container_width=True)
        
        st.subheader("Fault Probability Matrix")
        st.progress(probabilities[0], text=f"Normal: {probabilities[0]*100:.1f}%")
        st.progress(probabilities[1], text=f"Hydraulic Fault: {probabilities[1]*100:.1f}%")
        st.progress(probabilities[2], text=f"Engine Fault: {probabilities[2]*100:.1f}%")
        st.progress(probabilities[3], text=f"Transmission Fault: {probabilities[3]*100:.1f}%")

with tab_fleet:
    st.subheader("Regional Fleet Overview: Northern India Sector")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Active Units", "142")
    kpi2.metric("Healthy Units", "136", "95.7%", delta_color="normal")
    kpi3.metric("Active Warnings", "4", "-2 from yesterday", delta_color="inverse")
    kpi4.metric("Critical Breakdowns", "2", "+1 from last hour", delta_color="inverse")
    
    st.markdown("---")
    
    # Mock Fleet Data Table
    fleet_data = pd.DataFrame({
        "Unit ID": ["TRX-9901", "TRX-4421", "TRX-8832", "TRX-1092", "TRX-5514"],
        "Location": ["Agra Sec-4", "Jaipur Field B", "Mohali Agri-Hub", "Faridabad Dep.", "Greater Noida Zone"],
        "Operating Hours": [1450, 890, 3200, 410, 2150],
        "AI Status": ["CRITICAL: ERR-ENG-002", "WARNING: ERR-HYD-001", "NOMINAL", "NOMINAL", "WARNING: ERR-ELE-004"]
    })
    
    st.dataframe(fleet_data, use_container_width=True, hide_index=True)

# Loop the app if streaming is active
if streaming:
    st.rerun()