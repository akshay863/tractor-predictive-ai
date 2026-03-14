import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time
import numpy as np

# 1. Page Configuration (Must be first)
st.set_page_config(page_title="Tractor Telemetry OS", layout="wide", initial_sidebar_state="expanded")

# 2. Lightweight CSS for clean typography and hiding default Streamlit clutter
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sleek Title Header */
    .premium-header {
        background: linear-gradient(90deg, #1f2937 0%, #111827 100%);
        padding: 24px;
        border-radius: 12px;
        border-left: 6px solid #00cc96;
        margin-bottom: 24px;
    }
    .premium-header h1 { margin: 0; color: #ffffff; font-size: 26px; font-weight: 600; font-family: 'Segoe UI', sans-serif;}
    .premium-header p { margin: 4px 0 0 0; color: #9ca3af; font-size: 13px; letter-spacing: 1px; }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
    <div class="premium-header">
        <h1>🚜 R&D Diagnostics Command Center</h1>
        <p>LIVE TELEMETRY • EXPLAINABLE AI • FLEET COMMAND</p>
    </div>
""", unsafe_allow_html=True)

# 3. Fast Model Loading
@st.cache_resource
def load_model():
    return joblib.load('tractor_health_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ AI Core Offline: tractor_health_model.pkl not found.")
    st.stop()

# 4. Industry Diagnostic Engine
diagnostics = {
    0: {"dtc": "None", "msg": "All subsystems operating nominally.", "fix": "Continue standard field operations.", "color": "success"},
    1: {"dtc": "ERR-HYD-001", "msg": "Hydraulic Pressure Loss + High Load", "fix": "Inspect pump seals. Replace fluid filter.", "color": "error"},
    2: {"dtc": "ERR-ENG-002", "msg": "Engine Thermal Overload Detected", "fix": "SHUTDOWN. Clean radiator fins. Check water pump.", "color": "error"},
    3: {"dtc": "ERR-TRN-003", "msg": "Transmission Slippage / Thermal Overload", "fix": "Recalibrate clutch packs. Check fluid viscosity.", "color": "warning"},
    4: {"dtc": "ERR-ELE-004", "msg": "System Voltage Drop / Alternator Fault", "fix": "Test alternator output. Inspect battery terminals.", "color": "error"},
    5: {"dtc": "ERR-PTO-005", "msg": "PTO RPM Drop under High Load", "fix": "Reduce implement speed. Inspect PTO shear pin.", "color": "warning"}
}

# 5. Sidebar - UX Optimized Controls
with st.sidebar:
    st.markdown("### 📡 Telemetry Link")
    streaming = st.toggle("🟢 Enable Live Data Streaming", value=False)
    st.divider()

    if 'sensors' not in st.session_state:
        st.session_state.sensors = { 'rpm': 1800, 'load': 45, 'temp': 85, 'pressure': 200, 'slip': 10, 'trans_temp': 75, 'battery': 13.8, 'pto': 540 }

    if streaming:
        st.session_state.sensors['rpm'] = int(np.clip(st.session_state.sensors['rpm'] + np.random.normal(0, 15), 800, 2500))
        st.session_state.sensors['temp'] = np.clip(st.session_state.sensors['temp'] + np.random.normal(0, 0.3), 70, 125)
        st.session_state.sensors['pressure'] = np.clip(st.session_state.sensors['pressure'] + np.random.normal(0, 1.5), 100, 250)
        time.sleep(0.3) 

    st.markdown("#### Engine & Powertrain")
    rpm = st.slider("Engine RPM", 800, 2500, st.session_state.sensors['rpm'])
    load = st.slider("Engine Load (%)", 0, 100, st.session_state.sensors['load'])
    temp = st.slider("Coolant Temp (°C)", 70, 125, int(st.session_state.sensors['temp']))
    trans_temp = st.slider("Transmission Temp (°C)", 50, 130, st.session_state.sensors['trans_temp'])

    st.markdown("#### Hydraulics & Implements")
    pressure = st.slider("Hydraulic Pressure (bar)", 100, 250, int(st.session_state.sensors['pressure']))
    pto = st.slider("PTO Speed (RPM)", 0, 600, st.session_state.sensors['pto'])
    slip = st.slider("Wheel Slip (%)", 0, 40, st.session_state.sensors['slip'])
    
    st.markdown("#### Electrical")
    battery = st.slider("Battery Voltage (V)", 10.0, 15.0, round(st.session_state.sensors['battery'], 1))

input_df = pd.DataFrame({'Engine_RPM': [rpm], 'Engine_Load_pct': [load], 'Coolant_Temp_C': [temp], 'Hydraulic_Pressure_bar': [pressure], 'Wheel_Slip_pct': [slip], 'Transmission_Temp_C': [trans_temp], 'Battery_Voltage_V': [battery], 'PTO_Speed_RPM': [pto]})

# 6. AI Inference
prediction = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]
confidence = max(probabilities) * 100
importances = model.feature_importances_
diag = diagnostics[prediction]

# 7. Main UI Grid Layout
tab_twin, tab_fleet = st.tabs(["⚡ Live Diagnostics", "🌍 Fleet Command"])

with tab_twin:
    # Use native Streamlit containers for that premium "card" look
    with st.container(border=True):
        col_alert, col_xai = st.columns([1.3, 1])
        
        with col_alert:
            st.markdown("### System Health Status")
            if prediction == 0:
                st.success(f"**✔️ ALL SYSTEMS NOMINAL** | AI Confidence: {confidence:.1f}%\n\n**Status:** {diag['msg']}")
            elif diag['color'] == "error":
                st.error(f"**⚠️ CRITICAL DTC: {diag['dtc']}** | AI Confidence: {confidence:.1f}%\n\n**Issue:** {diag['msg']}\n\n**🔧 Fix:** {diag['fix']}")
            else:
                st.warning(f"**⚠️ WARNING DTC: {diag['dtc']}** | AI Confidence: {confidence:.1f}%\n\n**Issue:** {diag['msg']}\n\n**🔧 Fix:** {diag['fix']}")
                
            m1, m2, m3 = st.columns(3)
            m1.metric("Engine Load", f"{load}%", "High" if load > 80 else "Normal", delta_color="inverse")
            m2.metric("Battery", f"{battery}V", "Low" if battery < 12.0 else "Normal", delta_color="inverse")
            m3.metric("PTO Speed", f"{pto} RPM", "Low" if pto < 500 else "Normal", delta_color="inverse")

        with col_xai:
            st.markdown("### AI Decision Drivers")
            fig_xai = px.bar(x=importances, y=input_df.columns, orientation='h', template="plotly_dark")
            fig_xai.update_layout(
                margin=dict(l=0, r=0, t=10, b=0),
                height=200,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="", yaxis_title="",
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(tickfont=dict(size=11, color="#e5e7eb"))
            )
            fig_xai.update_traces(marker_color='#00cc96', marker_line_color='#047857', marker_line_width=1.5, opacity=0.9)
            st.plotly_chart(fig_xai, use_container_width=True, config={'displayModeBar': False})

    # Bottom Row Gauges in a distinct container
    with st.container(border=True):
        st.markdown("### Live ISOBUS Telemetry")
        def sleek_gauge(val, title, min_v, max_v, color):
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=val, title={'text': title, 'font': {'size': 14, 'color': '#e5e7eb'}},
                number={'font': {'color': color}},
                gauge={
                    'axis': {'range': [min_v, max_v], 'tickwidth': 1, 'tickcolor': "#4b5563"},
                    'bar': {'color': color, 'thickness': 0.8},
                    'bgcolor': "#1f2937",
                    'borderwidth': 0
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=200, margin=dict(l=15, r=15, t=30, b=10))
            return fig

        g1, g2, g3, g4 = st.columns(4)
        with g1: st.plotly_chart(sleek_gauge(rpm, "Engine (RPM)", 800, 2500, "#3b82f6"), use_container_width=True, config={'displayModeBar': False})
        with g2: st.plotly_chart(sleek_gauge(temp, "Coolant (°C)", 70, 125, "#ef4444"), use_container_width=True, config={'displayModeBar': False})
        with g3: st.plotly_chart(sleek_gauge(pressure, "Hydraulic (bar)", 100, 250, "#10b981"), use_container_width=True, config={'displayModeBar': False})
        with g4: st.plotly_chart(sleek_gauge(slip, "Wheel Slip (%)", 0, 40, "#f59e0b"), use_container_width=True, config={'displayModeBar': False})

with tab_fleet:
    with st.container(border=True):
        st.markdown("### Global Fleet Status")
        k1, k2, k3 = st.columns(3)
        k1.metric("Total Online Units", "1,402", "+12 today")
        k2.metric("Critical Alerts", "3", "-2 from last hour", delta_color="inverse")
        k3.metric("Preventative Maintenance Scheduled", "41", "+5 today")
        
        st.divider()
        fleet_data = pd.DataFrame({
            "Unit ID": ["TRX-9901", "TRX-4421", "TRX-8832", "TRX-1092"],
            "Location": ["Agra Zone A", "Jaipur Sector 4", "Mohali Hub", "Greater Noida"],
            "Engine Hrs": [1450, 890, 3200, 2150],
            "Health Score": ["64% ⚠️", "82% ⚠️", "99% ✔️", "95% ✔️"],
            "Active DTC": ["ERR-ENG-002", "ERR-HYD-001", "None", "None"]
        })
        st.dataframe(fleet_data, use_container_width=True, hide_index=True)

if streaming:
    st.rerun()