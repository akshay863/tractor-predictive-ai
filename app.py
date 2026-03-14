import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time
import numpy as np

# 1. Page Configuration (Must be first)
st.set_page_config(page_title="Tractor Telemetry OS", layout="wide", initial_sidebar_state="expanded")

# 2. Inject Premium Custom CSS
st.markdown("""
    <style>
    /* Hide Streamlit default headers and footers for a clean app look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sleek Background and Typography */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Premium Header Styling */
    .premium-header {
        background: linear-gradient(90deg, #1f2937 0%, #111827 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #00cc96;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 25px;
    }
    .premium-header h1 { margin: 0; color: #ffffff; font-size: 28px; font-weight: 600; }
    .premium-header p { margin: 5px 0 0 0; color: #9ca3af; font-size: 14px; letter-spacing: 1px; }

    /* Custom Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1f2937;
        border: 1px solid #374151;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Smooth sliders */
    .stSlider > div > div > div > div { background-color: #00cc96 !important; }
    </style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
    <div class="premium-header">
        <h1>🚜 Precision R&D Diagnostics OS</h1>
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

# 5. Sidebar - Sleek Inputs & Streaming
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png", width=40)
    st.markdown("### 📡 Sensor Injection")
    
    streaming = st.toggle("🟢 Enable Live Data Streaming", value=False)
    st.markdown("---")

    if 'sensors' not in st.session_state:
        st.session_state.sensors = { 'rpm': 1800, 'load': 45, 'temp': 85, 'pressure': 200, 'slip': 10, 'trans_temp': 75, 'battery': 13.8, 'pto': 540 }

    if streaming:
        # Smooth, realistic sensor noise
        st.session_state.sensors['rpm'] = int(np.clip(st.session_state.sensors['rpm'] + np.random.normal(0, 15), 800, 2500))
        st.session_state.sensors['temp'] = np.clip(st.session_state.sensors['temp'] + np.random.normal(0, 0.3), 70, 125)
        st.session_state.sensors['pressure'] = np.clip(st.session_state.sensors['pressure'] + np.random.normal(0, 1.5), 100, 250)
        time.sleep(0.3) # Faster refresh rate for smooth UI

    # Compact sliders
    rpm = st.slider("Engine RPM", 800, 2500, st.session_state.sensors['rpm'], help="Engine Revolutions Per Minute")
    load = st.slider("Engine Load (%)", 0, 100, st.session_state.sensors['load'])
    temp = st.slider("Coolant Temp (°C)", 70, 125, int(st.session_state.sensors['temp']))
    pressure = st.slider("Hydraulic Pressure (bar)", 100, 250, int(st.session_state.sensors['pressure']))
    slip = st.slider("Wheel Slip (%)", 0, 40, st.session_state.sensors['slip'])
    trans_temp = st.slider("Transmission Temp (°C)", 50, 130, st.session_state.sensors['trans_temp'])
    battery = st.slider("Battery Voltage (V)", 10.0, 15.0, round(st.session_state.sensors['battery'], 1))
    pto = st.slider("PTO Speed (RPM)", 0, 600, st.session_state.sensors['pto'])

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
    # Top Row: Alert Box & XAI
    col_alert, col_xai = st.columns([1.2, 1])
    
    with col_alert:
        st.markdown("#### System Status")
        if prediction == 0:
            st.success(f"**✔️ ALL SYSTEMS NOMINAL** | Confidence: {confidence:.1f}%\n\n{diag['msg']}")
        elif diag['color'] == "error":
            st.error(f"**⚠️ CRITICAL DTC: {diag['dtc']}** | Confidence: {confidence:.1f}%\n\n**Issue:** {diag['msg']}\n\n**🔧 Fix:** {diag['fix']}")
        else:
            st.warning(f"**⚠️ WARNING DTC: {diag['dtc']}** | Confidence: {confidence:.1f}%\n\n**Issue:** {diag['msg']}\n\n**🔧 Fix:** {diag['fix']}")
            
        # Quick metrics under alert
        m1, m2, m3 = st.columns(3)
        m1.metric("Engine Load", f"{load}%", "High" if load > 80 else "Normal", delta_color="inverse")
        m2.metric("Battery", f"{battery}V", "Low" if battery < 12.0 else "Normal", delta_color="inverse")
        m3.metric("PTO Speed", f"{pto} RPM", "Low" if pto < 500 else "Normal", delta_color="inverse")

    with col_xai:
        st.markdown("#### AI Decision Drivers (XAI)")
        # Ultra-sleek Dark Mode Bar Chart
        fig_xai = px.bar(x=importances, y=input_df.columns, orientation='h', template="plotly_dark")
        fig_xai.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            height=220,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="", yaxis_title="",
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(tickfont=dict(size=10))
        )
        fig_xai.update_traces(marker_color='#00cc96')
        st.plotly_chart(fig_xai, use_container_width=True, config={'displayModeBar': False})

    st.markdown("---")
    
    # Bottom Row: Ultra-Fast Plotly Gauges
    st.markdown("#### Live ISOBUS Telemetry")
    def sleek_gauge(val, title, min_v, max_v, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=val, title={'text': title, 'font': {'size': 14, 'color': 'white'}},
            gauge={
                'axis': {'range': [min_v, max_v], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': color, 'thickness': 0.7},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2, 'bordercolor': "#374151"
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=200, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    g1, g2, g3, g4 = st.columns(4)
    with g1: st.plotly_chart(sleek_gauge(rpm, "Engine (RPM)", 800, 2500, "#3b82f6"), use_container_width=True, config={'displayModeBar': False})
    with g2: st.plotly_chart(sleek_gauge(temp, "Coolant (°C)", 70, 125, "#ef4444"), use_container_width=True, config={'displayModeBar': False})
    with g3: st.plotly_chart(sleek_gauge(pressure, "Hydraulic (bar)", 100, 250, "#10b981"), use_container_width=True, config={'displayModeBar': False})
    with g4: st.plotly_chart(sleek_gauge(slip, "Wheel Slip (%)", 0, 40, "#f59e0b"), use_container_width=True, config={'displayModeBar': False})

with tab_fleet:
    st.markdown("#### Global Fleet Status")
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Online Units", "1,402", "+12 today")
    k2.metric("Critical Alerts", "3", "-2 from last hour", delta_color="inverse")
    k3.metric("Preventative Maintenance Scheduled", "41", "+5 today")
    
    fleet_data = pd.DataFrame({
        "Unit ID": ["TRX-9901", "TRX-4421", "TRX-8832", "TRX-1092"],
        "Location": ["Agra Zone A", "Jaipur Sector 4", "Mohali Hub", "Greater Noida"],
        "Engine Hrs": [1450, 890, 3200, 2150],
        "Health Score": ["64% ⚠️", "82% ⚠️", "99% ✔️", "95% ✔️"],
        "Active DTC": ["ERR-ENG-002", "ERR-HYD-001", "None", "None"]
    })
    st.dataframe(fleet_data, use_container_width=True, hide_index=True)

# Smooth Streaming Loop
if streaming:
    st.rerun()