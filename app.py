import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import time
import numpy as np

st.set_page_config(page_title="High-Dim Telemetry OS", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .premium-header {
        background: linear-gradient(90deg, #1f2937 0%, #111827 100%);
        padding: 24px; border-radius: 12px; border-left: 6px solid #eab308; margin-bottom: 24px;
    }
    .premium-header h1 { margin: 0; color: #ffffff; font-size: 26px; font-weight: 600; font-family: 'Segoe UI', sans-serif;}
    .premium-header p { margin: 4px 0 0 0; color: #9ca3af; font-size: 13px; letter-spacing: 1px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="premium-header">
        <h1>🚜 100-Channel R&D Diagnostics Command</h1>
        <p>PROCESSING 100 SIMULTANEOUS SENSORS • DYNAMIC CONTEXTUAL GENERATION • HIGH-DIMENSIONAL XAI</p>
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

# --- Sidebar Inputs ---
with st.sidebar:
    st.markdown("### 📡 Macro Control Hub")
    streaming = st.toggle("🟢 Auto-Streaming", value=False)
    st.divider()

    if 'sensors' not in st.session_state:
        st.session_state.sensors = { 'rpm': 1800, 'load': 45, 'temp': 85, 'pressure': 200, 'slip': 10, 'trans_temp': 75, 'battery': 13.8, 'pto': 540 }

    if streaming:
        st.session_state.sensors['rpm'] = int(np.clip(st.session_state.sensors['rpm'] + np.random.normal(0, 15), 800, 2500))
        st.session_state.sensors['pressure'] = np.clip(st.session_state.sensors['pressure'] + np.random.normal(0, 1.5), 100, 250)
        time.sleep(0.3) 

    rpm = st.slider("Engine RPM", 800, 2500, st.session_state.sensors['rpm'])
    load = st.slider("Engine Load (%)", 0, 100, st.session_state.sensors['load'])
    temp = st.slider("Coolant Temp (°C)", 70, 125, int(st.session_state.sensors['temp']))
    trans_temp = st.slider("Transmission Temp (°C)", 50, 130, st.session_state.sensors['trans_temp'])
    pressure = st.slider("Hydraulic Pressure (bar)", 100, 250, int(st.session_state.sensors['pressure']))
    pto = st.slider("PTO Speed (RPM)", 0, 600, st.session_state.sensors['pto'])
    slip = st.slider("Wheel Slip (%)", 0, 40, st.session_state.sensors['slip'])
    battery = st.slider("Battery Voltage (V)", 10.0, 15.0, round(st.session_state.sensors['battery'], 1))

# --- Build the 100-Feature Dictionary dynamically ---
input_dict = {
    'Engine_RPM': rpm, 'Engine_Load_pct': load, 'Coolant_Temp_C': temp,
    'Hydraulic_Pressure_bar': pressure, 'Wheel_Slip_pct': slip,
    'Transmission_Temp_C': trans_temp, 'Battery_Voltage_V': battery, 'PTO_Speed_RPM': pto
}

np.random.seed(int(time.time())) 
for i in range(1, 24): input_dict[f'Engine_Micro_Vib_{i}_Hz'] = rpm * np.random.uniform(0.01, 0.05)
for i in range(1, 24): input_dict[f'Hyd_Valve_Pressure_{i}_bar'] = pressure * np.random.uniform(0.8, 1.2)
for i in range(1, 24): input_dict[f'Trans_Gear_Temp_{i}_C'] = trans_temp * np.random.uniform(0.9, 1.1)
for i in range(1, 24): input_dict[f'CAN_Node_Volt_{i}_V'] = battery * np.random.uniform(0.95, 1.05)

input_df = pd.DataFrame([input_dict])

# --- AI Inference ---
prediction = model.predict(input_df)[0]
confidence = max(model.predict_proba(input_df)[0]) * 100
importances = model.feature_importances_

# --- DYNAMIC SENTENCE GENERATOR ---
def generate_dynamic_status(p_rpm, p_load, p_temp, p_press, p_trans, p_slip, p_batt, p_pto, dtc):
    conditions = []
    
    # Analyze the specific combination of current inputs
    if p_rpm > 2200: conditions.append(f"overspeeding at {p_rpm} RPM")
    elif p_rpm < 1000: conditions.append(f"idling low at {p_rpm} RPM")
    
    if p_load > 85: conditions.append(f"sustaining severe mechanical load ({p_load}%)")
    
    if p_temp > 105: conditions.append(f"experiencing critical thermal stress ({p_temp}°C)")
    
    if p_press < 140: conditions.append(f"losing hydraulic flow ({p_press} bar)")
    elif p_press > 230: conditions.append(f"pushing extreme hydraulic pressure ({p_press} bar)")
    
    if p_trans > 110: conditions.append(f"overheating the transmission clutch packs ({p_trans}°C)")
    
    if p_slip > 25: conditions.append(f"suffering heavy traction loss ({p_slip}% slip)")
    
    if p_batt < 12.0: conditions.append("detecting dangerous voltage drops")
    
    if p_pto < 500 and p_load > 60: conditions.append("bogging down the PTO shaft under draft load")

    # Construct the final highly-customized sentence
    if dtc == 0:
        if not conditions:
            return "✅ **STATUS:** The vehicle is operating flawlessly within the nominal baseline. No stress signatures detected across the 100-node network."
        else:
            return f"⚠️ **STATUS:** The core AI confirms no critical failures yet, however, the system is currently " + ", and ".join(conditions) + "."
    
    # If a failure is happening, combine the AI diagnosis with the live physics
    dtc_names = {1: "HYD-001 (Hydraulic)", 2: "ENG-002 (Engine)", 3: "TRN-003 (Transmission)", 4: "ELE-004 (Electrical)", 5: "PTO-005 (PTO)"}
    fault = dtc_names.get(dtc, "Unknown")
    
    base_msg = f"❌ **CRITICAL AI FAULT DETECTED [{fault}]:** The predictive model has identified a failure signature because the machine is "
    return base_msg + ", and ".join(conditions) + "."

# Generate the custom sentence for this exact millisecond
custom_status_message = generate_dynamic_status(rpm, load, temp, pressure, trans_temp, slip, battery, pto, prediction)

# --- UI Tabs ---
tab_diag, tab_raw = st.tabs(["⚡ Core Diagnostics", "🔢 Raw 100-Channel CAN Feed"])

with tab_diag:
    with st.container(border=True):
        col_alert, col_xai = st.columns([1.3, 1])
        with col_alert:
            st.markdown("### 100-Node Health Status")
            
            # Print the dynamically generated sentence
            if prediction == 0:
                st.info(custom_status_message)
            else:
                st.error(custom_status_message)

        with col_xai:
            st.markdown("### Top 10 High-Impact Features")
            top_indices = np.argsort(importances)[-10:]
            top_features = input_df.columns[top_indices]
            top_importances = importances[top_indices]
            
            fig_xai = px.bar(x=top_importances, y=top_features, orientation='h')
            fig_xai.update_layout(
                margin=dict(l=0, r=0, t=10, b=0), height=200,
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(tickfont=dict(size=10))
            )
            fig_xai.update_traces(marker_color='#eab308')
            st.plotly_chart(fig_xai, use_container_width=True, config={'displayModeBar': False})

    with st.container(border=True):
        st.markdown("### Primary Subsystem Gauges")
        def sleek_gauge(val, title, min_v, max_v, color):
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=val, title={'text': title, 'font': {'size': 14}},
                number={'font': {'color': color}},
                gauge={'axis': {'range': [min_v, max_v]}, 'bar': {'color': color, 'thickness': 0.8}, 'bgcolor': "rgba(0,0,0,0.05)"}
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=200, margin=dict(l=15, r=15, t=30, b=10))
            return fig

        g1, g2, g3, g4 = st.columns(4)
        with g1: st.plotly_chart(sleek_gauge(rpm, "Engine (RPM)", 800, 2500, "#3b82f6"), use_container_width=True, config={'displayModeBar': False})
        with g2: st.plotly_chart(sleek_gauge(temp, "Coolant (°C)", 70, 125, "#ef4444"), use_container_width=True, config={'displayModeBar': False})
        with g3: st.plotly_chart(sleek_gauge(pressure, "Hydraulic (bar)", 100, 250, "#10b981"), use_container_width=True, config={'displayModeBar': False})
        with g4: st.plotly_chart(sleek_gauge(battery, "Electrical (V)", 10, 15, "#f59e0b"), use_container_width=True, config={'displayModeBar': False})

with tab_raw:
    st.markdown("### SAE J1939 Live CAN Bus Datastream (100 Nodes)")
    st.dataframe(input_df.T.rename(columns={0: "Live Sensor Value"}), height=500, use_container_width=True)

if streaming:
    st.rerun()