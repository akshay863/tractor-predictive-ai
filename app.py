import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Service ADVISOR - Diagnostics", layout="wide")

# Ruggedized Industry Header
st.markdown("""
    <div style='background-color: #1E1E1E; padding: 15px; border-radius: 5px; color: white; border-left: 8px solid #00cc96;'>
        <h2 style='margin:0;'>🚜 Advanced Field Diagnostics Terminal</h2>
        <p style='margin:0; color: #a8a8a8;'>Link Status: <span style='color: #00cc96;'>ACTIVE</span> | Protocol: ISOBUS 11783 | AI Diagnostic Engine: ONLINE</p>
    </div>
    <br>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('tractor_health_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file not found.")
    st.stop()

st.sidebar.header("CAN Bus Simulation Interface")
st.sidebar.markdown("Inject sensor data to test AI Prescriptive Engine.")

rpm = st.sidebar.slider("Engine RPM", 800, 2500, 1800)
load = st.sidebar.slider("Engine Load (%)", 0, 100, 45)
temp = st.sidebar.slider("Coolant Temp (°C)", 70, 125, 85)
pressure = st.sidebar.slider("Hydraulic Pressure (bar)", 100, 250, 200)
slip = st.sidebar.slider("Wheel Slip (%)", 0, 40, 10)
trans_temp = st.sidebar.slider("Transmission Temp (°C)", 50, 130, 75)
battery = st.sidebar.slider("Battery Voltage (V)", 10.0, 15.0, 13.8)
pto = st.sidebar.slider("PTO Speed (RPM)", 0, 600, 540)

input_data = pd.DataFrame({
    'Engine_RPM': [rpm], 'Engine_Load_pct': [load], 'Coolant_Temp_C': [temp],
    'Hydraulic_Pressure_bar': [pressure], 'Wheel_Slip_pct': [slip],
    'Transmission_Temp_C': [trans_temp], 'Battery_Voltage_V': [battery],
    'PTO_Speed_RPM': [pto]
})

prediction = model.predict(input_data)[0]

# Industry Diagnostic Dictionary (The "Prescriptive" Part)
diagnostics = {
    0: {"status": "NOMINAL", "dtc": "None", "severity": "Normal", "msg": "All subsystems operating within defined parameters.", "fix": "Continue standard field operation. No action required."},
    1: {"status": "WARNING", "dtc": "ERR-HYD-001", "severity": "High", "msg": "Hydraulic Pressure Loss under Heavy Engine Load.", "fix": "1. Inspect main hydraulic pump seals for leaks.\n2. Replace hydraulic oil filter.\n3. Check implement quick-connects for fluid bypass."},
    2: {"status": "CRITICAL", "dtc": "ERR-ENG-002", "severity": "Severe", "msg": "Engine Thermal Overload / Coolant Spike.", "fix": "1. IMMEDIATE ENGINE SHUTDOWN.\n2. Clean radiator fins of field debris.\n3. Inspect water pump drive belt tension.\n4. Check coolant reservoir levels."},
    3: {"status": "WARNING", "dtc": "ERR-TRN-003", "severity": "High", "msg": "Transmission Slippage / Thermal Overload.", "fix": "1. Recalibrate clutch packs via diagnostic service menu.\n2. Check transmission fluid level and viscosity.\n3. Reduce implement draft load."},
    4: {"status": "CRITICAL", "dtc": "ERR-ELE-004", "severity": "Severe", "msg": "System Voltage Drop / Alternator Fault.", "fix": "1. Test alternator output using multimeter.\n2. Inspect battery terminals for severe corrosion.\n3. Check serpentine belt tension."},
    5: {"status": "WARNING", "dtc": "ERR-PTO-005", "severity": "Moderate", "msg": "PTO RPM Drop under High Engine Load.", "fix": "1. Reduce implement operating speed.\n2. Inspect PTO shaft shear pin.\n3. Verify PTO clutch engagement pressure."}
}

diag = diagnostics[prediction]

# Prescriptive Action Dashboard
st.subheader("Automated Diagnostic Output")
if prediction == 0:
    st.success(f"✔️ **SYSTEM STATUS:** {diag['status']} | {diag['msg']}")
else:
    st.error(f"⚠️ **DIAGNOSTIC TROUBLE CODE (DTC):** {diag['dtc']} | **SEVERITY:** {diag['severity']}")
    st.warning(f"**Issue Detected:** {diag['msg']}")
    st.info(f"**🔧 RECOMMENDED TECHNICIAN ACTION:**\n\n{diag['fix']}")

st.markdown("---")

# Digital Twin KPI Metrics
st.subheader("Subsystem Telemetry KPIs")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Engine Load", f"{load}%", "High Load" if load > 80 else "Nominal", delta_color="inverse")
col2.metric("Battery Voltage", f"{battery} V", "Low Voltage" if battery < 12.0 else "Nominal", delta_color="inverse")
col3.metric("PTO Speed", f"{pto} RPM", "Dropping" if pto < 500 else "Nominal")
col4.metric("Active DTCs", "0" if prediction == 0 else "1", "Fault Detected" if prediction != 0 else "Clear", delta_color="inverse")