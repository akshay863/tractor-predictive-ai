import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# 1. Configure the web page
st.set_page_config(page_title="Full-Vehicle Digital Twin", layout="wide")
st.title("🚜 R&D Digital Twin: Full-Vehicle Predictive Maintenance")

# 2. Load the trained AI model
@st.cache_resource
def load_model():
    return joblib.load('tractor_health_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file not found. Please ensure tractor_health_model.pkl is uploaded.")
    st.stop()

# 3. Create the Sidebar for Live Inputs (6 Sensors now!)
st.sidebar.header("Live CAN Bus Telemetry")
st.sidebar.markdown("Simulate sensor data across all vehicle subsystems.")

rpm = st.sidebar.slider("Engine RPM", 800, 2500, 1800)
temp = st.sidebar.slider("Coolant Temp (°C)", 70, 125, 85)
pressure = st.sidebar.slider("Hydraulic Pressure (bar)", 100, 250, 200)
slip = st.sidebar.slider("Wheel Slip (%)", 0, 40, 10)
trans_temp = st.sidebar.slider("Transmission Temp (°C)", 50, 130, 75)
battery = st.sidebar.slider("Battery Voltage (V)", 10.0, 15.0, 13.8)

# 4. Format the input for the AI
input_data = pd.DataFrame({
    'Engine_RPM': [rpm],
    'Coolant_Temp_C': [temp],
    'Hydraulic_Pressure_bar': [pressure],
    'Wheel_Slip_pct': [slip],
    'Transmission_Temp_C': [trans_temp],
    'Battery_Voltage_V': [battery]
})

# 5. Make the live prediction
prediction = model.predict(input_data)[0]

# 6. Build the Diagnostic Output Panel
st.subheader("Central AI Diagnostic Unit")

if prediction == 0:
    st.success("✅ ALL SYSTEMS NOMINAL: The vehicle is operating within optimal parameters.")
elif prediction == 1:
    st.error("⚠️ FAULT CODE 1: Hydraulic Pump Degradation Detected! (Pressure Drop + Heat)")
elif prediction == 2:
    st.error("⚠️ FAULT CODE 2: Engine Overheating & Load Stress Detected!")
elif prediction == 3:
    st.error("⚠️ FAULT CODE 3: Transmission Slippage & Thermal Overload Detected!")
elif prediction == 4:
    st.error("⚠️ FAULT CODE 4: Alternator/Electrical System Failure Detected! (Voltage Drop)")

st.markdown("---")

# 7. Create an Organized Tabbed Interface for the Gauges
st.subheader("Live ISOBUS Terminal Feed")
tab1, tab2 = st.tabs(["Engine & Hydraulics", "Powertrain & Electrical"])

# Helper function to draw gauges quickly
def create_gauge(val, title, min_val, max_val, color):
    return go.Figure(go.Indicator(
        mode="gauge+number", value=val, title={'text': title},
        gauge={'axis': {'range': [min_val, max_val]}, 'bar': {'color': color}}
    ))

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1: st.plotly_chart(create_gauge(rpm, "Engine RPM", 800, 2500, "darkblue"), use_container_width=True)
    with col2: st.plotly_chart(create_gauge(temp, "Coolant Temp (°C)", 70, 125, "red"), use_container_width=True)
    with col3: st.plotly_chart(create_gauge(pressure, "Hydraulic Pressure (bar)", 100, 250, "green"), use_container_width=True)

with tab2:
    col4, col5, col6 = st.columns(3)
    with col4: st.plotly_chart(create_gauge(slip, "Wheel Slip (%)", 0, 40, "orange"), use_container_width=True)
    with col5: st.plotly_chart(create_gauge(trans_temp, "Transmission Temp (°C)", 50, 130, "purple"), use_container_width=True)
    with col6: st.plotly_chart(create_gauge(battery, "Battery Voltage (V)", 10.0, 15.0, "gold"), use_container_width=True)