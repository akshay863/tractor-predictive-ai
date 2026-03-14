import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# 1. Configure the web page
st.set_page_config(page_title="Tractor Telemetry Dashboard", layout="wide")
st.title("🚜 Real-Time Tractor Telemetry & Predictive Maintenance")

# 2. Load the trained AI model
@st.cache_resource
def load_model():
    return joblib.load('tractor_health_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file not found. Please upload tractor_health_model.pkl to your repository.")
    st.stop()

# 3. Create the Sidebar for Live Inputs
st.sidebar.header("Live Sensor Data Input")
st.sidebar.markdown("Adjust the sliders below to simulate live CAN bus data.")

rpm = st.sidebar.slider("Engine RPM", 1000, 2500, 2000)
temp = st.sidebar.slider("Coolant Temp (°C)", 70, 115, 88)
pressure = st.sidebar.slider("Hydraulic Pressure (bar)", 100, 250, 200)
slip = st.sidebar.slider("Wheel Slip (%)", 0, 30, 10)

# 4. Format the input for the AI
input_data = pd.DataFrame({
    'Engine_RPM': [rpm],
    'Coolant_Temp_C': [temp],
    'Hydraulic_Pressure_bar': [pressure],
    'Wheel_Slip_pct': [slip]
})

# 5. Make the live prediction
prediction = model.predict(input_data)[0]

# 6. Build the Visual Dashboard Layout
st.subheader("System Diagnostic Status")

# Trigger visual alerts based on the AI's output
if prediction == 0:
    st.success("✅ SYSTEM HEALTHY: All parameters within optimal range. No degradation detected.")
else:
    st.error("⚠️ CRITICAL WARNING: Impending Hydraulic Pump Failure Detected! Pressure dropping while thermal load increases.")

st.markdown("---")
st.subheader("Live ISOBUS Terminal Feed")

# Create 3 columns for our gauges
col1, col2, col3 = st.columns(3)

with col1:
    # Engine RPM Gauge
    fig_rpm = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = rpm,
        title = {'text': "Engine RPM"},
        gauge = {'axis': {'range': [1000, 2500]}, 'bar': {'color': "darkblue"}}
    ))
    st.plotly_chart(fig_rpm, use_container_width=True)

with col2:
    # Coolant Temp Gauge
    fig_temp = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = temp,
        title = {'text': "Coolant Temp (°C)"},
        gauge = {
            'axis': {'range': [70, 115]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [70, 95], 'color': "#00cc96"},   # Green
                {'range': [95, 105], 'color': "#FFA500"},  # Orange/Warning
                {'range': [105, 115], 'color': "#ff4b4b"}  # Red/Danger
            ]
        }
    ))
    st.plotly_chart(fig_temp, use_container_width=True)

with col3:
    # Hydraulic Pressure Gauge
    fig_pressure = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pressure,
        title = {'text': "Hydraulic Pressure (bar)"},
        gauge = {
            'axis': {'range': [100, 250]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [100, 160], 'color': "#ff4b4b"},  # Red (Too low)
                {'range': [160, 220], 'color': "#00cc96"},  # Green (Optimal)
                {'range': [220, 250], 'color': "#ff4b4b"}   # Red (Too high)
            ],
        }
    ))
    st.plotly_chart(fig_pressure, use_container_width=True)