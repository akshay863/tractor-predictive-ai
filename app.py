import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# 1. Configure the web page
st.set_page_config(page_title="Tractor Telemetry Dashboard", layout="wide")
st.title("🚜 Real-Time Tractor Telemetry & Predictive Maintenance")

# 2. Load the trained AI model from Step 3
@st.cache_resource
def load_model():
    return joblib.load('tractor_health_model.pkl')

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file not found. Please run predictive_model.py first.")
    st.stop()

# 3. Create the Sidebar for Live Inputs
st.sidebar.header("Live Sensor Data Input")
st.sidebar.markdown("Adjust the sliders below to simulate live CAN bus data. Watch how the AI analyzes the combination of metrics to predict machine health.")

rpm = st.sidebar.slider("Engine RPM", 1000, 2500, 2000)
temp = st.sidebar.slider("Coolant Temp (°C)", 70, 110, 85)
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
col1, col2 = st.columns(2)

with col1:
    st.subheader("System Diagnostic Status")
    st.write("The Machine Learning model is analyzing the live telemetry feed...")
    
    # Trigger visual alerts based on the AI's output
    if prediction == 0:
        st.success("✅ SYSTEM HEALTHY: All parameters within optimal range. No degradation detected.")
    else:
        st.error("⚠️ WARNING: Impending Hydraulic Pump Failure Detected! Schedule maintenance immediately.")

with col2:
    # Build a professional gauge chart for the hydraulic pressure
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pressure,
        title = {'text': "Live Hydraulic Pressure (bar)"},
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
    st.plotly_chart(fig, use_container_width=True)
    
st.markdown("---")
st.write("### How this R&D System Works:")
st.write("This dashboard utilizes a **Random Forest Classifier** trained on synthetic agricultural telemetry. Rather than relying on static mechanical thresholds, the algorithm analyzes the complex relationships between engine load, thermal dynamics, and fluid pressure to predict component degradation before catastrophic failure occurs in the field.")