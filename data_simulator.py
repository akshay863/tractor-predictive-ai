import pandas as pd
import numpy as np

print("Generating 15-Primary Sensor Telemetry Data...")
np.random.seed(42)
n_samples = 3000

# 1. Generate 15 Macro Sensors
data = {
    'Engine_RPM': np.random.normal(1800, 200, n_samples),
    'Engine_Load_pct': np.random.normal(45, 10, n_samples),
    'Coolant_Temp_C': np.random.normal(85, 5, n_samples),
    'Exhaust_Gas_Temp_C': np.random.normal(400, 50, n_samples), # EGT
    'Fuel_Rail_Pressure_bar': np.random.normal(1500, 100, n_samples),
    'Intake_Air_Temp_C': np.random.normal(40, 5, n_samples),
    'Transmission_Temp_C': np.random.normal(75, 5, n_samples),
    'Hydraulic_Pressure_bar': np.random.normal(200, 10, n_samples),
    'Hydraulic_Flow_Lpm': np.random.normal(80, 5, n_samples),
    'PTO_Speed_RPM': np.random.normal(540, 10, n_samples),
    'Draft_Load_kN': np.random.normal(15, 5, n_samples), # Pulling force
    'Wheel_Slip_pct': np.random.normal(10, 2, n_samples),
    'Radar_Speed_kmh': np.random.normal(12, 2, n_samples),
    'Steering_Angle_deg': np.random.normal(0, 5, n_samples),
    'Battery_Voltage_V': np.random.normal(13.8, 0.2, n_samples)
}

# 2. Generate 85 Micro-Sensors (Total = 100 features)
for i in range(1, 22): data[f'Engine_Micro_{i}'] = data['Engine_RPM'] * np.random.uniform(0.01, 0.05)
for i in range(1, 22): data[f'Hyd_Micro_{i}'] = data['Hydraulic_Pressure_bar'] * np.random.uniform(0.8, 1.2)
for i in range(1, 22): data[f'Trans_Micro_{i}'] = data['Transmission_Temp_C'] * np.random.uniform(0.9, 1.1)
for i in range(1, 23): data[f'Elec_Micro_{i}'] = data['Battery_Voltage_V'] * np.random.uniform(0.95, 1.05)

data['Failure_Code'] = 0
df = pd.DataFrame(data)

# Inject Failures (Teaching the AI the new sensors)
df.loc[1800:1950, ['Hydraulic_Pressure_bar', 'Hydraulic_Flow_Lpm']] = [120, 45]
df.loc[1800:1950, 'Failure_Code'] = 1 # HYD

df.loc[1951:2100, ['Coolant_Temp_C', 'Exhaust_Gas_Temp_C']] = [118, 680]
df.loc[1951:2100, 'Failure_Code'] = 2 # ENG

df.loc[2101:2300, ['Transmission_Temp_C', 'Wheel_Slip_pct', 'Draft_Load_kN']] = [115, 30, 45]
df.loc[2101:2300, 'Failure_Code'] = 3 # TRN

df.loc[2301:2499, ['Battery_Voltage_V', 'Fuel_Rail_Pressure_bar']] = [11.0, 900]
df.loc[2301:2499, 'Failure_Code'] = 4 # ELE

df.to_csv('industry_tractor_telemetry.csv', index=False)
print("Success: 15-Primary Sensor Dataset built.")