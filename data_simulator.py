import pandas as pd
import numpy as np

print("Generating High-Dimensional 100-Sensor Telemetry Data...")
np.random.seed(42)
n_samples = 3000

# 1. Generate the 8 Macro Sensors
data = {
    'Engine_RPM': np.random.normal(1800, 200, n_samples),
    'Engine_Load_pct': np.random.normal(45, 10, n_samples),
    'Coolant_Temp_C': np.random.normal(85, 5, n_samples),
    'Hydraulic_Pressure_bar': np.random.normal(200, 10, n_samples),
    'Wheel_Slip_pct': np.random.normal(10, 2, n_samples),
    'Transmission_Temp_C': np.random.normal(75, 5, n_samples),
    'Battery_Voltage_V': np.random.normal(13.8, 0.2, n_samples),
    'PTO_Speed_RPM': np.random.normal(540, 10, n_samples),
}

# 2. Generate 92 Micro-Sensors to reach exactly 100 features
for i in range(1, 24): # 23 Engine localized sensors
    data[f'Engine_Micro_Vib_{i}_Hz'] = data['Engine_RPM'] * np.random.uniform(0.01, 0.05)
for i in range(1, 24): # 23 Hydraulic localized sensors
    data[f'Hyd_Valve_Pressure_{i}_bar'] = data['Hydraulic_Pressure_bar'] * np.random.uniform(0.8, 1.2)
for i in range(1, 24): # 23 Transmission localized sensors
    data[f'Trans_Gear_Temp_{i}_C'] = data['Transmission_Temp_C'] * np.random.uniform(0.9, 1.1)
for i in range(1, 24): # 23 Electrical localized sensors
    data[f'CAN_Node_Volt_{i}_V'] = data['Battery_Voltage_V'] * np.random.uniform(0.95, 1.05)

data['Failure_Code'] = 0
df = pd.DataFrame(data) # Total 101 columns (100 features + 1 target)

# 3. Inject Failures (The AI learns complex patterns across all 100 sensors)
# HYD-001
df.loc[1800:1950, 'Hydraulic_Pressure_bar'] = np.random.normal(120, 15, 151)
df.loc[1800:1950, 'Engine_Load_pct'] = np.random.normal(85, 5, 151)
df.loc[1800:1950, 'Failure_Code'] = 1

# ENG-002
df.loc[1951:2100, 'Coolant_Temp_C'] = np.random.normal(118, 4, 150)
df.loc[1951:2100, 'Engine_RPM'] = np.random.normal(2300, 100, 150)
df.loc[1951:2100, 'Failure_Code'] = 2

# TRN-003
df.loc[2101:2300, 'Transmission_Temp_C'] = np.random.normal(115, 5, 200)
df.loc[2101:2300, 'Wheel_Slip_pct'] = np.random.normal(30, 4, 200)
df.loc[2101:2300, 'Failure_Code'] = 3

# ELE-004
df.loc[2301:2499, 'Battery_Voltage_V'] = np.random.normal(11.0, 0.3, 199)
df.loc[2301:2499, 'Failure_Code'] = 4

df.to_csv('industry_tractor_telemetry.csv', index=False)
print(f"Success: Generated dataset with {df.shape[1]-1} sensors and 1 target variable.")