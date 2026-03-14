import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 2500

print("Generating Full-Vehicle Digital Twin Data...")

# 1. Generate Baseline Normal Data (Healthy Tractor)
data = {
    'Engine_RPM': np.random.normal(1800, 200, n_samples),
    'Coolant_Temp_C': np.random.normal(85, 5, n_samples),
    'Hydraulic_Pressure_bar': np.random.normal(200, 10, n_samples),
    'Wheel_Slip_pct': np.random.normal(10, 2, n_samples),
    'Transmission_Temp_C': np.random.normal(75, 5, n_samples),
    'Battery_Voltage_V': np.random.normal(13.8, 0.2, n_samples),
    'Failure_Code': 0  # 0 = All Systems Healthy
}
df = pd.DataFrame(data)

# 2. Inject Failure 1: Hydraulic Pump Degradation (Code 1)
df.loc[1800:1950, 'Hydraulic_Pressure_bar'] = np.random.normal(130, 15, 151)
df.loc[1800:1950, 'Coolant_Temp_C'] = np.random.normal(105, 5, 151)
df.loc[1800:1950, 'Failure_Code'] = 1

# 3. Inject Failure 2: Engine Overheating (Code 2)
df.loc[1951:2100, 'Coolant_Temp_C'] = np.random.normal(118, 4, 150)
df.loc[1951:2100, 'Engine_RPM'] = np.random.normal(1200, 150, 150)
df.loc[1951:2100, 'Failure_Code'] = 2

# 4. Inject Failure 3: Transmission Slippage (Code 3)
df.loc[2101:2300, 'Transmission_Temp_C'] = np.random.normal(112, 6, 200)
df.loc[2101:2300, 'Wheel_Slip_pct'] = np.random.normal(28, 4, 200)
df.loc[2101:2300, 'Failure_Code'] = 3

# 5. Inject Failure 4: Alternator/Electrical Fault (Code 4)
df.loc[2301:2499, 'Battery_Voltage_V'] = np.random.normal(11.2, 0.4, 199)
df.loc[2301:2499, 'Failure_Code'] = 4

# Save the new comprehensive dataset
df.to_csv('full_tractor_telemetry.csv', index=False)
print("Success: Generated 'full_tractor_telemetry.csv' with 6 sensors and 4 failure modes.")