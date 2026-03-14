import pandas as pd
import numpy as np

print("Generating Tier-1 Industry Telemetry Data...")
np.random.seed(42)
n_samples = 3000

# 1. Normal Baseline Data
data = {
    'Engine_RPM': np.random.normal(1800, 200, n_samples),
    'Engine_Load_pct': np.random.normal(45, 10, n_samples),
    'Coolant_Temp_C': np.random.normal(85, 5, n_samples),
    'Hydraulic_Pressure_bar': np.random.normal(200, 10, n_samples),
    'Wheel_Slip_pct': np.random.normal(10, 2, n_samples),
    'Transmission_Temp_C': np.random.normal(75, 5, n_samples),
    'Battery_Voltage_V': np.random.normal(13.8, 0.2, n_samples),
    'PTO_Speed_RPM': np.random.normal(540, 10, n_samples),
    'Failure_Code': 0
}
df = pd.DataFrame(data)

# 2. Inject Failure 1: HYD-001 (Hydraulic Drop + High Engine Load)
df.loc[1800:1950, 'Hydraulic_Pressure_bar'] = np.random.normal(120, 15, 151)
df.loc[1800:1950, 'Engine_Load_pct'] = np.random.normal(85, 5, 151)
df.loc[1800:1950, 'Failure_Code'] = 1

# 3. Inject Failure 2: ENG-002 (Coolant Overheat + High RPM)
df.loc[1951:2100, 'Coolant_Temp_C'] = np.random.normal(118, 4, 150)
df.loc[1951:2100, 'Engine_RPM'] = np.random.normal(2300, 100, 150)
df.loc[1951:2100, 'Failure_Code'] = 2

# 4. Inject Failure 3: TRN-003 (Transmission Overheat + Slip)
df.loc[2101:2300, 'Transmission_Temp_C'] = np.random.normal(115, 5, 200)
df.loc[2101:2300, 'Wheel_Slip_pct'] = np.random.normal(30, 4, 200)
df.loc[2101:2300, 'Failure_Code'] = 3

# 5. Inject Failure 4: ELE-004 (Alternator/Battery Drop)
df.loc[2301:2499, 'Battery_Voltage_V'] = np.random.normal(11.0, 0.3, 199)
df.loc[2301:2499, 'Failure_Code'] = 4

# 6. Inject Failure 5: PTO-005 (PTO Drop + Engine Load Spike)
df.loc[2500:2700, 'PTO_Speed_RPM'] = np.random.normal(450, 20, 201)
df.loc[2500:2700, 'Engine_Load_pct'] = np.random.normal(95, 3, 201)
df.loc[2500:2700, 'Failure_Code'] = 5

df.to_csv('industry_tractor_telemetry.csv', index=False)
print("Success: Generated 'industry_tractor_telemetry.csv' with 8 sensors and 5 DTCs.")