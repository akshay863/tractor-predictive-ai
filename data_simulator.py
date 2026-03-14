import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_tractor_telemetry(num_records=1000, degradation_start=700):
    """
    Simulates CAN bus telemetry data for a modern agricultural tractor.
    Introduces a mechanical failure (hydraulic pump degradation) towards the end of the dataset.
    """
    print("🚜 Starting Tractor Telemetry Simulation...")
    
    # Initialize lists to store our data
    timestamps = []
    engine_rpm = []
    coolant_temp = []
    hydraulic_pressure = []
    wheel_slip = []
    failure_label = [] # 0 = Normal, 1 = Impending Failure

    # Start time for our simulation (simulating 1 data point per minute)
    current_time = datetime.now()

    for i in range(num_records):
        timestamps.append(current_time)
        
        # --- NORMAL OPERATING CONDITIONS ---
        if i < degradation_start:
            # Engine RPM cruising between 1800 and 2200 (typical PTO working range)
            rpm = np.random.normal(2000, 100)
            
            # Coolant temp steady around 85C to 90C
            temp = np.random.normal(88, 2)
            
            # Hydraulic pressure healthy at ~200 bar
            pressure = np.random.normal(200, 5)
            
            # Wheel slip fluctuating based on field conditions (5% - 15% is optimal)
            slip = np.random.normal(10, 2)
            
            status = 0 # Machine is healthy
            
        # --- DEGRADATION / ANOMALY CONDITIONS ---
        else:
            # Introduce a hydraulic pump failure and engine overheating
            # RPM stays roughly the same, maybe drops slightly due to load
            rpm = np.random.normal(1950, 150)
            
            # Coolant temp starts spiking as the engine strains
            temp = np.random.normal(95 + (i - degradation_start) * 0.05, 3) 
            
            # Hydraulic pressure drops as the pump fails
            pressure = np.random.normal(170 - (i - degradation_start) * 0.1, 8)
            
            # Slip might increase slightly if implement depth control fails
            slip = np.random.normal(14, 3)
            
            status = 1 # Impending failure flagged!

        # Append generated data
        engine_rpm.append(round(rpm, 1))
        coolant_temp.append(round(temp, 1))
        hydraulic_pressure.append(round(pressure, 1))
        wheel_slip.append(round(slip, 1))
        failure_label.append(status)
        
        # Advance time by 1 minute for the next reading
        current_time += timedelta(minutes=1)

    # Compile the lists into a structured pandas DataFrame
    data = {
        'Timestamp': timestamps,
        'Engine_RPM': engine_rpm,
        'Coolant_Temp_C': coolant_temp,
        'Hydraulic_Pressure_bar': hydraulic_pressure,
        'Wheel_Slip_pct': wheel_slip,
        'Machine_Status': failure_label
    }
    
    df = pd.DataFrame(data)
    
    # Ensure the 'data' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # Save to a CSV file
    file_path = 'data/simulated_tractor_data.csv'
    df.to_csv(file_path, index=False)
    print(f"✅ Simulation complete! {num_records} rows of data saved to {file_path}")

# Run the function if this script is executed
if __name__ == "__main__":
    generate_tractor_telemetry()