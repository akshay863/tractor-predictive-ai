# 🚜 Agricultural Machinery Telemetry & Predictive Maintenance 

## Overview
This project is an R&D-focused software pipeline designed for modern agricultural machinery. It simulates CAN bus telemetry data (Engine RPM, Coolant Temperature, Hydraulic Pressure, and Wheel Slip) and utilizes a Machine Learning model to predict impending mechanical failures before they occur in the field. 

## The Engineering Problem
Modern farm fleets experience severe financial losses due to unexpected downtime. Traditional maintenance is scheduled based on static engine hours. This project introduces a dynamic, predictive approach using AI to monitor the complex relationships between thermal dynamics and fluid pressures to flag anomalies in real-time.

## System Architecture
1. **Data Simulation (`data_simulator.py`):** Generates synthetic, realistic tractor telemetry, injecting degradation patterns into hydraulic and thermal systems over time.
2. **Predictive AI Model (`predictive_model.py`):** Uses a `RandomForestClassifier` trained on the synthetic dataset to achieve high-accuracy fault detection. 
3. **Interactive Dashboard (`app.py`):** A Streamlit-based web interface acting as a Virtual Terminal, allowing users to input live sensor data and receive instantaneous system health diagnostics.

## Tech Stack
* **Language:** Python 3.9+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Frontend UI:** Streamlit, Plotly

## How to Run Locally
1. Clone this repository to your local machine.
2. Activate your virtual environment: `.\venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`
4. Generate the telemetry data: `python data_simulator.py`
5. Train the Machine Learning model: `python predictive_model.py`
6. Launch the dashboard: `streamlit run app.py`

## Future Research & Expansion
This architecture serves as a foundational model for integrating embedded ISOBUS terminal data with cloud-based fleet management analytics.