import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_predictive_model():
    print("🧠 Loading tractor telemetry data...")
    
    # 1. Load the data we generated in Step 2
    file_path = 'data/simulated_tractor_data.csv'
    if not os.path.exists(file_path):
        print("Error: Could not find the data file. Did you run data_simulator.py?")
        return

    df = pd.read_csv(file_path)

    # 2. Define our Features (X) and Target (y)
    # The 'Features' are the sensors the AI will monitor:
    X = df[['Engine_RPM', 'Coolant_Temp_C', 'Hydraulic_Pressure_bar', 'Wheel_Slip_pct']]
    
    # The 'Target' is what we want the AI to predict (0 = Healthy, 1 = Failing):
    y = df['Machine_Status']

    # 3. Split the data into Training and Testing sets
    # We give the AI 80% of the data to study, and hold back 20% to test its knowledge.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("⚙️ Training the Random Forest AI...")
    
    # 4. Initialize and train the Machine Learning model
    # A Random Forest works perfectly for mechanical diagnostics because it builds 
    # hundreds of "decision trees" to vote on whether the machine is breaking down.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Test the model's accuracy on the 20% of data it hasn't seen yet
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"✅ Model trained successfully!")
    print(f"🎯 AI Accuracy Rating: {accuracy * 100:.2f}%\n")
    print("Detailed Diagnostic Report:")
    print(classification_report(y_test, predictions))

    # 6. Save the trained 'brain' so our web dashboard can plug into it
    model_filename = 'tractor_health_model.pkl'
    joblib.dump(model, model_filename)
    print(f"💾 Model saved as '{model_filename}' in your project folder.")

if __name__ == "__main__":
    train_predictive_model()