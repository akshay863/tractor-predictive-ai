import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset...")
df = pd.read_csv('industry_tractor_telemetry.csv')
X = df.drop('Failure_Code', axis=1)
y = df['Failure_Code']

# Train Supervised Model (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1) 
rf_model.fit(X_train, y_train)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)) * 100:.2f}%")
joblib.dump(rf_model, 'tractor_health_model.pkl')

# Train Unsupervised Model (Isolation Forest for Anomaly Detection)
print("Training Isolation Forest (Anomaly Detection)...")
anomaly_model = IsolationForest(contamination=0.03, random_state=42, n_jobs=-1)
anomaly_model.fit(X) # Trains on all data to find outliers
joblib.dump(anomaly_model, 'anomaly_model.pkl')

print("Success: Both AI models saved successfully.")