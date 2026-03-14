import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset...")
df = pd.read_csv('industry_tractor_telemetry.csv')
X = df.drop('Failure_Code', axis=1)
y = df['Failure_Code']

# Feature Scaling (Crucial for high-dimensional anomaly detection)
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1) 
rf_model.fit(X_train, y_train)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)) * 100:.2f}%")
joblib.dump(rf_model, 'tractor_health_model.pkl')

print("Training Isolation Forest on normalized data...")
anomaly_model = IsolationForest(contamination=0.03, random_state=42, n_jobs=-1)
anomaly_model.fit(X_scaled) 
joblib.dump(anomaly_model, 'anomaly_model.pkl')

print("Success: Models and Scaler saved.")