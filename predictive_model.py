import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('industry_tractor_telemetry.csv')
X = df.drop('Failure_Code', axis=1)
y = df['Failure_Code']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1) 
model.fit(X_train, y_train)

print(f"15-Primary / 100-Total Node AI Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")
joblib.dump(model, 'tractor_health_model.pkl')