import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("Loading Industry Dataset...")
df = pd.read_csv('industry_tractor_telemetry.csv')

X = df.drop('Failure_Code', axis=1)
y = df['Failure_Code']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Advanced Prescriptive AI...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print(f"Industry AI Accuracy: {accuracy_score(y_test, model.predict(X_test)) * 100:.2f}%")
joblib.dump(model, 'tractor_health_model.pkl')
print("Saved Master AI as 'tractor_health_model.pkl'.")