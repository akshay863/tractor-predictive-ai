import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Loading full-vehicle dataset...")
df = pd.read_csv('full_tractor_telemetry.csv')

# Split features (X) and target failure codes (y)
X = df.drop('Failure_Code', axis=1)
y = df['Failure_Code']

# 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Multi-Class Random Forest AI...")
# We use more estimators (trees) because the patterns are now much more complex
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Test the accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\n--- AI Training Complete ---")
print(f"Multi-Class Model Accuracy: {accuracy * 100:.2f}%\n")

# Save the new, smarter brain
joblib.dump(model, 'tractor_health_model.pkl')
print("Saved new upgraded AI as 'tractor_health_model.pkl'.")