import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
data_path = os.path.join(BASE_DIR, "data", "placement_data.csv")
output_dir = os.path.join(BASE_DIR, "outputs")
model_path = os.path.join(output_dir, "model.pkl")

# Load dataset
df = pd.read_csv(data_path)

# Convert Yes/No → 1/0
df["placement"] = df["placement"].map({"Yes": 1, "No": 0})

# Features & target
X = df.drop("placement", axis=1)
y = df["placement"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Base model
base_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

# 🔥 Calibrated model
model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_prob)
print(f"\nROC-AUC Score: {roc:.3f}")

# Save model
os.makedirs(output_dir, exist_ok=True)
joblib.dump(model, model_path)

print(f"\nModel saved at: {model_path}")