import pandas as pd
import numpy as np
import os

# Base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "placement_data.csv")

# Rows
n = np.random.randint(18000, 20001)

data = pd.DataFrame({
    "cgpa": np.random.uniform(5, 10, n),
    "projects": np.random.randint(0, 6, n),
    "internships": np.random.randint(0, 3, n),
    "certifications": np.random.randint(0, 5, n),
    "backlogs": np.random.randint(0, 14, n),
})

# Balanced scoring
score = (
    data["cgpa"] * 0.4 +
    data["projects"] * 0.3 +
    data["internships"] * 0.4 +
    data["certifications"] * 0.2 -
    data["backlogs"] * 1.2
)

# Add noise
noise = np.random.normal(0, 1.2, n)
score = score + noise

# Sigmoid
prob = 1 / (1 + np.exp(-score))

# 🔥 Balanced threshold
data["placement"] = (prob > 0.5).astype(int)

# Convert to Yes/No
data["placement"] = data["placement"].map({1: "Yes", 0: "No"})

# Save
data.to_csv(data_path, index=False)

# Check distribution
print(f"Dataset created with {n} rows\n")
print(data["placement"].value_counts())
print(data.head())