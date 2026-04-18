import joblib
import os
import pandas as pd

# Base path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "outputs", "model.pkl")

# Load model
model = joblib.load(model_path)

features = ["cgpa", "projects", "internships", "certifications", "backlogs"]

# Smooth probability (prevents extreme values)
def smooth_prob(p):
    return 0.05 + 0.9 * p


def suggest_improvements(student_dict):
    student_df = pd.DataFrame([student_dict])
    base_prob = model.predict_proba(student_df)[0][1]
    base_prob = smooth_prob(base_prob)

    suggestions = {}

    for feature in features:
        modified = student_dict.copy()

        # 🔥 Improvement logic
        if feature == "cgpa":
            modified[feature] += 0.5
        elif feature == "projects":
            modified[feature] += 1
        elif feature == "internships":
            modified[feature] += 1
        elif feature == "certifications":
            modified[feature] += 1
        elif feature == "backlogs":
            # 🔥 FIXED: stronger + dynamic reduction
            modified[feature] = max(0, modified[feature] - max(1, modified[feature] // 2))

        modified_df = pd.DataFrame([modified])
        new_prob = model.predict_proba(modified_df)[0][1]
        new_prob = smooth_prob(new_prob)

        suggestions[feature] = new_prob - base_prob

    return base_prob, suggestions


def display_result(student):
    prob, suggestions = suggest_improvements(student)

    print("\n==============================")
    print("Student:", student)
    print(f"Placement Probability: {prob:.2f}\n")

    if prob > 0.85:
        print("Strong profile")
    elif prob > 0.6:
        print("Close to placement")
    else:
        print("Needs improvement")

    total_impact = sum(abs(v) for v in suggestions.values())

    normalized = {
        k: (abs(v) / total_impact) * 100 if total_impact != 0 else 0
        for k, v in suggestions.items()
    }

    sorted_suggestions = sorted(normalized.items(), key=lambda x: x[1], reverse=True)

    print("\nTop Improvements:")
    for k, v in sorted_suggestions[:3]:
        print(f"{k}: {round(v, 1)}%")

    print("==============================")


# Test cases
if __name__ == "__main__":

    students = [
        {"cgpa": 5.5, "projects": 0, "internships": 0, "certifications": 0, "backlogs": 4},
        {"cgpa": 6.0, "projects": 1, "internships": 0, "certifications": 1, "backlogs": 3},
        {"cgpa": 7.0, "projects": 2, "internships": 1, "certifications": 1, "backlogs": 2},
        {"cgpa": 8.0, "projects": 3, "internships": 1, "certifications": 2, "backlogs": 1},
        {"cgpa": 9.0, "projects": 4, "internships": 2, "certifications": 3, "backlogs": 0},
    ]

    for s in students:
        display_result(s)