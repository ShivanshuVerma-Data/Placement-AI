import streamlit as st
import joblib
import os
import pandas as pd
import random

st.set_page_config(page_title="Placement AI", layout="wide")

# =========================
# UI STYLING
# =========================
st.markdown("""
<style>
.block-container {
    max-width: 1100px;
    margin: auto;
}
.title {
    text-align: center;
    font-size: 38px;
    font-weight: 600;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
}
.big {
    font-size: 70px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "outputs", "model.pkl"))

features = ["cgpa", "projects", "internships", "certifications", "backlogs"]

def smooth_prob(p):
    return 0.05 + 0.9 * p

# =========================
# MODEL LOGIC
# =========================
def suggest(student):
    df = pd.DataFrame([student])
    base = smooth_prob(model.predict_proba(df)[0][1])

    suggestions = {}

    for f in features:
        mod = student.copy()

        if f == "cgpa":
            mod[f] += 0.5
        elif f == "backlogs":
            if mod[f] > 6:
                mod[f] -= 3
            elif mod[f] > 3:
                mod[f] -= 2
            else:
                mod[f] -= 1
            mod[f] = max(0, mod[f])
        else:
            mod[f] += 1

        new = smooth_prob(model.predict_proba(pd.DataFrame([mod]))[0][1])
        suggestions[f] = new - base

    return base, suggestions

# =========================
# THRESHOLDS
# =========================
thresholds = {
    "cgpa": 8.5,
    "projects": 3,
    "internships": 2,
    "certifications": 3,
    "backlogs": 0
}

def is_improvable(feature, value):
    if feature == "backlogs":
        return value > thresholds[feature]
    return value < thresholds[feature]

# =========================
# SUMMARY
# =========================
def generate_summary(student, prob, suggestions):
    valid = [
        (f, imp) for f, imp in suggestions.items()
        if is_improvable(f, student[f])
    ]

    if not valid:
        return random.choice([
            "Your profile is already strong across all major factors. Focus on interviews and execution.",
            "You’re in a solid position overall. No major weaknesses detected.",
            "Your profile looks well-balanced. Focus on converting opportunities.",
            "You’ve optimized most key areas. Now focus on performance.",
            "No critical gaps detected. Keep refining your strengths."
        ])

    top_feature = sorted(valid, key=lambda x: abs(x[1]), reverse=True)[0][0]

    templates = [
        f"Your placement probability is around {int(prob*100)}%. The biggest opportunity lies in {top_feature}.",
        f"At {int(prob*100)}%, improving {top_feature} will have the most impact.",
        f"Your current standing is {int(prob*100)}%. Focus on {top_feature}.",
        f"With {int(prob*100)}%, your biggest gap is {top_feature}.",
        f"You’re at {int(prob*100)}%. {top_feature} is your next priority.",
        f"{int(prob*100)}% chance — {top_feature} is limiting you.",
        f"To move beyond {int(prob*100)}%, improve {top_feature}.",
        f"Your growth depends on improving {top_feature}.",
        f"Your current performance suggests improving {top_feature}.",
        f"{top_feature} is your biggest leverage point right now.",
        f"Improving {top_feature} will give best results.",
        f"You should focus on {top_feature} next.",
        f"{top_feature} is currently your weakest area.",
        f"Addressing {top_feature} will improve your profile.",
        f"{top_feature} is holding back your full potential.",
        f"You can improve most by fixing {top_feature}.",
        f"{top_feature} is the smartest next step.",
        f"Your profile depends heavily on {top_feature}.",
        f"{top_feature} needs attention right now.",
        f"Improving {top_feature} will boost outcomes significantly."
    ]

    return random.choice(templates)

# =========================
# ADVICE (FIXED PROPERLY)
# =========================
def generate_ai_advice(student, suggestions):
    valid = [
        (f, abs(imp)) for f, imp in suggestions.items()
        if is_improvable(f, student[f])
    ]

    if not valid:
        return ["You're already in a strong position. Focus on consistency and interviews."]

    sorted_items = sorted(valid, key=lambda x: x[1], reverse=True)

    advice = []

    for feature, _ in sorted_items:
        val = student[feature]

        if feature == "projects":
            options = [
                f"You have {val} projects — increase this.",
                f"{val} projects is low — build more.",
                f"Projects are your weak point.",
                f"Improve your project portfolio.",
                f"Add more real-world projects.",
                f"Your projects are insufficient.",
                f"Projects at {val} won’t stand out.",
                f"Focus on building projects.",
                f"Your portfolio lacks depth.",
                f"Increase project count.",
                f"Projects are limiting you.",
                f"Build stronger projects.",
                f"Projects need improvement.",
                f"More projects will help.",
                f"Strengthen your project work."
            ]

        elif feature == "internships":
            options = [
                f"You have {val} internships — improve this.",
                f"Gain more real-world experience.",
                f"Internships are low.",
                f"Increase industry exposure.",
                f"You need more internships.",
                f"Experience is limited.",
                f"Internships at {val} is weak.",
                f"Focus on internships.",
                f"Improve work experience.",
                f"Internships need growth.",
                f"Exposure is low.",
                f"Add internships.",
                f"Real-world experience is lacking.",
                f"Work experience is weak.",
                f"Internships are your gap."
            ]

        elif feature == "cgpa":
            options = [
                f"Your CGPA is {val:.1f} — improve it.",
                f"Academic score is average.",
                f"CGPA needs improvement.",
                f"Increase your CGPA.",
                f"Academic performance is moderate.",
                f"CGPA is not strong.",
                f"Improve academics.",
                f"CGPA is limiting opportunities.",
                f"Academic score needs boost.",
                f"Your CGPA is average.",
                f"Improve academic strength.",
                f"CGPA needs attention.",
                f"Academic performance can improve.",
                f"Boost your CGPA.",
                f"CGPA is a weak point."
            ]

        elif feature == "certifications":
            options = [
                f"Certifications are {val} — focus on skills.",
                f"Certifications have limited impact.",
                f"Focus on practical work.",
                f"Certifications won’t help much.",
                f"Real skills matter more.",
                f"Certificates are secondary.",
                f"Don’t rely on certifications.",
                f"Focus on execution.",
                f"Certifications aren’t enough.",
                f"Skills > certifications.",
                f"Improve practical knowledge.",
                f"Certifications are weak leverage.",
                f"Focus beyond certificates.",
                f"Real work matters more.",
                f"Certifications add little value."
            ]

        elif feature == "backlogs":
            options = [
                f"You have {val} backlogs — fix this.",
                f"Backlogs are a major issue.",
                f"Reduce your backlogs.",
                f"Backlogs limit your chances.",
                f"This is your biggest problem.",
                f"Backlogs are too high.",
                f"Clear backlogs urgently.",
                f"Backlogs hurt your profile.",
                f"This is a red flag.",
                f"Backlogs are critical.",
                f"Fix backlogs immediately.",
                f"Backlogs block opportunities.",
                f"This is your weakness.",
                f"Backlogs reduce credibility.",
                f"Backlogs need urgent attention."
            ]

        advice.append(random.choice(options))

    return advice[:3]

# =========================
# UI LAYOUT
# =========================
st.markdown('<div class="title">Placement AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered career insights</div>', unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### Your Profile")

    cgpa = st.slider("CGPA", 5.0, 10.0, 7.0, 0.1)
    projects = st.slider("Projects", 0, 6, 2)
    internships = st.slider("Internships", 0, 3, 1)
    certifications = st.slider("Certifications", 0, 5, 1)
    backlogs = st.slider("Backlogs", 0, 15, 2)

student = {
    "cgpa": cgpa,
    "projects": projects,
    "internships": internships,
    "certifications": certifications,
    "backlogs": backlogs
}

prob, suggestions = suggest(student)
summary = generate_summary(student, prob, suggestions)
advice = generate_ai_advice(student, suggestions)

with col2:
    st.markdown("### Placement Score")

    st.markdown(f'<div class="big">{int(prob*100)}%</div>', unsafe_allow_html=True)

    st.progress(prob)

    st.markdown(f"<div style='text-align:center; color:#cbd5f5'>{summary}</div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Key Insights")

    for line in advice:
        st.write(f"• {line}")

st.caption("Built with ML • AI-style insights")