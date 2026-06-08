import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from groq import Groq
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
import io
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="AI Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { 
        background: linear-gradient(135deg, #0f1117 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    h1 { 
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 2.5em !important;
        font-weight: 700 !important;
    }
    p { color: #888; text-align: center; }
    .result-box {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        text-align: center;
        margin: 10px 0;
        animation: fadeIn 0.5s ease;
    }
    .advice-box {
        background: linear-gradient(135deg, #1e2e1e, #2a3e2a);
        border-radius: 16px;
        padding: 25px;
        border-left: 4px solid #a6e3a1;
        margin: 10px 0;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .stButton button {
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎓 AI Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:16px'>Enter your details — AI predicts your performance and gives study advice</p>", unsafe_allow_html=True)
st.divider()

api_key = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=api_key)

# Generate synthetic training data
np.random.seed(42)
n = 500

study_hours = np.random.uniform(1, 10, n)
attendance = np.random.uniform(50, 100, n)
prev_marks = np.random.uniform(40, 100, n)
sleep_hours = np.random.uniform(4, 10, n)
assignments = np.random.uniform(50, 100, n)

# Final marks formula with some noise
final_marks = (
    study_hours * 4 +
    attendance * 0.3 +
    prev_marks * 0.4 +
    sleep_hours * 1.5 +
    assignments * 0.2 +
    np.random.normal(0, 3, n)
).clip(0, 100)

# Create DataFrame
df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "prev_marks": prev_marks,
    "sleep_hours": sleep_hours,
    "assignments": assignments,
    "final_marks": final_marks
})

# Train model
X = df[["study_hours", "attendance", "prev_marks", "sleep_hours", "assignments"]]
y = df["final_marks"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
mae = mean_absolute_error(y_test, model.predict(X_test))

def get_grade(marks):
    if marks >= 90: return "A+", "#00d4ff"
    elif marks >= 80: return "A", "#a6e3a1"
    elif marks >= 70: return "B", "#fab387"
    elif marks >= 60: return "C", "#f9e2af"
    else: return "D", "#f38ba8"

def generate_pdf(name, predicted, grade, advice, inputs):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                           rightMargin=20*mm, leftMargin=20*mm,
                           topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                fontSize=18, textColor=colors.HexColor('#00d4ff'),
                                spaceAfter=10)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                               fontSize=10, spaceAfter=6, leading=14)
    story = []
    story.append(Paragraph(f"Student Performance Report: {name}", title_style))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(f"Predicted Score: {predicted:.1f}/100 — Grade: {grade}", body_style))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("Input Details:", title_style))
    for key, val in inputs.items():
        story.append(Paragraph(f"• {key}: {val}", body_style))
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("AI Study Advice:", title_style))
    for line in advice.split('\n'):
        if line.strip():
            story.append(Paragraph(line.strip(), body_style))
    doc.build(story)
    buffer.seek(0)
    return buffer

# Input form
st.markdown("### 📝 Enter Your Details")
name = st.text_input("Your Name:", placeholder="Rohit Savan")

col1, col2 = st.columns(2)
with col1:
    study_h = st.slider("📚 Daily Study Hours:", 1.0, 10.0, 5.0, 0.5)
    attendance_p = st.slider("🏫 Attendance %:", 50, 100, 75)
    prev_m = st.slider("📊 Previous Marks:", 40, 100, 70)
with col2:
    sleep_h = st.slider("😴 Sleep Hours:", 4.0, 10.0, 7.0, 0.5)
    assignment_p = st.slider("📋 Assignment Completion %:", 50, 100, 80)

if st.button("🎯 Predict My Performance", use_container_width=True):
    if name:
        input_data = pd.DataFrame({
            "study_hours": [study_h],
            "attendance": [attendance_p],
            "prev_marks": [prev_m],
            "sleep_hours": [sleep_h],
            "assignments": [assignment_p]
        })

        predicted = model.predict(input_data)[0]
        predicted = np.clip(predicted, 0, 100)
        grade, color = get_grade(predicted)

        st.markdown(f"""
        <div class="result-box">
            <p style="color:#00d4ff; font-size:14px; font-weight:600; margin:0">PREDICTED SCORE</p>
            <p style="color:{color}; font-size:64px; font-weight:800; margin:8px 0">{predicted:.1f}</p>
            <p style="color:{color}; font-size:24px; font-weight:600; margin:0">Grade: {grade}</p>
        </div>
        """, unsafe_allow_html=True)

        # Feature importance chart
        st.markdown("### 📊 What affects your score most?")
        features = ["Study Hours", "Attendance", "Prev Marks", "Sleep", "Assignments"]
        importance = model.feature_importances_
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor('#1e1e2e')
        ax.set_facecolor('#1e1e2e')
        bars = ax.barh(features, importance, color='#00d4ff', alpha=0.8)
        ax.set_title("Feature Importance", color='white')
        ax.tick_params(colors='white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # AI advice
        with st.spinner("Getting personalized study advice..."):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert academic coach. Give specific, actionable study advice."},
                    {"role": "user", "content": f"""
Student: {name}
Predicted Score: {predicted:.1f}/100 (Grade: {grade})
Study hours/day: {study_h}
Attendance: {attendance_p}%
Previous marks: {prev_m}
Sleep hours: {sleep_h}
Assignment completion: {assignment_p}%

Give 5 specific, personalized tips to improve their performance.
Be encouraging but honest. Keep it concise.
"""}
                ]
            )
            advice = response.choices[0].message.content

        st.markdown("### 💡 Personalized Study Advice")
        st.markdown(f"""
        <div class="advice-box">
            <p style="color:#a6e3a1; font-size:14px; line-height:1.8; margin:0">{advice}</p>
        </div>
        """, unsafe_allow_html=True)

        # PDF download
        st.divider()
        inputs_dict = {
            "Study Hours": f"{study_h} hrs/day",
            "Attendance": f"{attendance_p}%",
            "Previous Marks": f"{prev_m}/100",
            "Sleep Hours": f"{sleep_h} hrs/day",
            "Assignment Completion": f"{assignment_p}%"
        }
        pdf = generate_pdf(name, predicted, grade, advice, inputs_dict)
        st.download_button(
            label="📥 Download Report as PDF",
            data=pdf,
            file_name=f"{name}_performance_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.warning("Please enter your name first!")

st.markdown("<p style='margin-top:20px'>Built by Rohit • Powered by Groq + Scikit-learn</p>", unsafe_allow_html=True)
