import streamlit as st
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import pandas as pd
import io

st.set_page_config(page_title="Assignment Evaluator", page_icon="🖼️", layout="centered")

@st.cache_resource
def load_model():
    base_model = VGG16(weights='imagenet')
    return Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

model = load_model()

def extract_features(img):
    img = img.resize((224, 224)).convert('RGB')
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features

def calculate_similarity_score(img1, img2):
    features1 = extract_features(img1)
    features2 = extract_features(img2)
    score = cosine_similarity(features1, features2)[0][0]
    return round(score * 100, 2)

def generate_feedback(score):
    if score >= 90:
        return "🌟 Excellent! Highly similar to the original."
    elif score >= 70:
        return "👍 Good! Close to the original with minor differences."
    elif score >= 50:
        return "⚠️ Average. Needs improvement in some areas."
    else:
        return "❌ Poor. Too different from the original."

# UI
st.markdown("<h1 style='text-align: center;'>🎨 Graphic Assignment Evaluator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Compare student submissions with the original design and download results as Excel.</p>", unsafe_allow_html=True)
st.markdown("---")

# Upload
st.subheader("📥 Upload Images")
original_file = st.file_uploader("1️⃣ Upload Original Design", type=["jpg", "jpeg", "png"])
student_files = st.file_uploader("2️⃣ Upload Student Submissions (multiple allowed)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

results = []

if original_file and student_files:
    original_image = Image.open(original_file)
    st.markdown("#### 🎯 Original Design")
    st.image(original_image, width=250)
    st.markdown("---")

    st.markdown("### 📊 Evaluation Results:")

    for student_file in student_files:
        student_image = Image.open(student_file)
        with st.spinner(f"Evaluating {student_file.name}..."):
            score = calculate_similarity_score(original_image, student_image)
            feedback = generate_feedback(score)

        # 👇 Show results directly (not inside dropdown)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(student_image, width=150, caption=student_file.name)
        with col2:
            st.markdown(f"**Score:** `{score}/100`")
            st.markdown(f"**Feedback:** _{feedback}_")
        st.markdown("---")

        results.append({
            "Student File": student_file.name,
            "Score": score,
            "Feedback": feedback
        })

    # Excel Report
    if results:
        df = pd.DataFrame(results)
        excel_file = io.BytesIO()
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
            writer.save()
        st.download_button(
            label="⬇️ Download Excel Report",
            data=excel_file.getvalue(),
            file_name="assignment_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

elif original_file and not student_files:
    st.warning("⚠️ Please upload one or more student submission images.")
elif not original_file:
    st.info("📌 Please upload the original design first.")

