# streamlit_app.py

import streamlit as st
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image

# Set page config
st.set_page_config(page_title="Assignment Evaluator", page_icon="🖼️", layout="centered")

# Load VGG16 model
@st.cache_resource
def load_model():
    base_model = VGG16(weights='imagenet')
    return Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

model = load_model()

# Feature extraction
def extract_features(img):
    img = img.resize((224, 224)).convert('RGB')
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features

# Score calculation
def calculate_similarity_score(img1, img2):
    features1 = extract_features(img1)
    features2 = extract_features(img2)
    score = cosine_similarity(features1, features2)[0][0]
    return round(score * 100, 2)

# Feedback generator
def generate_feedback(score):
    if score >= 90:
        return "🌟 Excellent! Highly similar to the original."
    elif score >= 70:
        return "👍 Good! Close to the original with minor differences."
    elif score >= 50:
        return "⚠️ Average. Needs improvement in some areas."
    else:
        return "❌ Poor. Too different from the original."

# Title and instructions
st.markdown("<h1 style='text-align: center;'>🎨 Graphic Assignment Evaluator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Compare multiple student submissions with the original design and get instant scores & feedback.</p>", unsafe_allow_html=True)
st.markdown("---")

# File uploaders
st.subheader("📥 Upload Images")

original_file = st.file_uploader("1️⃣ Upload Original Design", type=["jpg", "jpeg", "png"])
student_files = st.file_uploader("2️⃣ Upload Student Submissions (multiple allowed)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if original_file and student_files:
    original_image = Image.open(original_file)

    st.markdown("---")
    st.image(original_image, caption="🎯 Original Design", use_column_width=False, width=300)
    st.markdown("### 📊 Results:")

    for student_file in student_files:
        student_image = Image.open(student_file)

        with st.spinner(f"🔍 Evaluating `{student_file.name}`..."):
            score = calculate_similarity_score(original_image, student_image)
            feedback = generate_feedback(score)

        with st.expander(f"🖼️ {student_file.name}", expanded=False):
            st.image(student_image, width=250)
            st.markdown(f"**🔢 Score:** `{score} / 100`")
            st.markdown(f"**💬 Feedback:** _{feedback}_")
            st.markdown("---")

elif original_file and not student_files:
    st.warning("⚠️ Please upload one or more student submission images.")
elif not original_file:
    st.info("📌 Please upload the original design first.")
