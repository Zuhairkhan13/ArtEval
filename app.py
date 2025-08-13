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

# Set page configuration with a modern theme
st.set_page_config(
    page_title="Graphic Assignment Evaluator",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced dark-themed UI
st.markdown("""
    <style>
    /* General styling */
    body {
        font-family: 'Roboto', 'Arial', sans-serif;
        background: linear-gradient(to bottom, #1c2526, #121212);
        color: #e8ecef;
    }
    .main-title {
        text-align: center;
        color: #ffffff;
        font-size: 2.8em;
        font-weight: 700;
        margin-bottom: 0.5em;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle {
        text-align: center;
        color: #b0bec5;
        font-size: 1.1em;
        margin-bottom: 1em;
    }
    .subheader {
        color: #ffffff;
        font-size: 1.6em;
        font-weight: 600;
        margin-top: 1.2em;
        margin-bottom: 0.5em;
    }
    .info-box {
        background-color: #006064;
        padding: 1.2em;
        border-radius: 12px;
        margin-bottom: 1.2em;
        color: #e8ecef;
        border: 1px solid #00838f;
    }
    .success-box {
        background-color: #2e7d32;
        padding: 1.2em;
        border-radius: 12px;
        margin-bottom: 1.2em;
        color: #e8ecef;
        border: 1px solid #4caf50;
    }
    .warning-box {
        background-color: #ff6d00;
        padding: 1.2em;
        border-radius: 12px;
        margin-bottom: 1.2em;
        color: #e8ecef;
        border: 1px solid #ff8f00;
    }
    .stButton>button {
        background-color: #00bcd4;
        color: #ffffff;
        border-radius: 10px;
        padding: 0.6em 1.5em;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #00acc1;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stFileUploader {
        background-color: #1e2526;
        padding: 1.2em;
        border-radius: 12px;
        border: 1px solid #37474f;
        transition: border-color 0.3s ease;
    }
    .stFileUploader:hover {
        border-color: #00bcd4;
    }
    .result-card {
        background-color: #1e2526;
        padding: 1.2em;
        border-radius: 12px;
        border: 1px solid #37474f;
        margin-bottom: 1.2em;
        transition: all 0.3s ease;
    }
    .result-card:hover {
        border-color: #00bcd4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background-color: #1c2526;
        color: #e8ecef;
    }
    .sidebar .subheader {
        color: #ffffff;
    }
    .footer {
        text-align: center;
        color: #b0bec5;
        font-size: 0.9em;
        margin-top: 2.5em;
        padding: 1.5em;
        border-top: 1px solid #37474f;
    }
    .footer a {
        color: #00bcd4;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    .footer a:hover {
        color: #00acc1;
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Load VGG16 model
@st.cache_resource
def load_model():
    base_model = VGG16(weights='imagenet')
    return Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

model = load_model()

# Feature extraction function
def extract_features(img):
    img = img.resize((224, 224)).convert('RGB')
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features

# Calculate similarity score
def calculate_similarity_score(img1, img2):
    features1 = extract_features(img1)
    features2 = extract_features(img2)
    score = cosine_similarity(features1, features2)[0][0]
    return round(score * 100, 2)

# Generate feedback and status based on score
def generate_feedback(score):
    if score >= 90:
        return "Pass", "🌟 Excellent! Highly similar to the original."
    elif score >= 70:
        return "Pass", "👍 Good! Close to the original with minor differences."
    elif score >= 50:
        return "Fail", "⚠️ Average. Needs improvement in some areas."
    else:
        return "Fail", "❌ Poor. Too different from the original."

# Sidebar for About section
with st.sidebar:
    st.markdown("<h2 class='subheader'>About This App</h2>", unsafe_allow_html=True)
    st.markdown("""
        **Graphic Design Assignment Evaluator** helps teachers quickly check multiple student submissions at once.

        Just upload:
        - **One original design**
        - **Multiple student designs**

        The app will:
        - Compare each submission with the original  
        - Show pass/fail status and feedback  
        - Display results with images side by side  
        - Allow downloading a full **Excel report**

        Built to save time for teachers and design instructors by automating the assignment checking process.
    """, unsafe_allow_html=True)

# Main UI
st.markdown("<div class='main-title'>🎨 Graphic Assignment Evaluator</div>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Compare student submissions with the original design and download results as Excel.</p>", unsafe_allow_html=True)
st.markdown("---")

# File upload section
st.markdown("<div class='subheader'>📥 Upload Images</div>", unsafe_allow_html=True)
original_file = st.file_uploader("1️⃣ Upload Original Design", type=["jpg", "jpeg", "png"], key="original")
student_files = st.file_uploader("2️⃣ Upload Student Submissions (multiple allowed)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="students")

results = []

# Display original image and results
if original_file and student_files:
    original_image = Image.open(original_file)
    st.markdown("<div class='subheader'>🎯 Original Design</div>", unsafe_allow_html=True)
    st.image(original_image, width=300, use_container_width=False)
    st.markdown("---")

    st.markdown("<div class='subheader'>📊 Evaluation Results</div>", unsafe_allow_html=True)
    
    # Collapsible results section
    with st.expander("View All Results", expanded=True):
        for student_file in student_files:
            student_image = Image.open(student_file)
            with st.spinner(f"Evaluating {student_file.name}..."):
                score = calculate_similarity_score(original_image, student_image)
                status, feedback = generate_feedback(score)

            # Display result in a card-like layout
            with st.container():
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(student_image, width=150, use_container_width=False, caption=student_file.name)
                with col2:
                    st.markdown(f"**File:** {student_file.name}")
                    st.markdown(f"**Status:** {status}")
                    st.markdown(f"**Feedback:** _{feedback}_")
                st.markdown("</div>", unsafe_allow_html=True)

            results.append({
                "Student File": student_file.name,
                "Status": status,
                "Feedback": feedback
            })

    # Excel download button
    if results:
        df = pd.DataFrame(results)
        excel_file = io.BytesIO()
        with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
        st.download_button(
            label="⬇️ Download Excel Report",
            data=excel_file.getvalue(),
            file_name="assignment_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download"
        )

elif original_file and not student_files:
    st.markdown("<div class='warning-box'>⚠️ Please upload one or more student submission images.</div>", unsafe_allow_html=True)
elif not original_file:
    st.markdown("<div class='info-box'>📌 Please upload the original design first.</div>", unsafe_allow_html=True)

# Footer with GitHub link
st.markdown("""
    <div class='footer'>
        Developed by Zuhair khan | 
        <a href='https://github.com/Zuhairkhan13' target='_blank'>GitHub: Zuhair khan</a>
    </div>
""", unsafe_allow_html=True)