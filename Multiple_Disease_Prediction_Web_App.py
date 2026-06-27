import streamlit as st
import numpy as np
import pickle
import os

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Multiple Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# =====================================
# MODEL LOADING
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Diabetes
diabetes_model = pickle.load(
    open(os.path.join(BASE_DIR, "trained_model.sav"), "rb")
)

diabetes_scaler = pickle.load(
    open(os.path.join(BASE_DIR, "scaler.sav"), "rb")
)

# Heart Disease
heart_model = pickle.load(
    open(
        os.path.join(
            BASE_DIR,
            "trained_model_heart_disease.sav"
        ),
        "rb"
    )
)

# Breast Cancer
breast_model = pickle.load(
    open(
        os.path.join(
            BASE_DIR,
            "trained_model_breast_cancer.sav"
        ),
        "rb"
    )
)

# Parkinson
parkinson_model = pickle.load(
    open(
        os.path.join(
            BASE_DIR,
            "trained_model_parkinson.sav"
        ),
        "rb"
    )
)

parkinson_scaler = pickle.load(
    open(
        os.path.join(
            BASE_DIR,
            "scaler_parkinson.sav"
        ),
        "rb"
    )
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 1rem;
}

.stButton>button {
    width: 100%;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
}

.card {
    background-color: #1E293B;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}

.big-title {
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#4CAF50;
}

.sub-title {
    text-align:center;
    font-size:18px;
    color:#B0B0B0;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("🏥 HealthAI Suite")

    st.markdown("---")

    selected = st.radio(
        "Select Disease Prediction",
        [
            "🏠 Home",
            "🩸 Diabetes Prediction",
            "❤️ Heart Disease Prediction",
            "🎗️ Breast Cancer Prediction",
            "🧠 Parkinson Prediction"
        ]
    )

    st.markdown("---")

    st.subheader("Developer")

    st.write("Prathama Debnath")
    st.write("B.Tech AIML")

    st.markdown("---")

    st.info(
        "Integrated AI Healthcare Prediction Dashboard"
    )

# =====================================
# HOME PAGE
# =====================================

if selected == "🏠 Home":

    st.markdown(
        '<p class="big-title">🏥 Multiple Disease Prediction System</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sub-title">AI Powered Healthcare Risk Assessment Platform</p>',
        unsafe_allow_html=True
    )

    # ===========================================
    # HOME DASHBOARD METRICS
    # ===========================================

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            label="🩺 Diseases",
            value="4"
        )

    with c2:
        st.metric(
            label="🤖 Models",
            value="4"
        )

    with c3:
        st.metric(
            label="⚙️ Algorithms",
            value="SVM + LR"
        )

    with c4:
        st.metric(
            label="🎯 Accuracy",
            value="High"
        )

    st.markdown("---")

    st.header("📋 Available Modules")

    col1, col2 = st.columns(2)

    with col1:

        st.success("""
    ### 🩸 Diabetes Prediction

    **Model Used:** Support Vector Machine (SVM)

    **Features:** 8 Clinical Parameters

    **Prediction:** Diabetic / Non-Diabetic
    """)

        st.success("""
    ### ❤️ Heart Disease Prediction

    **Model Used:** Logistic Regression

    **Features:** 13 Clinical Parameters

    **Prediction:** Heart Disease Risk
    """)

    with col2:

        st.success("""
    ### 🎀 Breast Cancer Prediction

    **Model Used:** Logistic Regression

    **Features:** 30 Clinical Parameters

    **Prediction:** Benign / Malignant
    """)

        st.success("""
    ### 🧠 Parkinson Prediction

    **Model Used:** Support Vector Machine (SVM)

    **Features:** 22 Voice Parameters

    **Prediction:** Parkinson's Disease
    """)

    st.markdown("---")

    st.subheader("🛠 Tech Stack")

    st.write("""
    • Python

    • Streamlit

    • Scikit-Learn

    • NumPy

    • Machine Learning

    • Logistic Regression

    • Support Vector Machine
    """)

    st.markdown("---")

    st.caption(
        "Developed by Prathama Debnath | AIML | Healthcare AI Dashboard"
    )


    # =====================================
# DIABETES PREDICTION PAGE
# =====================================

elif selected == "🩸 Diabetes Prediction":

    st.title("🩸 Diabetes Prediction System")

    st.markdown(
        "<h4 style='text-align:center;'>AI Powered Diabetes Risk Assessment</h4>",
        unsafe_allow_html=True
    )

    st.write("")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Features", "8")

    with m2:
        st.metric("Model", "SVM")

    with m3:
        st.metric("Prediction", "Instant")

    st.divider()

    st.subheader("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        Pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            value=0,
            key="db1"
        )

        Glucose = st.number_input(
            "Glucose Level",
            min_value=0,
            value=120,
            key="db2"
        )

        BloodPressure = st.number_input(
            "Blood Pressure",
            min_value=0,
            value=70,
            key="db3"
        )

        SkinThickness = st.number_input(
            "Skin Thickness",
            min_value=0,
            value=20,
            key="db4"
        )

    with col2:

        Insulin = st.number_input(
            "Insulin Level",
            min_value=0,
            value=80,
            key="db5"
        )

        BMI = st.number_input(
            "BMI",
            min_value=0.0,
            value=25.0,
            key="db6"
        )

        DiabetesPedigreeFunction = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            value=0.5,
            key="db7"
        )

        Age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30,
            key="db8"
        )

    st.write("")

    if st.button(
        "🔍 Predict Diabetes",
        key="predict_diabetes"
    ):

        input_data = [
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age
        ]

        input_data_np = np.asarray(input_data)

        input_data_reshaped = input_data_np.reshape(1, -1)

        std_data = diabetes_scaler.transform(
            input_data_reshaped
        )

        prediction = diabetes_model.predict(
            std_data
        )

        probability = diabetes_model.predict_proba(
            std_data
        )

        risk_percent = round(
            probability[0][1] * 100,
            2
        )

        st.divider()

        st.subheader("📊 Risk Assessment")

        st.progress(
            min(
                int(risk_percent),
                100
            )
        )

        st.metric(
            "Risk Score",
            f"{risk_percent}%"
        )

        if prediction[0] == 1:

            st.error(
                f"""
⚠️ HIGH RISK

Estimated Diabetes Risk:
{risk_percent}%

The model predicts that the patient may have diabetes.

Please consult a healthcare professional.
"""
            )

        else:

            st.success(
                f"""
✅ LOW RISK

Estimated Diabetes Risk:
{risk_percent}%

The model predicts that the patient is unlikely to have diabetes.
"""
            )

    st.markdown("---")

    st.caption(
        "Developed by Prathama Debnath | AIML | Diabetes Prediction Module"
    )


    # =====================================
# HEART DISEASE PREDICTION PAGE
# =====================================

elif selected == "❤️ Heart Disease Prediction":

    st.title("❤️ Heart Disease Prediction System")

    st.markdown(
        "<h4 style='text-align:center;'>AI Powered Medical Risk Assessment</h4>",
        unsafe_allow_html=True
    )

    st.write("")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Features", "13")

    with m2:
        st.metric("Model", "Logistic Regression")

    with m3:
        st.metric("Prediction", "Instant")

    st.divider()

    st.subheader("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        Age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=40,
            key="heart1"
        )

        Sex = st.selectbox(
            "Sex",
            [0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male",
            key="heart2"
        )

        Chestpain = st.number_input(
            "Chest Pain Type",
            min_value=0,
            max_value=3,
            value=1,
            key="heart3"
        )

        Resting_blood_pressure = st.number_input(
            "Resting Blood Pressure",
            value=120,
            key="heart4"
        )

        Serumcholesterol = st.number_input(
            "Serum Cholesterol",
            value=200,
            key="heart5"
        )

        Fasting_blood_sugar = st.selectbox(
            "Fasting Blood Sugar (>120 mg/dl)",
            [0, 1],
            key="heart6"
        )

        RE = st.number_input(
            "Resting ECG Result",
            min_value=0,
            max_value=2,
            value=0,
            key="heart7"
        )

    with col2:

        RE_ex = st.number_input(
            "Maximum Heart Rate Achieved",
            value=150,
            key="heart8"
        )

        Exercise_induced_angina = st.selectbox(
            "Exercise Induced Angina",
            [0, 1],
            key="heart9"
        )

        oldpeak = st.number_input(
            "Oldpeak",
            value=1.0,
            key="heart10"
        )

        slope = st.number_input(
            "Slope Value",
            min_value=0,
            max_value=2,
            value=1,
            key="heart11"
        )

        ca = st.number_input(
            "Number of Major Vessels",
            min_value=0,
            max_value=4,
            value=0,
            key="heart12"
        )

        thal = st.number_input(
            "Thal Value",
            min_value=0,
            max_value=3,
            value=2,
            key="heart13"
        )

    st.write("")

    if st.button(
        "🔍 Predict Heart Disease",
        key="predict_heart"
    ):

        input_data = [
            Age,
            Sex,
            Chestpain,
            Resting_blood_pressure,
            Serumcholesterol,
            Fasting_blood_sugar,
            RE,
            RE_ex,
            Exercise_induced_angina,
            oldpeak,
            slope,
            ca,
            thal
        ]

        input_data_np = np.asarray(input_data)

        input_data_reshaped = input_data_np.reshape(1, -1)

        prediction = heart_model.predict(
            input_data_reshaped
        )

        st.divider()

        st.subheader("📊 Risk Assessment")

        if prediction[0] == 1:

            st.error(
                """
⚠️ HIGH RISK

The model predicts that the patient may have heart disease.

Please consult a healthcare professional immediately.
"""
            )

        else:

            st.success(
                """
✅ LOW RISK

The model predicts that the patient is unlikely to have heart disease.
"""
            )

    st.markdown("---")

    st.caption(
        "Developed by Prathama Debnath | AIML | Heart Disease Prediction Module"
    )


    # =====================================
# BREAST CANCER PREDICTION PAGE
# =====================================

elif selected == "🎗️ Breast Cancer Prediction":

    st.markdown(
        "<h1 style='text-align:center;color:#FF4B91;'>🎗️ Breast Cancer Prediction System</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<h4 style='text-align:center;'>AI Powered Clinical Risk Assessment using Logistic Regression</h4>",
        unsafe_allow_html=True
    )

    st.divider()

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Features", "30")

    with m2:
        st.metric("Model", "Logistic Regression")

    with m3:
        st.metric("Prediction", "Instant")

    st.divider()

    def breast_cancer_prediction(input_data):

        input_data_as_numpy_array = np.asarray(input_data)

        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        prediction = breast_model.predict(
            input_data_reshaped
        )

        probability = breast_model.predict_proba(
            input_data_reshaped
        )

        confidence = round(
            max(probability[0]) * 100,
            2
        )

        return prediction[0], confidence

    st.subheader("🩺 Tumor Measurements")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Mean Features")

        mean_radius = st.number_input(
            "Mean Radius",
            value=14.0,
            key="bc1"
        )

        mean_texture = st.number_input(
            "Mean Texture",
            value=19.0,
            key="bc2"
        )

        mean_perimeter = st.number_input(
            "Mean Perimeter",
            value=90.0,
            key="bc3"
        )

        mean_area = st.number_input(
            "Mean Area",
            value=600.0,
            key="bc4"
        )

        mean_smoothness = st.number_input(
            "Mean Smoothness",
            value=0.10,
            format="%.5f",
            key="bc5"
        )

        mean_compactness = st.number_input(
            "Mean Compactness",
            value=0.10,
            format="%.5f",
            key="bc6"
        )

        mean_concavity = st.number_input(
            "Mean Concavity",
            value=0.10,
            format="%.5f",
            key="bc7"
        )

        mean_concave_points = st.number_input(
            "Mean Concave Points",
            value=0.05,
            format="%.5f",
            key="bc8"
        )

        mean_symmetry = st.number_input(
            "Mean Symmetry",
            value=0.18,
            format="%.5f",
            key="bc9"
        )

        mean_fractal_dimension = st.number_input(
            "Mean Fractal Dimension",
            value=0.06,
            format="%.5f",
            key="bc10"
        )

        st.subheader("🧪 Error Features")

        radius_error = st.number_input(
            "Radius Error",
            value=0.40,
            format="%.5f",
            key="bc11"
        )

        texture_error = st.number_input(
            "Texture Error",
            value=1.20,
            format="%.5f",
            key="bc12"
        )

        perimeter_error = st.number_input(
            "Perimeter Error",
            value=2.80,
            format="%.5f",
            key="bc13"
        )

        area_error = st.number_input(
            "Area Error",
            value=40.0,
            key="bc14"
        )

        smoothness_error = st.number_input(
            "Smoothness Error",
            value=0.005,
            format="%.6f",
            key="bc15"
        )

    with col2:

        st.subheader("⚙ Error Features")

        compactness_error = st.number_input(
            "Compactness Error",
            value=0.020,
            format="%.6f",
            key="bc16"
        )

        concavity_error = st.number_input(
            "Concavity Error",
            value=0.030,
            format="%.6f",
            key="bc17"
        )

        concave_points_error = st.number_input(
            "Concave Points Error",
            value=0.010,
            format="%.6f",
            key="bc18"
        )

        symmetry_error = st.number_input(
            "Symmetry Error",
            value=0.020,
            format="%.6f",
            key="bc19"
        )

        fractal_dimension_error = st.number_input(
            "Fractal Dimension Error",
            value=0.003,
            format="%.6f",
            key="bc20"
        )

        st.subheader("⚠ Worst Features")


        worst_radius = st.number_input(
            "Worst Radius",
            value=16.0,
            key="bc21"
        )

        worst_texture = st.number_input(
            "Worst Texture",
            value=25.0,
            key="bc22"
        )

        worst_perimeter = st.number_input(
            "Worst Perimeter",
            value=105.0,
            key="bc23"
        )

        worst_area = st.number_input(
            "Worst Area",
            value=800.0,
            key="bc24"
        )

        worst_smoothness = st.number_input(
            "Worst Smoothness",
            value=0.14,
            format="%.5f",
            key="bc25"
        )

        worst_compactness = st.number_input(
            "Worst Compactness",
            value=0.25,
            format="%.5f",
            key="bc26"
        )

        worst_concavity = st.number_input(
            "Worst Concavity",
            value=0.30,
            format="%.5f",
            key="bc27"
        )

        worst_concave_points = st.number_input(
            "Worst Concave Points",
            value=0.12,
            format="%.5f",
            key="bc28"
        )

        worst_symmetry = st.number_input(
            "Worst Symmetry",
            value=0.30,
            format="%.5f",
            key="bc29"
        )

        worst_fractal_dimension = st.number_input(
            "Worst Fractal Dimension",
            value=0.08,
            format="%.5f",
            key="bc30"
        )

    if st.button(
        "🔍 Predict Breast Cancer",
        key="predict_breast"
    ):

        prediction, confidence = breast_cancer_prediction([

            mean_radius,
            mean_texture,
            mean_perimeter,
            mean_area,
            mean_smoothness,
            mean_compactness,
            mean_concavity,
            mean_concave_points,
            mean_symmetry,
            mean_fractal_dimension,

            radius_error,
            texture_error,
            perimeter_error,
            area_error,
            smoothness_error,

            compactness_error,
            concavity_error,
            concave_points_error,
            symmetry_error,
            fractal_dimension_error,

            worst_radius,
            worst_texture,
            worst_perimeter,
            worst_area,
            worst_smoothness,
            worst_compactness,
            worst_concavity,
            worst_concave_points,
            worst_symmetry,
            worst_fractal_dimension
        ])

        st.markdown("---")

        st.subheader("📈 Prediction Result")

        st.progress(
            min(
                int(confidence),
                100
            )
        )

        st.metric(
            "Model Confidence",
            f"{confidence}%"
        )

        if prediction == 1:

            st.success(
                f"""
✅ BENIGN TUMOR

Confidence Score:
{confidence}%

The model predicts that the tumor is BENIGN (non-cancerous).

Regular medical checkups are still recommended.
"""
            )

        else:

            st.error(
                f"""
⚠️ MALIGNANT TUMOR DETECTED

Confidence Score:
{confidence}%

The model predicts that the tumor may be MALIGNANT (cancerous).

Please consult a healthcare professional immediately.
"""
            )

    st.markdown("---")

    st.caption(
        "Developed by Prathama Debnath | AIML | Breast Cancer Prediction Module"
    )



    # =====================================
# PARKINSON PREDICTION PAGE
# =====================================

elif selected == "🧠 Parkinson Prediction":

    st.title("🧠 Parkinson Disease Prediction System")

    st.markdown(
        "<h4 style='text-align:center;'>AI Powered Parkinson Risk Assessment using SVM</h4>",
        unsafe_allow_html=True
    )

    st.divider()

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Features", "22")

    with m2:
        st.metric("Model", "SVM")

    with m3:
        st.metric("Prediction", "Instant")

    st.divider()

    def predict_parkinson(input_data):

        input_data_np = np.asarray(input_data)

        input_data_reshaped = input_data_np.reshape(1, -1)

        std_data = parkinson_scaler.transform(
            input_data_reshaped
        )

        prediction = parkinson_model.predict(
            std_data
        )

        return prediction[0]

    col1, col2 = st.columns(2)

    with col1:

        fo = st.number_input(
            "MDVP:Fo(Hz)",
            value=120.267,
            key="pk1"
        )

        fhi = st.number_input(
            "MDVP:Fhi(Hz)",
            value=137.244,
            key="pk2"
        )

        flo = st.number_input(
            "MDVP:Flo(Hz)",
            value=114.820,
            key="pk3"
        )

        jitter_percent = st.number_input(
            "MDVP:Jitter(%)",
            value=0.00333,
            format="%.5f",
            key="pk4"
        )

        jitter_abs = st.number_input(
            "MDVP:Jitter(Abs)",
            value=0.00003,
            format="%.5f",
            key="pk5"
        )

        rap = st.number_input(
            "MDVP:RAP",
            value=0.00155,
            format="%.5f",
            key="pk6"
        )

        ppq = st.number_input(
            "MDVP:PPQ",
            value=0.00202,
            format="%.5f",
            key="pk7"
        )

        ddp = st.number_input(
            "Jitter:DDP",
            value=0.00466,
            format="%.5f",
            key="pk8"
        )

        shimmer = st.number_input(
            "MDVP:Shimmer",
            value=0.01608,
            format="%.5f",
            key="pk9"
        )

        shimmer_db = st.number_input(
            "MDVP:Shimmer(dB)",
            value=0.140,
            key="pk10"
        )

        apq3 = st.number_input(
            "Shimmer:APQ3",
            value=0.00779,
            format="%.5f",
            key="pk11"
        )

    with col2:

        apq5 = st.number_input(
            "Shimmer:APQ5",
            value=0.00937,
            format="%.5f",
            key="pk12"
        )

        apq = st.number_input(
            "MDVP:APQ",
            value=0.01351,
            format="%.5f",
            key="pk13"
        )

        dda = st.number_input(
            "Shimmer:DDA",
            value=0.02337,
            format="%.5f",
            key="pk14"
        )

        nhr = st.number_input(
            "NHR",
            value=0.02211,
            format="%.5f",
            key="pk15"
        )

        hnr = st.number_input(
            "HNR",
            value=21.033,
            key="pk16"
        )

        rpde = st.number_input(
            "RPDE",
            value=0.414783,
            format="%.6f",
            key="pk17"
        )

        dfa = st.number_input(
            "DFA",
            value=0.815285,
            format="%.6f",
            key="pk18"
        )

        spread1 = st.number_input(
            "spread1",
            value=-4.813031,
            key="pk19"
        )

        spread2 = st.number_input(
            "spread2",
            value=0.266482,
            key="pk20"
        )

        d2 = st.number_input(
            "D2",
            value=2.301442,
            key="pk21"
        )

        ppe = st.number_input(
            "PPE",
            value=0.284654,
            key="pk22"
        )

    if st.button(
        "🔍 Predict Parkinson Disease",
        key="predict_parkinson"
    ):

        prediction = predict_parkinson([

            fo,
            fhi,
            flo,
            jitter_percent,
            jitter_abs,
            rap,
            ppq,
            ddp,
            shimmer,
            shimmer_db,
            apq3,
            apq5,
            apq,
            dda,
            nhr,
            hnr,
            rpde,
            dfa,
            spread1,
            spread2,
            d2,
            ppe
        ])

        st.divider()

        st.subheader("📊 Prediction Result")

        if prediction == 1:

            st.error(
                """
⚠️ PARKINSON'S DISEASE DETECTED

The model predicts that the person may have Parkinson's Disease.

Please consult a healthcare professional.
"""
            )

        else:

            st.success(
                """
✅ NO PARKINSON'S DISEASE DETECTED

The model predicts that the person is unlikely to have Parkinson's Disease.
"""
            )

    st.markdown("---")

    st.caption(
        "🏥 HealthAI Suite | AI Powered Multiple Disease Prediction System"
    )

    st.caption(
        "Developed by Prathama Debnath | B.Tech AIML | "
        "Streamlit • Scikit-Learn • Machine Learning"
    )