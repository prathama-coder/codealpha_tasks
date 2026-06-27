import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="wide"
)

# --------------------------------------------------
# Custom Styling
# --------------------------------------------------
st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

h1 {
    color: #4CAF50;
}

[data-testid="metric-container"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load Model & Scaler
# --------------------------------------------------
model = joblib.load("credit_card_model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("📊 Model Information")

st.sidebar.info("""
Model: Logistic Regression

Accuracy: 73%

ROC-AUC Score: 0.66

Features Used: 85

Purpose:
Predict customer credit card default risk using machine learning.
""")

# --------------------------------------------------
# Main Header
# --------------------------------------------------
st.title("💳 Credit Risk & Default Prediction Dashboard")

st.markdown("""
This dashboard predicts the likelihood of customer credit card default
using a machine learning model trained on customer financial behavior.
Upload a customer dataset to perform batch risk analysis.
""")

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📁 Upload Customer Dataset (CSV)",
    type=["csv"]
)

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Dataset")

    st.dataframe(
        data.head(),
        use_container_width=True
    )

    # ----------------------------------------------
    # Preprocessing
    # ----------------------------------------------
    if "CUST_ID" in data.columns:
        data = data.drop("CUST_ID", axis=1)

    if "CAT_GAMBLING" in data.columns:

        data["CAT_GAMBLING"] = data["CAT_GAMBLING"].map({
            "Low": 0,
            "Medium": 1,
            "High": 2
        })

        data["CAT_GAMBLING"] = data["CAT_GAMBLING"].fillna(0)

    if "DEFAULT" in data.columns:
        data = data.drop("DEFAULT", axis=1)

    # ----------------------------------------------
    # Scaling
    # ----------------------------------------------
    scaled_data = scaler.transform(data)

    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------
    predictions = model.predict(scaled_data)

    result = pd.DataFrame()

    result["Prediction"] = predictions

    result["Risk Level"] = result["Prediction"].map({
        0: "🟢 Low Risk",
        1: "🔴 High Risk"
    })

    # ----------------------------------------------
    # Summary Metrics
    # ----------------------------------------------
    total = len(result)

    defaults = (result["Prediction"] == 1).sum()

    safe = (result["Prediction"] == 0).sum()

    default_rate = round((defaults / total) * 100, 2)

    st.subheader("📈 Prediction Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        total
    )

    col2.metric(
        "🔴 High Risk Customers",
        defaults
    )

    col3.metric(
        "🟢 Low Risk Customers",
        safe
    )

    col4.metric(
        "Default Rate",
        f"{default_rate}%"
    )

    st.success("✅ Prediction completed successfully!")

    # ----------------------------------------------
    # Pie Chart
    # ----------------------------------------------
    chart_data = pd.DataFrame({
        "Category": ["🟢 Low Risk", "🔴 High Risk"],
        "Count": [safe, defaults]
    })

    fig = px.pie(
        chart_data,
        names="Category",
        values="Count",
        hole=0.45,
        title="Customer Risk Distribution"
    )

    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ----------------------------------------------
    # Prediction Results
    # ----------------------------------------------
    st.subheader("🔍 Customer Risk Assessment")

    st.dataframe(
        result,
        use_container_width=True
    )

    # ----------------------------------------------
    # Download Report
    # ----------------------------------------------
    csv = result.to_csv(index=False)

    st.download_button(
        label="📥 Download Risk Analysis Report",
        data=csv,
        file_name="credit_risk_report.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")

st.markdown("""
### 👨‍💻 Project Information

**Credit Risk & Default Prediction Dashboard**

- Machine Learning Model: Logistic Regression
- Accuracy: 73%
- ROC-AUC Score: 0.66
- Features Used: 85

Developed by **Prathama Debnath**  
B.Tech AIML, SRM University Ramapuram
""")