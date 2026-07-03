import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
try:
    pipeline = joblib.load("models/customer_churn_pipeline_v1.pkl")
    st.success("✅ Model Loaded Successfully")
except Exception as e:
    st.error(f"❌ Failed to load model.\n\n{e}")
    st.stop()

# -----------------------------
# Title
# -----------------------------
st.title("📊 Customer Churn Prediction")
st.write("Machine Learning Powered Customer Churn Prediction System")

# -----------------------------
# Customer Information
# -----------------------------
st.header("Customer Information")

col1, col2 = st.columns(2)

# Left Column
with col1:

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure",
        min_value=0,
        max_value=10,
        value=5
    )

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=50000.0
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0
    )

# Right Column
with col2:

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    num_products = st.selectbox(
        "Number of Products",
        [1, 2, 3, 4]
    )

    has_card = st.selectbox(
        "Has Credit Card",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    active_member = st.selectbox(
        "Is Active Member",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

# -----------------------------
# Predict Button
# -----------------------------
predict_button = st.button(
    "🔍 Predict Customer Churn",
    use_container_width=True
)

# -----------------------------
# Prediction
# -----------------------------
if predict_button:

    input_df = pd.DataFrame({
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_products],
        "HasCrCard": [has_card],
        "IsActiveMember": [active_member],
        "EstimatedSalary": [estimated_salary]
    })

    prediction = pipeline.predict(input_df)[0]

    probability = pipeline.predict_proba(input_df)[0]

    churn_probability = probability[1]

    st.divider()
    st.subheader("Prediction Result")

    # Prediction Result
    if prediction == 1:
        st.error("❌ Customer is likely to Churn")
    else:
        st.success("✅ Customer is likely to Stay")

    # Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Prediction",
        "Churn" if prediction == 1 else "Stay"
    )

    col2.metric(
        "Probability",
        f"{churn_probability*100:.2f}%"
    )

    if churn_probability < 0.30:
        risk = "Low"
    elif churn_probability < 0.60:
        risk = "Medium"
    else:
        risk = "High"

    col3.metric(
        "Risk Level",
        risk
    )

    st.divider()

    # Recommendation
    if churn_probability < 0.30:

        st.success("🟢 Low Risk Customer")

        st.info("""
### Recommendation

Customer is unlikely to churn.

Continue providing excellent customer service and maintain engagement.
""")

    elif churn_probability < 0.60:

        st.warning("🟡 Medium Risk Customer")

        st.info("""
### Recommendation

Customer shows moderate churn risk.

Consider offering personalized promotions or loyalty benefits.
""")

    else:

        st.error("🔴 High Risk Customer")

        st.info("""
### Recommendation

Customer has a high probability of churning.

Immediate retention strategies such as discounts, personalized offers, or direct customer engagement are recommended.
""")