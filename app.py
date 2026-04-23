import streamlit as st
import pandas as pd
import joblib

# ===============================
# 1. LOAD ASSETS
# ===============================
model = joblib.load('final_churn_model.pkl')
scaler = joblib.load('scaler.pkl')
ohe_gen = joblib.load('ohe_general.pkl')
ohe_pay = joblib.load('ohe_payment.pkl')

st.set_page_config(page_title="Customer Churn Predictor", layout="wide", page_icon="📞")

st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .right-bar {
        background-color: #f1f3f5;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

import os

def set_theme(is_dark):
    os.makedirs(".streamlit", exist_ok=True)
    theme_str = "[theme]\nbase='dark'\n" if is_dark else "[theme]\nbase='light'\n"
    with open(".streamlit/config.toml", "w") as f:
        f.write(theme_str)

if "dark_mode" not in st.session_state:
    try:
        with open(".streamlit/config.toml", "r") as f:
            content = f.read()
            st.session_state.dark_mode = "base='dark'" in content or 'base="dark"' in content
    except FileNotFoundError:
        st.session_state.dark_mode = False

# ===============================
# SIDEBAR — About & context
# ===============================
with st.sidebar:
    st.markdown("### 🎨 Appearance")
    dark_mode_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode_toggle
        set_theme(dark_mode_toggle)
        st.rerun()

    st.markdown("---")
    st.markdown("## 📖 About this app")
    st.markdown("---")

    with st.expander("**What this website does**", expanded=True):
        st.markdown("""
        This is a **Customer Churn Predictor** for telecom/service customers.  
        You enter a customer's profile (demographics, services, contract, charges),  
        and the app predicts the **probability they will churn** (leave the company).  
        Results are shown as **Low / Medium / High risk** with a percentage.
        """)

    with st.expander("**Use cases**"):
        st.markdown("""
        - **Retention teams** — Find high-risk customers and target them with offers or support.
        - **Sales & support** — Prioritize calls and interventions by churn risk.
        - **Marketing** — Design campaigns for at-risk segments (e.g. month-to-month, high charges).
        - **Product** — See which services (e.g. no tech support, no contract) correlate with churn.
        - **Executives** — Track and reduce churn rate using data-driven predictions.
        """)

    with st.expander("**Where the features come from**"):
        st.markdown("""
        The inputs mirror a **telecom/cable-style customer dataset**:
        - **Demographics** — Gender, senior citizen, partner, dependents (from customer records).
        - **Tenure** — How long they've been with the company (months).
        - **Services** — Phone, internet, multiple lines, streaming TV/movies, security, backup, device protection, tech support (from product/subscription data).
        - **Billing** — Contract type, paperless billing, payment method, monthly and total charges (from billing systems).
        The model was trained on a customer churn dataset with these same kinds of features.
        """)

    st.markdown("---")
    st.markdown("👤 **Developer:** [shereefthr@gmail.com](mailto:shereefthr@gmail.com) | [GitHub Profile](https://github.com/mhd-shereef)")

st.title("📞 Customer Churn Predictor")
st.markdown("Analyze customer profiles to estimate the likelihood of churn based on demographics, services, and billing patterns.")

# ===============================
# 2. MAIN LAYOUT (Left = Inputs, Right = Results)
# ===============================
main_col, right_col = st.columns([2, 1], gap="large")

with main_col:
    st.subheader("📝 Customer Profile Inputs")
    tab1, tab2, tab3 = st.tabs(["👤 Demographics", "🛠️ Services", "💳 Billing"])
    
    with tab1:
        st.markdown("##### Customer Demographics & Tenure")
        col11, col12 = st.columns(2)
        gender = col11.radio("Gender", ["Male", "Female"])
        senior = col12.radio("Senior Citizen", ["Yes", "No"])
        partner = col11.radio("Partner", ["Yes", "No"])
        dependents = col12.radio("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        
    with tab2:
        st.markdown("##### Subscribed Services")
        col21, col22 = st.columns(2)
        phone = col21.radio("Phone Service", ["Yes", "No"])
        multiple = col22.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = col21.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = col22.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = col21.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        protection = col22.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        support = col21.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = col22.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = col21.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
    with tab3:
        st.markdown("##### Contract & Payment Details")
        col31, col32 = st.columns(2)
        contract = col31.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = col32.radio("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        m_charges = col31.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
        
        # Auto-calculate total charges
        t_charges = tenure * m_charges
        col32.info(f"💰 **Estimated Total Charges:** ${t_charges:.2f}")

with right_col:
    st.subheader("📊 Output & Model Details")
    
    # Model Info Card
    # Change text colors depending on the theme so it stays readable in dark mode
    card_bg = "#1e1e1e" if st.session_state.dark_mode else "#e9ecef"
    text_col = "#ffffff" if st.session_state.dark_mode else "#000000"
    
    st.markdown(f"""
        <div style="background-color: {card_bg}; color: {text_col}; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #228be6;">
            <h5 style="margin: 0 0 10px 0; color: #4dabf7;">🧠 Model Insights</h5>
            <p style="margin: 0; font-size: 14px;"><strong>Algorithm:</strong> Random Forest Classifier</p>
            <p style="margin: 0; font-size: 14px;"><strong>Accuracy:</strong> ~77.2%</p>
            <p style="margin: 0; font-size: 14px;"><strong>Optimization:</strong> GridSearchCV Tuned</p>
        </div>
    """, unsafe_allow_html=True)
    
    predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True)

    if predict_btn:
        # User Demographics Summary Card
        demo_bg = "#2b2210" if st.session_state.dark_mode else "#fdf4e3"
        st.markdown(f"""
            <div style="background-color: {demo_bg}; color: {text_col}; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #f5a623;">
                <h6 style="margin: 0 0 10px 0; color: #ffc078;">👤 Selected Demographics</h6>
                <ul style="margin: 0; padding-left: 20px; font-size: 14px;">
                    <li><strong>Gender:</strong> {gender}</li>
                    <li><strong>Senior Citizen:</strong> {senior}</li>
                    <li><strong>Partner:</strong> {partner}</li>
                    <li><strong>Dependents:</strong> {dependents}</li>
                    <li><strong>Tenure:</strong> {tenure} months</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        data = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multiple,
            'InternetService': internet,
            'OnlineSecurity': security,
            'OnlineBackup': backup,
            'DeviceProtection': protection,
            'TechSupport': support,
            'StreamingTV': tv,
            'StreamingMovies': movies,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': m_charges,
            'TotalCharges': t_charges
        }

        df_input = pd.DataFrame([data])

        # Preprocessing
        df_input['gender'] = df_input['gender'].map({'Male': 0, 'Female': 1})
        for c in ['Partner', 'SeniorCitizen', 'Dependents', 'PhoneService', 'PaperlessBilling']:
            df_input[c] = df_input[c].map({'Yes': 1, 'No': 0})

        ohe_cols = [
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract'
        ]

        gen_enc = ohe_gen.transform(df_input[ohe_cols])
        gen_df = pd.DataFrame(gen_enc, columns=ohe_gen.get_feature_names_out(ohe_cols), index=df_input.index)

        pay_enc = ohe_pay.transform(df_input[['PaymentMethod']])
        pay_df = pd.DataFrame(pay_enc, columns=ohe_pay.get_feature_names_out(['PaymentMethod']), index=df_input.index)

        df_final = df_input.drop(columns=ohe_cols + ['PaymentMethod'])
        df_final = pd.concat([df_final, gen_df, pay_df], axis=1)

        df_final[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
            df_final[['tenure', 'MonthlyCharges', 'TotalCharges']]
        )

        df_final = df_final[model.feature_names_in_]

        # Prediction
        prob = model.predict_proba(df_final)[0][1]

        st.markdown("### 🎯 Prediction Result")
        if prob >= 0.5:
            st.error(f"**HIGH RISK: CHURN** ❌\n\nProbability: {prob:.1%}")
        elif prob >= 0.3:
            st.warning(f"**MEDIUM RISK** ⚠️\n\nProbability: {prob:.1%}")
        else:
            st.success(f"**LOW RISK: STAY** ✅\n\nProbability: {prob:.1%}")