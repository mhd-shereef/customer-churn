import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import streamlit.components.v1 as components

# ===============================
# 0. HTML RENDERER HELPER
# ===============================
def render_html(html_str):
    """
    Safely renders HTML in Streamlit by stripping ALL leading spaces from every line.
    This strictly prevents Streamlit's Markdown parser from accidentally treating 
    indented HTML lines as <pre><code> blocks!
    """
    cleaned_html = "\n".join([line.strip() for line in html_str.split("\n")])
    st.markdown(cleaned_html, unsafe_allow_html=True)


# ===============================
# 1. LOAD ASSETS & DATA
# ===============================
@st.cache_resource
def load_models():
    model = joblib.load('final_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    ohe_gen = joblib.load('ohe_general.pkl')
    ohe_pay = joblib.load('ohe_payment.pkl')
    return model, scaler, ohe_gen, ohe_pay

model, scaler, ohe_gen, ohe_pay = load_models()

@st.cache_data
def load_data():
    return pd.read_csv('Customer_churn.csv')

df_historic = load_data()

st.set_page_config(page_title="Churn Analytics Dashboard", layout="wide", page_icon="📈")

# Invisible anchor for "Scroll to Top"
render_html("<div id='top-of-page'></div>")

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

if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False

# Theme Variables
bg_col = "#1e1e1e" if st.session_state.dark_mode else "#ffffff"
card_bg = "#262730" if st.session_state.dark_mode else "#f8f9fa"
border_col = "#333333" if st.session_state.dark_mode else "#e9ecef"
text_col = "#ffffff" if st.session_state.dark_mode else "#212529"
sub_text_col = "#a0a0a0" if st.session_state.dark_mode else "#6c757d"
accent_col = "#3b82f6"

# Custom CSS for Responsive Design and Typography
render_html(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif !important;
    }}
    
    .stButton>button {{
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border-radius: 8px !important;
        padding: 15px !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2) !important;
        width: 100% !important;
    }}
    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3) !important;
    }}
    
    /* Responsive Grids */
    .responsive-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
        width: 100%;
    }}
    
    .about-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
        width: 100%;
    }}
    
    /* Cards */
    .metric-card {{
        background-color: {card_bg};
        border: 1px solid {border_col};
        border-radius: 12px;
        padding: 25px 20px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        transition: transform 0.2s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-3px);
    }}
    .metric-title {{
        color: {sub_text_col};
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }}
    .metric-val {{
        color: {text_col};
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }}
    
    .about-card {{
        background-color: {card_bg};
        border-top: 4px solid {accent_col};
        border-radius: 12px;
        padding: 25px 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.03);
        color: {text_col};
    }}
    .about-card h4 {{
        color: {text_col};
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 12px;
        font-size: 1.15rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .about-card p {{
        color: {sub_text_col};
        font-size: 0.95rem;
        line-height: 1.5;
        margin-bottom: 0;
    }}
    .about-card ul {{
        margin-top: 10px;
        margin-bottom: 0;
        padding-left: 20px;
        color: {sub_text_col};
        font-size: 0.95rem;
    }}
    
    /* Guidance Box */
    .guidance-box {{
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
        border-left: 5px solid {accent_col};
        padding: 15px 25px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 40px;
        margin-top: 10px;
        animation: pulseGuidance 2s infinite;
    }}
    @keyframes pulseGuidance {{
        0% {{ box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.2); }}
        70% {{ box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }}
    }}
    .guidance-text {{
        font-size: 1.05rem;
        color: {text_col};
        margin: 0;
    }}
    .guidance-icon {{
        font-size: 1.8rem;
    }}

    /* Timeline */
    .timeline {{
        position: relative;
        max-width: 100%;
        margin: 20px 0;
        padding-left: 30px;
        border-left: 3px solid {accent_col};
    }}
    .timeline-item {{
        position: relative;
        margin-bottom: 25px;
    }}
    .timeline-item::before {{
        content: '';
        position: absolute;
        left: -39px;
        top: 5px;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        background-color: {accent_col};
        border: 3px solid {card_bg};
    }}
    .timeline-content {{
        background-color: {card_bg};
        padding: 15px 20px;
        border-radius: 8px;
        border: 1px solid {border_col};
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }}
    .timeline-title {{
        font-weight: 700;
        font-size: 1.05rem;
        color: {text_col};
        margin-bottom: 5px;
    }}
    .timeline-desc {{
        color: {sub_text_col};
        font-size: 0.95rem;
        margin: 0;
    }}
    
    /* Typography */
    .section-title {{
        font-weight: 800;
        color: {text_col};
        margin-top: 40px;
        margin-bottom: 20px;
        font-size: 1.8rem;
        letter-spacing: -0.5px;
    }}
    .hero-title {{
        font-weight: 900;
        font-size: clamp(2rem, 5vw, 3.2rem);
        color: {text_col};
        margin-bottom: 5px;
        line-height: 1.2;
        letter-spacing: -1px;
    }}
    .hero-subtitle {{
        font-size: clamp(1rem, 2vw, 1.2rem);
        color: {sub_text_col};
        margin-top: 0;
        margin-bottom: 35px;
        font-weight: 500;
    }}
    
    /* Container utilities */
    .styled-container {{
        background-color: {card_bg};
        border: 1px solid {border_col};
        border-radius: 16px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        height: 100%;
    }}
    
    /* Educational Layout Grid */
    .edu-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }}
    @media (max-width: 768px) {{
        .edu-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    
    /* Block default padding from Streamlit containers */
    .block-container {{
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }}
    
    /* Custom style for streamlit expander title */
    .streamlit-expanderHeader {{
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: {text_col} !important;
    }}
</style>
""")

# ===============================
# GLOBAL JS: SCROLL TO TOP BUTTON
# ===============================
components.html(
    f"""
    <script>
    const parent = window.parent.document;
    if (!parent.getElementById('scroll-to-top-btn')) {{
        const btn = parent.createElement('button');
        btn.id = 'scroll-to-top-btn';
        btn.innerHTML = '↑';
        
        const style = parent.createElement('style');
        style.innerHTML = `
            #scroll-to-top-btn {{
                position: fixed;
                bottom: 30px;
                right: 30px;
                width: 50px;
                height: 50px;
                background: {accent_col};
                color: white;
                border: none;
                border-radius: 50%;
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                cursor: pointer;
                font-size: 24px;
                display: none; /* hidden by default */
                align-items: center;
                justify-content: center;
                z-index: 999999;
                transition: all 0.3s ease;
                opacity: 0.8;
                font-weight: bold;
            }}
            #scroll-to-top-btn:hover {{
                opacity: 1;
                transform: translateY(-3px);
                box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            }}
        `;
        parent.head.appendChild(style);
        
        btn.onclick = () => {{
            const topEl = parent.getElementById('top-of-page');
            if (topEl) {{
                topEl.scrollIntoView({{behavior: 'smooth'}});
            }} else {{
                parent.querySelector('.main').scrollTo({{top: 0, behavior: 'smooth'}});
            }}
        }};
        
        parent.body.appendChild(btn);
        
        const scrollContainer = parent.querySelector('.main') || parent.window;
        scrollContainer.addEventListener('scroll', () => {{
            const scrollTop = scrollContainer.scrollTop || parent.window.scrollY;
            if (scrollTop > 400) {{
                btn.style.display = 'flex';
            }} else {{
                btn.style.display = 'none';
            }}
        }});
    }}
    </script>
    """,
    height=0,
    width=0
)

# ===============================
# SIDEBAR (Customer Input)
# ===============================
with st.sidebar:
    render_html(f"<h2 style='text-align: center; font-weight: 800; color: {accent_col}; margin-bottom: 20px;'>Customer Churn<br>Predictor</h2>")
    
    st.markdown("### 🎨 Appearance")
    dark_mode_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_mode_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode_toggle
        set_theme(dark_mode_toggle)
        st.rerun()

    st.markdown("---")
    st.markdown("### 📝 Input Form")

    with st.expander("👤 Demographics", expanded=True):
        gender = st.radio("Gender", ["Male", "Female"])
        senior = st.radio("Senior Citizen", ["Yes", "No"])
        partner = st.radio("Partner", ["Yes", "No"])
        dependents = st.radio("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        
    with st.expander("🛠️ Services", expanded=False):
        phone = st.radio("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        
    with st.expander("💳 Billing", expanded=False):
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.radio("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        m_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
        
        t_charges = tenure * m_charges

    render_html("<br>")
    predict_btn = st.button("🔮 Predict Churn Risk")
    if predict_btn:
        st.session_state.show_prediction = True


# ===============================
# 1. HERO / INTRODUCTION
# ===============================
render_html(f"<h1 class='hero-title'>AI Churn Analytics</h1>")
render_html(f"<p class='hero-subtitle'>Predict customer retention proactively using machine learning.</p>")

render_html(f"""
<div class='responsive-grid'>
    <div class='metric-card'>
        <div class='metric-title'>Total Customers</div>
        <div class='metric-val'>7,043</div>
    </div>
    <div class='metric-card'>
        <div class='metric-title'>Historical Churn Rate</div>
        <div class='metric-val' style='color: #ef4444;'>26.5%</div>
    </div>
    <div class='metric-card'>
        <div class='metric-title'>Prediction Accuracy</div>
        <div class='metric-val' style='color: #10b981;'>77.2%</div>
    </div>
    <div class='metric-card'>
        <div class='metric-title'>Business Impact</div>
        <div class='metric-val' style='color: {accent_col};'>5x ROI</div>
    </div>
</div>
""")

# ===============================
# 2. ABOUT THIS PREDICTOR
# ===============================
render_html(f"<div class='section-title'>💡 About This Predictor</div>")

render_html(f"""
<div class='styled-container' style='margin-bottom: 40px; border-left: 5px solid {accent_col}; padding: 30px;'>
    <p style='color: {sub_text_col}; font-size: 1.15rem; line-height: 1.7; margin: 0;'>
        This AI-powered <strong>customer churn predictor</strong> uses telecom customer data such as <strong>service usage</strong>, <strong>billing details</strong>, and <strong>contract information</strong> to identify customers who may leave a service. The system helps users and businesses quickly understand churn risk through <strong>real-time predictions</strong>, <strong>interactive dataset insights</strong>, and <strong>machine learning analysis</strong>, supporting smarter customer retention decisions.
    </p>
</div>
""")

# ===============================
# 3. DATASET INSIGHTS
# ===============================
render_html(f"<div class='section-title'>📊 Dataset Overview</div>")

c1, c2 = st.columns(2)

with c1:
    render_html(f"<div class='styled-container'>")
    churn_counts = df_historic['Churn'].value_counts().reset_index()
    churn_counts.columns = ['Status', 'Count']
    fig_pie = px.pie(
        churn_counts, 
        names='Status', 
        values='Count',
        title="Overall Churn Distribution",
        color='Status',
        color_discrete_map={'No': '#10b981', 'Yes': '#ef4444'},
        hole=0.45
    )
    fig_pie.update_layout(height=350, margin=dict(t=40, b=10, l=10, r=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': text_col, 'family': 'Inter'}, title_font={'size': 15, 'color': sub_text_col})
    st.plotly_chart(fig_pie, use_container_width=True)
    render_html("</div>")

with c2:
    render_html(f"<div class='styled-container'>")
    fig_contract = px.histogram(
        df_historic, 
        x="Contract", 
        color="Churn", 
        barmode="group",
        title="Churn by Contract Type",
        color_discrete_map={'No': '#10b981', 'Yes': '#ef4444'}
    )
    fig_contract.update_layout(height=350, margin=dict(t=40, b=10, l=10, r=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font={'color': text_col, 'family': 'Inter'}, title_font={'size': 15, 'color': sub_text_col}, legend_title_text='Churn')
    st.plotly_chart(fig_contract, use_container_width=True)
    render_html("</div>")


# ===============================
# 4. PREDICTION SECTION (MAIN FOCUS)
# ===============================
if st.session_state.show_prediction:
    render_html("<div id='prediction-section'></div>")
    
    if predict_btn:
        components.html(
            """
            <script>
            const parent = window.parent.document;
            const target = parent.getElementById('prediction-section');
            if (target) {
                target.scrollIntoView({behavior: 'smooth', block: 'start'});
            }
            </script>
            """,
            height=0,
            width=0
        )

    render_html(f"<div class='section-title' style='margin-top: 50px; font-size: 2.2rem;'>🎯 Prediction Result</div>")
    
    # Preprocessing
    data = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone, 'MultipleLines': multiple, 'InternetService': internet,
        'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': protection,
        'TechSupport': support, 'StreamingTV': tv, 'StreamingMovies': movies, 'Contract': contract,
        'PaperlessBilling': paperless, 'PaymentMethod': payment, 'MonthlyCharges': m_charges, 'TotalCharges': t_charges
    }
    df_input = pd.DataFrame([data])
    df_input['gender'] = df_input['gender'].map({'Male': 0, 'Female': 1})
    for c in ['Partner', 'SeniorCitizen', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df_input[c] = df_input[c].map({'Yes': 1, 'No': 0})
    ohe_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract']
    gen_enc = ohe_gen.transform(df_input[ohe_cols])
    gen_df = pd.DataFrame(gen_enc, columns=ohe_gen.get_feature_names_out(ohe_cols), index=df_input.index)
    pay_enc = ohe_pay.transform(df_input[['PaymentMethod']])
    pay_df = pd.DataFrame(pay_enc, columns=ohe_pay.get_feature_names_out(['PaymentMethod']), index=df_input.index)
    df_final = df_input.drop(columns=ohe_cols + ['PaymentMethod'])
    df_final = pd.concat([df_final, gen_df, pay_df], axis=1)
    df_final[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(df_final[['tenure', 'MonthlyCharges', 'TotalCharges']])
    df_final = df_final[model.feature_names_in_]

    # Prediction
    prob = model.predict_proba(df_final)[0][1]

    # Banner Styling
    if prob >= 0.5:
        bg_banner = "#fee2e2" if not st.session_state.dark_mode else "#7f1d1d"
        text_banner = "#991b1b" if not st.session_state.dark_mode else "#fca5a5"
        border_banner = "#ef4444"
        risk_text = "HIGH RISK OF CHURN"
        icon = "🚨"
    elif prob >= 0.3:
        bg_banner = "#fef3c7" if not st.session_state.dark_mode else "#78350f"
        text_banner = "#92400e" if not st.session_state.dark_mode else "#fcd34d"
        border_banner = "#f59e0b"
        risk_text = "MEDIUM RISK"
        icon = "⚠️"
    else:
        bg_banner = "#d1fae5" if not st.session_state.dark_mode else "#064e3b"
        text_banner = "#065f46" if not st.session_state.dark_mode else "#6ee7b7"
        border_banner = "#10b981"
        risk_text = "LOW RISK (SAFE)"
        icon = "✅"

    # Banner + Gauge Side by Side
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        render_html(f"""
        <div style='background-color: {bg_banner}; color: {text_banner}; border-left: 12px solid {border_banner}; padding: 35px; border-radius: 16px; height: 100%; display: flex; flex-direction: column; justify-content: center; box-shadow: 0 4px 15px rgba(0,0,0,0.05);'>
            <p style='margin: 0; font-size: 1.1rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;'>Model Assessment {icon}</p>
            <h2 style='margin: 10px 0; font-weight: 900; font-size: clamp(2rem, 4vw, 2.8rem); color: {text_banner}; line-height: 1.1;'>{risk_text}</h2>
            <p style='margin: 0; font-size: 1.2rem; font-weight: 500;'>Confidence Score: <strong>{prob:.1%}</strong> probability of leaving.</p>
        </div>
        """)
        
    with res_col2:
        render_html(f"<div class='styled-container' style='display: flex; align-items: center; justify-content: center; padding: 10px;'>")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': text_col},
                'bar': {'color': "rgba(0,0,0,0)"},
                'steps': [
                    {'range': [0, 30], 'color': "#10b981"},
                    {'range': [30, 50], 'color': "#f59e0b"},
                    {'range': [50, 100], 'color': "#ef4444"}
                ],
                'threshold': {'line': {'color': text_col, 'width': 4}, 'thickness': 0.75, 'value': prob * 100}
            }
        ))
        fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': text_col, 'family': 'Inter', 'size': 14})
        st.plotly_chart(fig_gauge, use_container_width=True)
        render_html("</div>")

    # ===============================
    # 5. MODEL PERFORMANCE SECTION
    # ===============================
    render_html(f"<div class='section-title' style='margin-top: 40px; font-size: 1.5rem;'>⚙️ Model Details</div>")
    
    perf_col1, perf_col2 = st.columns([1, 2])
    
    with perf_col1:
        render_html(f"""
        <div class='styled-container' style='padding: 25px;'>
            <h4 style='color: {sub_text_col}; margin-top: 0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;'>Algorithm</h4>
            <h2 style='color: {text_col}; font-weight: 800; font-size: 1.5rem; margin: 5px 0 20px 0;'>Random Forest</h2>
            <div style='display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 0.95rem;'>
                <span style='color: {sub_text_col};'>Accuracy:</span><span style='color: {text_col}; font-weight: 700;'>77.2%</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 0.95rem;'>
                <span style='color: {sub_text_col};'>Precision:</span><span style='color: {text_col}; font-weight: 700;'>75.4%</span>
            </div>
            <div style='display: flex; justify-content: space-between; font-size: 0.95rem;'>
                <span style='color: {sub_text_col};'>Recall:</span><span style='color: {text_col}; font-weight: 700;'>74.8%</span>
            </div>
        </div>
        """)

    with perf_col2:
        render_html(f"<div class='styled-container' style='padding: 15px;'>")
        metrics_df = pd.DataFrame({'Metric': ['Accuracy', 'Precision', 'Recall'], 'Score (%)': [77.2, 75.4, 74.8]})
        fig_metrics = px.bar(metrics_df, x='Score (%)', y='Metric', orientation='h', text='Score (%)', color='Metric', color_discrete_sequence=['#3b82f6', '#8b5cf6', '#14b8a6'])
        fig_metrics.update_layout(height=180, showlegend=False, margin=dict(l=10, r=20, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': text_col, 'family': 'Inter'}, xaxis=dict(showgrid=False, zeroline=False, visible=False), yaxis=dict(title=""))
        fig_metrics.update_traces(texttemplate='<b>%{text:.1f}%</b>', textposition='inside', insidetextanchor='middle')
        st.plotly_chart(fig_metrics, use_container_width=True)
        render_html("</div>")

    # Educational section removed to keep UI compact

# ===============================
# 7. FOOTER
# ===============================
render_html("<br><br>")
render_html("<hr style='border: 1px solid " + border_col + ";'>")

footer_html = f"""
<div style='display: flex; justify-content: space-between; align-items: center; padding: 10px 0; color: {sub_text_col}; font-family: "Inter", sans-serif; font-size: 0.95rem;'>
    <div style='display: flex; gap: 20px;'>
        <a href='#' style='color: {sub_text_col}; text-decoration: none; transition: color 0.2s ease;' onmouseover="this.style.color='{text_col}'" onmouseout="this.style.color='{sub_text_col}'">Contact Us</a>
        <a href='#' style='color: {sub_text_col}; text-decoration: none; transition: color 0.2s ease;' onmouseover="this.style.color='{text_col}'" onmouseout="this.style.color='{sub_text_col}'">Help</a>
    </div>
    <div style='font-weight: 600;'>
        &copy; RP2
    </div>
</div>
"""
render_html(footer_html)