import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download
import joblib


MODEL_REPO = "Yashwanth-R19/loan-default-classifier"
FEATURES = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio", "Education",
    "EmploymentType", "MaritalStatus", "HasMortgage", "HasDependents",
    "LoanPurpose", "HasCoSigner",
]


def field_heading(title, detail):
    """Show a plain-language field title and entry hint above each widget."""
    st.markdown(f"**{title}**")
    st.caption(detail)

st.set_page_config(page_title="CreditLens | Loan Risk Assessment", page_icon="◈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
.stApp { background: #f5f7fb; color: #17243a; }
[data-testid="stHeader"] { background: rgba(245,247,251,.88); }
[data-testid="stSidebar"] { background: #101d32; }
[data-testid="stSidebar"] * { color: #eaf0f9 !important; }
h1,h2,h3 { font-family: 'Manrope', sans-serif; letter-spacing: -.035em; }
p, label, div { font-family: 'DM Sans', sans-serif; }
.hero { padding: 1.6rem 0 1rem; }
.eyebrow { color:#3579e8; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
.hero h1 { font-size:2.55rem; line-height:1.1; margin:.55rem 0; color:#142238; }
.hero p { color:#64748b; max-width:700px; font-size:1.02rem; }
.panel { background:white; border:1px solid #e6ebf2; border-radius:16px; padding:1.1rem 1.25rem; }
.risk-card { background:linear-gradient(135deg,#14243b,#203e60); color:#fff; padding:1.5rem; border-radius:18px; margin:.6rem 0 1rem; }
.risk-card h2 { color:#fff; margin:.3rem 0; }
.risk-card p { color:#c2d0e2; margin-bottom:0; }
.small-note { color:#6b778a; font-size:.82rem; }
div.stButton > button { background:#286fe5; color:white; border:0; border-radius:10px; padding:.65rem 1rem; font-weight:700; }
div.stButton > button:hover { background:#1d5ec7; color:white; border:0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading the published model…")
def load_model():
    path = hf_hub_download(repo_id=MODEL_REPO, filename="model.pkl")
    return joblib.load(path)


with st.sidebar:
    st.markdown("## CreditLens")
    st.caption("Pretrained model · CPU inference")
    st.markdown("---")
    st.markdown("**About this demo**")
    st.write("The model estimates loan default risk from applicant and loan characteristics. It is a research/demo aid, not a loan approval decision.")
    st.markdown("[Model card on Hugging Face](https://huggingface.co/Yashwanth-R19/loan-default-classifier)")

st.title("CreditLens — Loan Risk Assessment")

left, right = st.columns([1.5, 1], gap="large")
with left:
    st.markdown("### Applicant and loan details")
    with st.container(border=True):
        a, b = st.columns(2)
        with a:
            field_heading("Applicant age", "Enter the applicant’s age in years.")
            age = st.number_input("Applicant age", 18, 100, 34, label_visibility="collapsed")
            field_heading("Annual income", "Enter gross income per year, in the same currency units as the loan amount.")
            income = st.number_input("Annual income", min_value=0, value=72000, step=1000, label_visibility="collapsed")
            field_heading("Requested loan amount", "Enter the total amount the applicant wants to borrow.")
            loan_amount = st.number_input("Requested loan amount", min_value=0, value=18000, step=500, label_visibility="collapsed")
            field_heading("Credit score", "Enter the applicant’s credit bureau score, if available.")
            credit_score = st.number_input("Credit score", min_value=0, max_value=1000, value=690, label_visibility="collapsed")
            field_heading("Employment length", "Enter how many months the applicant has been employed.")
            months_employed = st.number_input("Employment length", min_value=0, value=48, step=6, label_visibility="collapsed")
            field_heading("Existing credit lines", "Count the applicant’s current open credit accounts or lines.")
            num_credit_lines = st.number_input("Existing credit lines", min_value=0, value=4, label_visibility="collapsed")
            field_heading("Interest rate", "Enter the annual interest rate as a percentage, for example 10.5.")
            interest_rate = st.number_input("Interest rate", min_value=0.0, value=10.5, step=0.25, label_visibility="collapsed")
            field_heading("Loan term", "Enter the repayment period in months.")
            loan_term = st.number_input("Loan term", min_value=1, value=36, step=12, label_visibility="collapsed")
        with b:
            field_heading("Debt-to-income ratio", "Enter monthly debt divided by monthly income as a decimal: 0.28 means 28%.")
            dti = st.number_input("Debt-to-income ratio", min_value=0.0, max_value=1.0, value=0.28, step=0.01, format="%.2f", label_visibility="collapsed")
            field_heading("Education level", "Choose the applicant’s highest completed education level.")
            education = st.selectbox("Education level", ["High School", "Bachelor's", "Master's", "PhD"], label_visibility="collapsed")
            field_heading("Employment type", "Choose the applicant’s current work arrangement.")
            employment = st.selectbox("Employment type", ["Full-time", "Part-time", "Self-employed", "Unemployed"], label_visibility="collapsed")
            field_heading("Marital status", "Choose the applicant’s current marital status.")
            marital = st.selectbox("Marital status", ["Single", "Married", "Divorced"], label_visibility="collapsed")
            field_heading("Loan purpose", "Choose the main reason the applicant is requesting the loan.")
            purpose = st.selectbox("Loan purpose", ["Auto", "Business", "Education", "Home", "Other"], label_visibility="collapsed")
            field_heading("Mortgage", "Select Yes if the applicant currently has a mortgage.")
            mortgage = st.selectbox("Mortgage", ["No", "Yes"], label_visibility="collapsed")
            field_heading("Dependents", "Select Yes if the applicant supports one or more dependents.")
            dependents = st.selectbox("Dependents", ["No", "Yes"], label_visibility="collapsed")
            field_heading("Co-signer", "Select Yes if another person will share responsibility for repayment.")
            cosigner = st.selectbox("Co-signer", ["No", "Yes"], label_visibility="collapsed")
    run = st.button("Estimate default risk", type="primary", use_container_width=True)

with right:
    st.markdown("### Risk estimate")
    if run:
        row = pd.DataFrame([{
            "Age": age,
            "Income": income,
            "LoanAmount": loan_amount,
            "CreditScore": credit_score,
            "MonthsEmployed": months_employed,
            "NumCreditLines": num_credit_lines,
            "InterestRate": interest_rate,
            "LoanTerm": loan_term,
            "DTIRatio": dti,
            "Education": education,
            "EmploymentType": employment,
            "MaritalStatus": marital,
            "HasMortgage": mortgage,
            "HasDependents": dependents,
            "LoanPurpose": purpose,
            "HasCoSigner": cosigner,
        }], columns=FEATURES)
        try:
            model = load_model()
            probability = float(model.predict_proba(row)[0, 1])
            risk = "Higher estimated risk" if probability >= 0.5 else "Lower estimated risk"
            st.markdown(f"""
            <div class="risk-card">
              <div class="eyebrow" style="color:#8fbaff">Estimated probability of default</div>
              <h2>{probability:.1%}</h2>
              <p>{risk} · model threshold 50%</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(probability, text="Estimated default probability")
            st.caption("A higher score means the model estimates greater default risk. This is not an approval recommendation.")
        except Exception as exc:
            st.error("The model could not be loaded or run. Check your internet connection and installed dependencies, then restart the app.")
            st.caption(f"Technical detail: {exc}")
    else:
        st.markdown("""
        <div class="panel">
          <div class="eyebrow">Ready when you are</div>
          <h3>Enter a profile to get an estimate</h3>
          <p class="small-note">The model downloads once from Hugging Face on first use, then is cached for this session.</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown(" ")
    st.markdown("<div class='panel'><b>How to read this</b><p class='small-note'>This demo model reports limited precision and recall on its published test set. Do not use its output to approve or deny real applicants.</p></div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Demo only · Uses a published XGBoost/scikit-learn pipeline from Hugging Face · No applicant data is saved by this app")
