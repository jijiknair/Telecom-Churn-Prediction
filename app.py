import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# =======================
# Custom CSS for Full Visibility
# =======================
st.markdown("""
<style>
/* App Background */
.stApp {
    background: linear-gradient(135deg, #1c1f2a, #283046, #31405a);
    color: white;
}

/* Cards, Inputs, and DataFrames */
.stMarkdown, .stDataFrame, .stSelectbox, .stNumberInput, .stButton {
    background-color: #2c3a50 !important;
    color: white !important;
    border-radius: 12px;
    padding: 10px;
    font-weight: bold;
}

/* Titles */
h1, h2, h3, h4 {
    color: #facc15;
    font-weight: bold;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1e293b;
    color: white;
}

/* Sidebar selectbox placeholder and text */
div[data-baseweb="select"] > div > div > div > span {
    color: #facc15 !important;
    font-weight: bold;
}

/* Sidebar selectbox label */
.stSelectbox label {
    color: #facc15 !important;
    font-weight: bold !important;
}

/* NumberInput label */
.stNumberInput label {
    color: #facc15 !important;
    font-weight: bold !important;
}

/* Buttons */
button {
    background-color: #facc15 !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 8px !important;
}

/* DataFrame table text */
.stDataFrame div.row_heading, .stDataFrame div.column_heading {
    color: white !important;
    font-weight: bold;
}
.stDataFrame div.cell {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# =======================
# Page Title
# =======================
st.markdown(
    "<h1 style='text-align:center; color:#facc15; font-size:48px;'>📊 Customer Churn Prediction</h1>",
    unsafe_allow_html=True
)

# =======================
# Load Dataset
# =======================
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    return df

df = load_data()

# =======================
# Sidebar Navigation
# =======================
st.sidebar.markdown(
    "<h2 style='color:#facc15;'>Navigation</h2>",
    unsafe_allow_html=True
)

option = st.sidebar.selectbox(
    "Choose an option",
    ["Dashboard", "Model Training", "Predict Churn"]
)

# =======================
# Short Labels for Charts
# =======================
payment_labels = {
    'Electronic check': 'E-Check',
    'Mailed check': 'M-Check',
    'Bank transfer (automatic)': 'Bank Transfer',
    'Credit card (automatic)': 'Credit Card'
}

# =======================
# Dashboard Section
# =======================
if option == "Dashboard":
    st.title("📈 Dashboard")
    # KPI Calculations
    total_customers = df.shape[0]
    total_churned = df[df['Churn'] == 'Yes'].shape[0]
    churn_rate = total_churned / total_customers * 100
    avg_monthly = df['MonthlyCharges'].mean()

    # Inject CSS for KPI styling      
    st.markdown("""
    <style>
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin-bottom: 20px;
    }
     .kpi-card {
        height: 150px;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.3);
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
        transform: scale(1.05);
        box-shadow: 0px 6px 18px rgba(255, 215, 0, 0.6); /* golden glow */
    }
    .kpi-card h3 {
        font-size: 18px;
        margin-bottom: 8px;
    }
    .kpi-card p {
        font-size: 22px;
        font-weight: bold;
        margin: 0;
    }
    .customers { background-color: #2ecc71; }   /* Green */
    .churned { background-color: #e74c3c; }     /* Red */
    .rate { background-color: #f39c12; }        /* Orange */
    .monthly { background-color: #3498db; }     /* Blue */
    </style>
""", unsafe_allow_html=True)

# Render KPI cards
st.markdown(f"""
<div class="kpi-container">
    <div class="kpi-card customers">
        <h3>Total Customers</h3>
        <p>{total_customers}</p>
    </div>
    <div class="kpi-card churned">
        <h3>Total Churned</h3>
        <p>{total_churned}</p>
    </div>
    <div class="kpi-card rate">
        <h3>Churn Rate</h3>
        <p>{churn_rate:.2f}%</p>
    </div>
    <div class="kpi-card monthly">
        <h3>Avg Monthly Charges</h3>
        <p>${avg_monthly:.2f}</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

    
    # ==============================
    # Charts Section (2 x 2 layout)
    # ==============================
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # Chart 1: Churn Distribution
    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x='Churn', data=df, palette="viridis", ax=ax)
        st.pyplot(fig)

    # Chart 2: Contract Types
    with col2:
        st.subheader("Contract Types")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x='Contract', data=df, palette="mako", ax=ax)
        plt.xticks(rotation=0)
        st.pyplot(fig)

    # Chart 3: Internet Service
    with col3:
        st.subheader("Internet Service")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x='InternetService', data=df, palette="rocket", ax=ax)
        ax.set_xticklabels([label.get_text() for label in ax.get_xticklabels()], rotation=0)
        st.pyplot(fig)

    # Chart 4: Payment Method
    with col4:
        st.subheader("Payment Method")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x=df['PaymentMethod'].map(payment_labels), palette="coolwarm", ax=ax)
        plt.xticks(rotation=0)
        st.pyplot(fig)

# =======================
# Model Training Section
# =======================
elif option == "Model Training":
    st.title("🤖 Model Training")

    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop("Churn_Yes", axis=1)
    y = df_encoded["Churn_Yes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBClassifier(eval_metric="logloss")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.subheader("📊 Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    joblib.dump(model, "churn_model.pkl")
    st.success("✅ Model trained and saved successfully!")

# =======================
# Predict Churn Section
# =======================
elif option == "Predict Churn":
    st.markdown(
        "<h2 style='white-space: nowrap; color: #facc15;'>🔮 Predict Churn for a New Customer</h2>",
        unsafe_allow_html=True
    )

    model = joblib.load("churn_model.pkl")

    # Input fields
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0, value=50)
    total_charges = st.number_input("Total Charges", min_value=0, value=500)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["E-Check", "M-Check", "Bank Transfer", "Credit Card"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    if st.button("Predict Churn"):
        new_data = pd.DataFrame({
            "gender": [gender],
            "SeniorCitizen": [senior],
            "tenure": [tenure],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
            "Contract": [contract],
            "PaymentMethod": [payment],
            "InternetService": [internet]
        })

        # Encode new data and align columns with training set
        new_df_encoded = pd.get_dummies(new_data)
        df_encoded = pd.get_dummies(df, drop_first=True)
        X_columns = df_encoded.drop("Churn_Yes", axis=1).columns
        new_df_encoded = new_df_encoded.reindex(columns=X_columns, fill_value=0)

        # Prediction
        prob = model.predict_proba(new_df_encoded)[:, 1][0]
        prediction = "Yes" if prob > 0.5 else "No"

        st.subheader("📢 Prediction Result")
        st.markdown(
            f"<h3>Churn Prediction: <span style='color:gold'>{prediction}</span></h3>",
            unsafe_allow_html=True
        )
        st.markdown(f"**Probability of Churn:** {prob:.2f}")
