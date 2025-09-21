import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# =======================
# =======================
# Custom CSS for Full Visibility
# =======================
st.markdown("""
<style>
/* =======================
   App Background
   ======================= */
.stApp {
    background: linear-gradient(135deg, #1c1f2a, #283046, #31405a); /* dark gradient but lighter for readability */
    color: white;
}

/* =======================
   Card, Inputs, and DataFrames
   ======================= */
.stMarkdown, .stDataFrame, .stSelectbox, .stNumberInput, .stButton {
    background-color: #2c3a50 !important; /* slightly lighter dark card */
    color: white !important;
    border-radius: 12px;
    padding: 10px;
    font-weight: bold;
}

/* =======================
   Titles
   ======================= */
h1, h2, h3, h4 {
    color: #facc15;  /* gold */
    font-weight: bold;
}

/* =======================
   Sidebar Styling
   ======================= */
section[data-testid="stSidebar"] {
    background: #1e293b;  /* lighter dark for sidebar */
    color: white;
}

/* Sidebar selectbox placeholder and text */
div[data-baseweb="select"] > div > div > div > span {
    color: #facc15 !important;  /* gold placeholder */
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

/* =======================
   Buttons
   ======================= */
button {
    background-color: #facc15 !important;  /* gold */
    color: black !important;
    font-weight: bold !important;
    border-radius: 8px !important;
}

/* =======================
   DataFrame table text
   ======================= */
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
# Example Page Title
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
    """
    <h2 style='color:#facc15;'>Navigation</h2>
    """,
    unsafe_allow_html=True
)

option = st.sidebar.selectbox(
    "Choose an option",
    ["Dashboard", "Model Training", "Predict Churn"]
)

# =======================
# Dashboard Section
# =======================
if option == "Dashboard":
    st.title("📈 Dashboard")

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    # Chart 1
    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x='Churn', data=df, palette="viridis", ax=ax)
        st.pyplot(fig)

    # Chart 2
    with col2:
        st.subheader("Contract Types")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(x='Contract', data=df, palette="mako", ax=ax)
        st.pyplot(fig)

 
   # Chart 3
    with col3:
         st.subheader("Internet Service")
         fig, ax = plt.subplots(figsize=(5, 4))

    # Plot
         sns.countplot(x='InternetService', data=df, palette="rocket", ax=ax)

    # Apply short labels
         short_labels = {
                    'Electronic check': 'E-Check',
                     'Mailed check': 'M-Check',
                    'Bank transfer (automatic)': 'Bank Transfer',
                     'Credit card (automatic)': 'Credit Card'
    }

    # Replace the tick labels
         ax.set_xticklabels([short_labels.get(label.get_text(), label.get_text()) for label in ax.get_xticklabels()])

         st.pyplot(fig)
    
  
# Chart 4
    with col4:
         st.subheader("Payment Method")
         fig, ax = plt.subplots(figsize=(5, 4))
         sns.countplot(x=df['PaymentMethod'].map(short_labels), palette="coolwarm", ax=ax)
         plt.xticks(rotation=45)
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
        "<h2 style='white-space: nowrap; color: white;'>🔮 Predict Churn for a New Customer</h1>",
        unsafe_allow_html=True
    )

    model = joblib.load("churn_model.pkl")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0, value=50)
    total_charges = st.number_input("Total Charges", min_value=0, value=500)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Echeck", "M check", "BT (auto)", "CC(auto)"])
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

        new_df_encoded = pd.get_dummies(new_data)
        df_encoded = pd.get_dummies(df, drop_first=True)
        X = df_encoded.drop("Churn_Yes", axis=1)

        new_df_encoded = new_df_encoded.reindex(columns=X.columns, fill_value=0)

        prob = model.predict_proba(new_df_encoded)[:, 1][0]
        prediction = "Yes" if prob > 0.5 else "No"

        st.subheader("📢 Prediction Result")
        st.markdown(f"<h3>Churn Prediction: <span style='color:gold'>{prediction}</span></h3>", unsafe_allow_html=True)
        st.write(f"**Probability of Churn:** {prob:.2f}")
