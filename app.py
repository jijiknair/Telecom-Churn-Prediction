import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
from streamlit_option_menu import option_menu

# ---------------------------
# Page config & background
# ---------------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# Custom CSS for solid gradient background and card styling
st.markdown("""
    <style>
    /* Background */
    .stApp {
        background: linear-gradient(180deg, #f0f3f8 0%, #e6eefc 100%);
    }

    /* Card style for charts and metrics */
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 2px 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* Metric style */
    .metric-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        font-size: 18px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar Navigation
# ---------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Dashboard", "Model Training", "Predict Churn"],
        icons=["bar-chart","gear","person-lines-fill"],
        menu_icon="cast",
        default_index=0
    )

# ---------------------------
# Upload CSV
# ---------------------------
st.title("📊 Customer Churn Prediction & Dashboard")
uploaded_file = st.file_uploader("Upload your Telco Churn dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preview dataset
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # Data Cleaning & Encoding
    # ---------------------------
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df_encoded = pd.get_dummies(df, drop_first=True)

    # ---------------------------
    # Dashboard Section
    # ---------------------------
    if selected == "Dashboard":
        st.subheader("📊 Churn Dashboard")

        churn_counts = df['Churn'].value_counts()
        churn_rate = churn_counts.get("Yes", 0) / churn_counts.sum() * 100
        month2month_churn = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack().fillna(0).get("Yes", {}).get("Month-to-month", 0)*100

        # Metrics at the top
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown(f"<div class='metric-container'><b>Overall Churn Rate</b><br>{churn_rate:.1f}%</div>", unsafe_allow_html=True)
        with col2: st.markdown(f"<div class='metric-container'><b>Month-to-Month Churn</b><br>{month2month_churn:.1f}%</div>", unsafe_allow_html=True)
        with col3: st.markdown(f"<div class='metric-container'><b>High-paying Churn Rate</b><br>{(df[df['MonthlyCharges'] > df['MonthlyCharges'].median()]['Churn']=='Yes').mean()*100:.1f}%</div>", unsafe_allow_html=True)

        # Arrange charts in two columns
        col1, col2 = st.columns(2)

        # Churn Pie Chart
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            fig1, ax1 = plt.subplots()
            ax1.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90, colors=['#2ca02c','#d62728'])
            ax1.set_title("Churn Distribution")
            st.pyplot(fig1)
            st.markdown("</div>", unsafe_allow_html=True)

        # Contract vs Churn
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots()
            sns.countplot(x="Contract", hue="Churn", data=df, palette=["#2ca02c","#d62728"], ax=ax2)
            plt.xticks(rotation=30)
            ax2.set_title("Contract Type vs Churn")
            st.pyplot(fig2)
            st.markdown("</div>", unsafe_allow_html=True)

        # Monthly Charges Boxplot and Tenure
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            fig3, ax3 = plt.subplots()
            sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette=["#2ca02c","#d62728"], ax=ax3)
            ax3.set_title("Monthly Charges vs Churn")
            st.pyplot(fig3)
            st.markdown("</div>", unsafe_allow_html=True)

        with col4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            fig4, ax4 = plt.subplots()
            sns.boxplot(x="Churn", y="tenure", data=df, palette=["#2ca02c","#d62728"], ax=ax4)
            ax4.set_title("Tenure vs Churn")
            st.pyplot(fig4)
            st.markdown("</div>", unsafe_allow_html=True)

        # Insights
        with st.expander("📌 Key Business Insights"):
            avg_tenure_churned = df[df["Churn"]=="Yes"]["tenure"].mean()
            avg_tenure_stayed = df[df["Churn"]=="No"]["tenure"].mean()
            high_charges_churn = df[df["MonthlyCharges"] > df["MonthlyCharges"].median()]["Churn"].value_counts(normalize=True).get("Yes",0)*100

            st.markdown(f"1️⃣ **Overall churn rate:** {churn_rate:.1f}%")
            st.markdown(f"2️⃣ **Month-to-Month contracts churn:** {month2month_churn:.1f}%")
            st.markdown(f"3️⃣ **High-paying customers churn rate:** {high_charges_churn:.1f}%")
            st.markdown(f"4️⃣ **Average tenure:** Churned {avg_tenure_churned:.1f} months, Loyal {avg_tenure_stayed:.1f} months")
            st.info("💡 Focus retention on month-to-month, high-charge, short-tenure customers.")

# ---------------------------
# Model Training Section
# ---------------------------
elif selected == "Model Training":
    st.subheader("⚙️ Model Training")

    X = df_encoded.drop('Churn_Yes', axis=1)
    y = df_encoded['Churn_Yes']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = XGBClassifier(random_state=42, eval_metric="logloss")
    model.fit(X_train_res, y_train_res)

    # Save model in session_state
    st.session_state['model'] = model
    st.session_state['X_test'] = X_test
    st.session_state['y_test'] = y_test
    st.session_state['threshold'] = 0.27

    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > st.session_state['threshold']).astype(int)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.metric("Model Accuracy", f"{acc:.2f}")
    st.metric("F1-score (Churn)", f"{f1:.2f}")

    # Feature Importance
    with st.expander("🔎 Top 10 Feature Importance"):
        importance = pd.Series(model.feature_importances_, index=X.columns)
        fig, ax = plt.subplots()
        importance.nlargest(10).plot(kind='barh', ax=ax, color="#1f77b4")
        ax.set_title("Top 10 Important Features")
        st.pyplot(fig)

# ---------------------------
# Predict Churn Section
# ---------------------------
elif selected == "Predict Churn":
    st.subheader("📌 Predict Churn for a New Customer")

    if 'model' not in st.session_state:
        st.warning("⚠️ Please train the model first in 'Model Training' section.")
    else:
        model = st.session_state['model']
        threshold = st.session_state['threshold']
        X_columns = st.session_state['X_test'].columns

        user_input = {}
        binary_mappings = {
            "gender_Male": ["Female", "Male"],
            "Partner_Yes": ["No", "Yes"],
            "Dependents_Yes": ["No", "Yes"],
            "PhoneService_Yes": ["No", "Yes"],
            "MultipleLines_No phone service": ["No phone service", "No"],
            "MultipleLines_Yes": ["No", "Yes"],
            "InternetService_Fiber optic": ["DSL", "Fiber optic"],
            "InternetService_No": ["DSL", "No"],
            "OnlineSecurity_No internet service": ["No internet service", "No"],
            "OnlineSecurity_Yes": ["No", "Yes"],
            "OnlineBackup_No internet service": ["No internet service", "No"],
            "OnlineBackup_Yes": ["No", "Yes"],
            "DeviceProtection_No internet service": ["No internet service", "No"],
            "DeviceProtection_Yes": ["No", "Yes"],
            "TechSupport_No internet service": ["No internet service", "No"],
            "TechSupport_Yes": ["No", "Yes"],
            "StreamingTV_No internet service": ["No internet service", "No"],
            "StreamingTV_Yes": ["No", "Yes"],
            "StreamingMovies_No internet service": ["No internet service", "No"],
            "StreamingMovies_Yes": ["No", "Yes"],
            "Contract_One year": ["Month-to-month", "One year"],
            "Contract_Two year": ["Month-to-month", "Two year"],
            "PaperlessBilling_Yes": ["No", "Yes"],
            "PaymentMethod_Credit card (automatic)": ["Electronic check", "Credit card (automatic)"],
            "PaymentMethod_Electronic check": ["Credit card (automatic)", "Electronic check"],
            "PaymentMethod_Mailed check": ["Credit card (automatic)", "Mailed check"]
        }

        for col in X_columns:
            if col in binary_mappings:
                user_input[col] = st.selectbox(col, binary_mappings[col])
                user_input[col] = 1 if user_input[col] == binary_mappings[col][1] else 0
            else:
                user_input[col] = st.slider(
                    col, float(df_encoded[col].min()), float(df_encoded[col].max()), float(df_encoded[col].mean())
                )

        user_df = pd.DataFrame([user_input])
        user_probs = model.predict_proba(user_df)[:, 1][0]
        user_pred = 1 if user_probs > threshold else 0

        st.markdown(f"### 🔮 Prediction: {'⚠️ Churn' if user_pred==1 else '✅ Not Churn'}")
        st.markdown(f"**Churn Probability:** {user_probs:.2f}")

