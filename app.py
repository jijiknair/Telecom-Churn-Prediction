import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
from streamlit_option_menu import option_menu

# ---------------------------
# Page config & background
# ---------------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(180deg, #f0f3f8 0%, #e6eefc 100%); }
.card { background-color: #fff; padding: 20px; border-radius: 12px; box-shadow: 2px 4px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }
.metric-container { background-color: #fff; padding: 15px; border-radius: 10px; box-shadow: 2px 4px 8px rgba(0,0,0,0.1); text-align: center; font-size: 18px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Dashboard", "Model Training", "Predict Churn"],
        icons=["bar-chart","gear","person-lines-fill"],
        menu_icon="cast",
        default_index=0
    )

# ---------------------------
# File upload
# ---------------------------
st.title("📊 Customer Churn Prediction & Dashboard")
uploaded_file = st.file_uploader("Upload your Telco Churn dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Preprocess
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df_encoded = pd.get_dummies(df, drop_first=True)

    # ---------------------------
    # Dashboard
    # ---------------------------
    if selected == "Dashboard":
        st.subheader("📊 Churn Dashboard")

        churn_counts = df['Churn'].value_counts()
        churn_rate = churn_counts.get("Yes", 0) / churn_counts.sum() * 100
        month2month_churn = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack().fillna(0).get("Yes", {}).get("Month-to-month", 0)*100

        col1, col2, col3 = st.columns(3)
        with col1: st.markdown(f"<div class='metric-container'><b>Overall Churn Rate</b><br>{churn_rate:.1f}%</div>", unsafe_allow_html=True)
        with col2: st.markdown(f"<div class='metric-container'><b>Month-to-Month Churn</b><br>{month2month_churn:.1f}%</div>", unsafe_allow_html=True)
        with col3: st.markdown(f"<div class='metric-container'><b>High-paying Churn Rate</b><br>{(df[df['MonthlyCharges'] > df['MonthlyCharges'].median()]['Churn']=='Yes').mean()*100:.1f}%</div>", unsafe_allow_html=True)

        # Charts with equal width
        col1, col2 = st.columns([1,1])
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            fig1, ax1 = plt.subplots()
            ax1.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90, colors=['#2ca02c','#d62728'])
            ax1.set_title("Churn Distribution")
            st.pyplot(fig1)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            fig2, ax2 = plt.subplots()
            sns.countplot(x="Contract", hue="Churn", data=df, palette=["#2ca02c","#d62728"], ax=ax2)
            plt.xticks(rotation=30)
            ax2.set_title("Contract Type vs Churn")
            st.pyplot(fig2)
            st.markdown("</div>", unsafe_allow_html=True)

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

    # ---------------------------
    # Model Training
    # ---------------------------
    elif selected == "Model Training":
        st.subheader("⚙️ Model Training")

        X = df_encoded.drop('Churn_Yes', axis=1)
        y = df_encoded['Churn_Yes']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)

        model = XGBClassifier(random_state=42, eval_metric="logloss")
        model.fit(X_train_res, y_train_res)

        st.session_state['model'] = model
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['threshold'] = 0.27

        y_pred = (model.predict_proba(X_test)[:,1] > 0.27).astype(int)
        st.metric("Model Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        st.metric("F1-score (Churn)", f"{f1_score(y_test, y_pred):.2f}")

        with st.expander("🔎 Top 10 Feature Importance"):
            importance = pd.Series(model.feature_importances_, index=X.columns)
            fig, ax = plt.subplots()
            importance.nlargest(10).plot(kind='barh', ax=ax, color="#1f77b4")
            ax.set_title("Top 10 Important Features")
            st.pyplot(fig)

    # ---------------------------
    # Predict Churn
    # ---------------------------
    elif selected == "Predict Churn":
        st.subheader("📌 Predict Churn for a New Customer")

        if 'model' not in st.session_state:
            st.warning("⚠️ Train the model first!")
        else:
            model = st.session_state['model']
            threshold = st.session_state['threshold']
            X_columns = st.session_state['X_test'].columns

            user_input = {}
            for col in X_columns:
                if "_Yes" in col or "_No" in col or "_Male" in col or "_Female" in col:
                    user_input[col] = st.selectbox(col, [0,1])
                else:
                    user_input[col] = st.slider(col, float(df_encoded[col].min()), float(df_encoded[col].max()), float(df_encoded[col].mean()))
            user_df = pd.DataFrame([user_input])
            prob = model.predict_proba(user_df)[:,1][0]
            pred = "⚠️ Churn" if prob>threshold else "✅ Not Churn"
            st.markdown(f"### 🔮 Prediction: {pred}")
            st.markdown(f"**Churn Probability:** {prob:.2f}")
