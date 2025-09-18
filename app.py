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

        # Metrics
        churn_counts = df['Churn'].value_counts()
        churn_rate = churn_counts.get("Yes", 0) / churn_counts.sum() * 100

        acc = f1 = None  # placeholders if user hasn't trained model yet

        col1, col2 = st.columns(2)
        col1.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
        col2.metric("Month-to-Month Churn", f"{(df[df['Contract']=='Month-to-month']['Churn']=='Yes').mean()*100:.1f}%")

        # Churn Distribution Pie
        fig1, ax1 = plt.subplots()
        ax1.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90, colors=['#2ca02c','#d62728'])
        ax1.set_title("Churn Distribution")
        st.pyplot(fig1)

        # Contract vs Churn Bar
        fig2, ax2 = plt.subplots()
        sns.countplot(x="Contract", hue="Churn", data=df, palette=["#2ca02c","#d62728"], ax=ax2)
        plt.xticks(rotation=30)
        ax2.set_title("Contract Type vs Churn")
        st.pyplot(fig2)

        # Monthly Charges Boxplot
        fig3, ax3 = plt.subplots()
        sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette=["#2ca02c","#d62728"], ax=ax3)
        ax3.set_title("Monthly Charges vs Churn")
        st.pyplot(fig3)

        # Business Insights
        with st.expander("📌 Key Business Insights"):
            month2month_churn = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack().fillna(0).get("Yes", {}).get("Month-to-month", 0)*100
            high_charges_churn = df[df["MonthlyCharges"] > df["MonthlyCharges"].median()]["Churn"].value_counts(normalize=True).get("Yes",0)*100
            avg_tenure_churned = df[df["Churn"]=="Yes"]["tenure"].mean()
            avg_tenure_stayed = df[df["Churn"]=="No"]["tenure"].mean()

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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

        model = XGBClassifier(random_state=42, eval_metric="logloss")
        model.fit(X_train_res, y_train_res)

        # Predictions
        y_probs = model.predict_proba(X_test)[:, 1]
        threshold = 0.27
        y_pred = (y_probs > threshold).astype(int)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        col1, col2 = st.columns(2)
        col1.metric("Model Accuracy", f"{acc:.2f}")
        col2.metric("F1-score (Churn)", f"{f1:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig4, ax4 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"], ax=ax4)
        ax4.set_title("Confusion Matrix")
        st.pyplot(fig4)

        # Feature Importance
        importance = pd.Series(model.feature_importances_, index=X.columns)
        with st.expander("🔎 Top 10 Feature Importance"):
            fig5, ax5 = plt.subplots()
            importance.nlargest(10).plot(kind='barh', ax=ax5, color="#1f77b4")
            ax5.set_title("Top 10 Important Features")
            st.pyplot(fig5)

    # ---------------------------
    # Predict New Customer Section
    # ---------------------------
    elif selected == "Predict Churn":
        
        st.subheader("📌 Predict Churn for a New Customer")

# Make sure the model exists
try:
    model
except NameError:
    st.warning("⚠️ Please train the model first in 'Model Training' section.")
else:
    user_input = {}
    for col in X.columns:
        if df_encoded[col].nunique() <= 2:  # binary
            user_input[col] = st.selectbox(col, [0, 1])
        else:  # numeric
            user_input[col] = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))

    user_df = pd.DataFrame([user_input])
    user_probs = model.predict_proba(user_df)[:, 1][0]
    user_pred = 1 if user_probs > threshold else 0

    st.markdown(f"### 🔮 Prediction: {'⚠️ Churn' if user_pred==1 else '✅ Not Churn'}")
    st.markdown(f"**Churn Probability:** {user_probs:.2f}")
