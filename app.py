import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)

# Title
st.title("📊 Customer Churn Prediction & Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your Telco Churn dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # =====================
    # Data Cleaning
    # =====================
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Encode categorical
    df_encoded = pd.get_dummies(df, drop_first=True)

    # =====================
    # Dashboard Visualizations
    # =====================
    st.subheader("📊 Churn Dashboard")

    # Churn distribution (Pie chart)
    churn_counts = df['Churn'].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'salmon'])
    ax1.set_title("Churn Distribution")
    st.pyplot(fig1)

    # Contract type vs churn (Bar chart)
    fig2, ax2 = plt.subplots()
    sns.countplot(x="Contract", hue="Churn", data=df, palette="Set2", ax=ax2)
    plt.xticks(rotation=30)
    st.pyplot(fig2)

    # MonthlyCharges vs Churn (Boxplot)
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df, palette="Set3", ax=ax3)
    st.pyplot(fig3)

    # =====================
    # Business Insights Section
    # =====================
    st.subheader("📌 Key Business Insights")

    churn_rate = churn_counts["Yes"] / churn_counts.sum() * 100
    st.write(f1️⃣ **Overall churn rate is {churn_rate:.1f}%** of customers.")

    # Contract churn
    contract_churn = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack().fillna(0)
    if "Yes" in contract_churn.columns:
        month2month_churn = contract_churn.loc["Month-to-month", "Yes"] * 100
        st.write(f"2️⃣ **Customers with Month-to-Month contracts have the highest churn (~{month2month_churn:.1f}%)**.")

    # Monthly charges insight
    high_charges_churn = df[df["MonthlyCharges"] > df["MonthlyCharges"].median()]["Churn"].value_counts(normalize=True)
    if "Yes" in high_charges_churn:
        st.write(f"3️⃣ **High-paying customers (above median charges) churn at {high_charges_churn['Yes']*100:.1f}% rate.**")

    # Tenure insight
    avg_tenure_churned = df[df["Churn"]=="Yes"]["tenure"].mean()
    avg_tenure_stayed = df[df["Churn"]=="No"]["tenure"].mean()
    st.write(f"4️⃣ **Average tenure of churned customers is {avg_tenure_churned:.1f} months, while loyal customers stay ~{avg_tenure_stayed:.1f} months.**")

    st.info("💡 Business Takeaway: Focus retention strategies on **month-to-month, high-charge, short-tenure customers**.")

    # =====================
    # Model Training (XGBoost + SMOTE)
    # =====================
    st.subheader("⚙️ Model Training")

    X = df_encoded.drop('Churn_Yes', axis=1)
    y = df_encoded['Churn_Yes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    # Train XGBoost
    model = XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False)
    model.fit(X_train_res, y_train_res)

    # Predict probabilities
    y_probs = model.predict_proba(X_test)[:, 1]

    # Threshold tuning
    threshold = 0.27
    y_pred = (y_probs > threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    st.write(f"✅ Model Accuracy: **{acc:.2f}**")
    st.write(f"✅ F1-score (Churn): **{f1:.2f}**")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig4, ax4 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax4)
    ax4.set_title("Confusion Matrix")
    st.pyplot(fig4)

    # Feature Importance
    st.subheader("🔎 Feature Importance")
    importance = pd.Series(model.feature_importances_, index=X.columns)
    fig5, ax5 = plt.subplots()
    importance.nlargest(10).plot(kind='barh', ax=ax5, color="skyblue")
    ax5.set_title("Top 10 Important Features")
    st.pyplot(fig5)

    # =====================
    # New Prediction
    # =====================
    st.subheader("📌 Predict Churn for a New Customer")
    user_input = {}
    for col in X.columns:
        if df_encoded[col].nunique() <= 2:  # binary features
            user_input[col] = st.selectbox(col, [0, 1])
        else:
            user_input[col] = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))

    user_df = pd.DataFrame([user_input])
    user_probs = model.predict_proba(user_df)[:, 1][0]
    user_pred = 1 if user_probs > threshold else 0
    st.write("### 🔮 Prediction:", "⚠️ Churn" if user_pred == 1 else "✅ Not Churn")
    st.write(f"Churn Probability: {user_probs:.2f}")
