import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
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
    # load
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------
    # Data Cleaning & Encoding
    # ---------------------------
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # ensure numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # encoded dataframe used for training
    df_encoded = pd.get_dummies(df, drop_first=True)

    # ---------------------------
    # Dashboard Section
    # ---------------------------
    # ---------------------------
# Dashboard Section
# ---------------------------
if selected == "Dashboard":
    st.subheader("📊 Churn Dashboard")

    churn_counts = df['Churn'].value_counts()
    churn_rate = churn_counts.get("Yes", 0) / churn_counts.sum() * 100
    month2month_churn = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack().fillna(0).get("Yes", {}).get("Month-to-month", 0)*100

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-container'><b>Overall Churn Rate</b><br>{churn_rate:.1f}%</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-container'><b>Month-to-Month Churn</b><br>{month2month_churn:.1f}%</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-container'><b>High-paying Churn Rate</b><br>{(df[df['MonthlyCharges'] > df['MonthlyCharges'].median()]['Churn']=='Yes').mean()*100:.1f}%</div>", unsafe_allow_html=True)

    # Fixed size for all charts (same height/width)
    fig_w, fig_h = (6, 4)

    colA, colB = st.columns(2)
    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h))
        ax1.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%',
                startangle=90, colors=['#2ca02c','#d62728'])
        ax1.set_title("Churn Distribution")
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
        st.markdown("</div>", unsafe_allow_html=True)

    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
        sns.countplot(x="Contract", hue="Churn", data=df,
                      palette=["#2ca02c","#d62728"], ax=ax2)
        ax2.set_title("Contract Type vs Churn")
        plt.xticks(rotation=30)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        st.markdown("</div>", unsafe_allow_html=True)

    colC, colD = st.columns(2)
    with colC:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(fig_w, fig_h))
        sns.boxplot(x="Churn", y="MonthlyCharges", data=df,
                    palette=["#2ca02c","#d62728"], ax=ax3)
        ax3.set_title("Monthly Charges vs Churn")
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
        st.markdown("</div>", unsafe_allow_html=True)

    with colD:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(fig_w, fig_h))
        sns.boxplot(x="Churn", y="tenure", data=df,
                    palette=["#2ca02c","#d62728"], ax=ax4)
        ax4.set_title("Tenure vs Churn")
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)
        st.markdown("</div>", unsafe_allow_html=True)


    # ---------------------------
    # Model Training Section
    # ---------------------------
    elif selected == "Model Training":
        st.subheader("⚙️ Model Training")

        # prepare features & label
        if 'Churn_Yes' not in df_encoded.columns:
            st.error("Encoded target column 'Churn_Yes' not found. Make sure your CSV contains 'Churn' and encoding ran correctly.")
        else:
            X = df_encoded.drop('Churn_Yes', axis=1)
            y = df_encoded['Churn_Yes']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            sm = SMOTE(random_state=42)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

            model = XGBClassifier(random_state=42, eval_metric="logloss")
            model.fit(X_train_res, y_train_res)

            # Save in session for prediction
            st.session_state['model'] = model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['threshold'] = 0.27
            st.session_state['X_columns'] = X.columns  # <-- important

            # Eval
            y_probs = model.predict_proba(X_test)[:, 1]
            y_pred = (y_probs > st.session_state['threshold']).astype(int)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model Accuracy", f"{acc:.2f}")
            with col2:
                st.metric("F1-score (Churn)", f"{f1:.2f}")

            # Confusion Matrix
            with st.expander("🧮 Confusion Matrix"):
                fig_cm, ax_cm = plt.subplots(figsize=(6,4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Not Churn", "Churn"], yticklabels=["Not Churn", "Churn"], ax=ax_cm)
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("Actual")
                plt.tight_layout()
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)

            # Feature Importance
            with st.expander("🔎 Top 10 Feature Importance"):
                importance = pd.Series(model.feature_importances_, index=X.columns)
                fig_imp, ax_imp = plt.subplots(figsize=(6,4))
                importance.nlargest(10).plot(kind='barh', ax=ax_imp, color="#1f77b4")
                ax_imp.set_title("Top 10 Important Features")
                plt.tight_layout()
                st.pyplot(fig_imp, use_container_width=True)
                plt.close(fig_imp)

    # ---------------------------
    # Predict Churn Section
    # ---------------------------
    elif selected == "Predict Churn":
        st.subheader("📌 Predict Churn for a New Customer")

        # Ensure model & columns exist
        if "model" not in st.session_state or "X_columns" not in st.session_state:
            st.warning("⚠️ Please train the model first in 'Model Training' section.")
        else:
            model = st.session_state["model"]
            X_columns = st.session_state["X_columns"]

            # Determine top features to ask user for
            importance = pd.Series(model.feature_importances_, index=X_columns)
            top_features = importance.nlargest(8).index.tolist()

            st.markdown("Fill key customer details (only most important features shown):")

            # Build form in 2 columns
            user_input = {}
            with st.form("predict_form"):
                cols = st.columns(2)
                for i, col in enumerate(top_features):
                    # If encoded binary-like column (has an underscore), show friendly label
                    if "_" in col:
                        base, val = col.split("_", 1)
                        label = f"{base} = {val}"
                        with cols[i % 2]:
                            choice = st.selectbox(label, ["No", "Yes"], key=f"{col}_key")
                            user_input[col] = 1 if choice == "Yes" else 0
                    else:
                        # numeric column
                        # use df_encoded stats to set slider range
                        col_min = float(df_encoded[col].min()) if col in df_encoded.columns else 0.0
                        col_max = float(df_encoded[col].max()) if col in df_encoded.columns else 1.0
                        col_mean = float(df_encoded[col].mean()) if col in df_encoded.columns else (col_min + col_max) / 2.0
                        with cols[i % 2]:
                            user_input[col] = st.slider(col, col_min, col_max, col_mean, key=f"{col}_key")

                submitted = st.form_submit_button("Predict Churn")

            if submitted:
                # Create full feature vector with zeros and then set provided top features
                user_row = pd.DataFrame(columns=X_columns)
                user_row.loc[0] = 0  # default 0

                for c, v in user_input.items():
                    if c in user_row.columns:
                        user_row.at[0, c] = v

                # Ensure column order matches training
                user_row = user_row.reindex(columns=X_columns, fill_value=0)

                # Predict
                prob = model.predict_proba(user_row)[:, 1][0]
                pred = "⚠️ Churn" if prob > st.session_state.get('threshold', 0.5) else "✅ Not Churn"

                st.markdown(f"""
                    <div style='padding:20px; border-radius:16px; background-color:#f0f8ff; text-align:center; box-shadow: 0px 6px 15px rgba(0,0,0,0.1);'>
                        <h3>Prediction: {pred}</h3>
                        <p>Churn Probability: <b>{prob:.2f}</b></p>
                    </div>
                """, unsafe_allow_html=True)

else:
    st.info("Upload a Telco churn CSV to get started.")
