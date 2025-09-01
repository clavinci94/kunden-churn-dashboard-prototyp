# =====================
# Libraries
# =====================
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# =====================
# Modell & Features laden
# =====================
model = joblib.load("models/model.pkl")
model_features = joblib.load("models/model_features.pkl")  # Spalten aus dem Training

# =====================
# Helper: Features vorbereiten
# =====================
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = pd.get_dummies(df, drop_first=True)

    # gleiche Reihenfolge & fehlende Spalten auffüllen
    for col in model_features:
        if col not in df_enc.columns:
            df_enc[col] = 0
    df_enc = df_enc[model_features]

    return df_enc

# =====================
# Streamlit Config
# =====================
st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📊", layout="wide")

# =====================
# Custom Dark Styling
# =====================
st.markdown("""
    <style>
    /* Hintergrund */
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Roboto', sans-serif;
    }
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #FF4B2B, #FF416C);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #FF416C, #FF4B2B);
    }
    /* Tabellen */
    .stDataFrame {
        border: 1px solid #333;
        border-radius: 10px;
        overflow: hidden;
    }
    /* Titel */
    h1, h2, h3 {
        color: #FF416C;
    }
    </style>
""", unsafe_allow_html=True)

# =====================
# Tabs
# =====================
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Einzelkunde",
    "📂 Batch CSV",
    "📊 Model Insights",
    "ℹ️ Infos"
])

# =====================
# TAB 1: Einzelkunde
# =====================
with tab1:
    st.header("👤 Einzelkunde Vorhersage")

    tenure = st.number_input("Vertragsdauer (Monate)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monatliche Kosten (€)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Gesamtkosten (€)", min_value=0.0, value=600.0)
    contract = st.selectbox("Vertragsart", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    payment_method = st.selectbox("Zahlungsmethode", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Contract": [contract],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "TechSupport": [tech_support],
        "PaymentMethod": [payment_method]
    })

    if st.button("🔮 Vorhersage berechnen"):
        input_enc = prepare_features(input_data)
        pred_prob = model.predict_proba(input_enc)[0][1] * 100

        # Ampel-Logik mit fancy Boxen
        # Ampel-Logik mit fancy Boxen
        if pred_prob < 30:
            st.markdown(
                f"<div style='background:#1B4332;padding:15px;border-radius:10px;color:white'>"
                f"✅ Geringes Kündigungsrisiko: <b>{pred_prob:.2f}%</b>"
                f"</div>",
                unsafe_allow_html=True
            )
        elif pred_prob < 70:
            st.markdown(
                f"<div style='background:#FFB703;padding:15px;border-radius:10px;color:black'>"
                f"⚠️ Mittleres Kündigungsrisiko: <b>{pred_prob:.2f}%</b>"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='background:#D00000;padding:15px;border-radius:10px;color:white'>"
                f"🚨 Hohes Kündigungsrisiko: <b>{pred_prob:.2f}%</b>"
                f"</div>",
                unsafe_allow_html=True
            )

# =====================
# TAB 2: Batch CSV hochladen
# =====================
with tab2:
    st.header("📂 Batch-Vorhersagen aus CSV")
    uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔎 Vorschau der hochgeladenen Daten:")
        st.dataframe(df.head())

        df_enc = prepare_features(df)

        df["Churn_Probability (%)"] = model.predict_proba(df_enc)[:, 1] * 100
        df["Churn_Probability (%)"] = df["Churn_Probability (%)"].round(2)

        st.success("✅ Vorhersagen wurden berechnet.")
        st.dataframe(df.head(20))

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Ergebnisse herunterladen",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

# =====================
# TAB 3: Model Insights
# =====================
with tab3:
    st.header("📊 Model Insights")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        features = model_features

        importance_df = pd.DataFrame({
            "Feature": features,
            "Importance": importances
        }).sort_values(by="Importance", ascending=True)

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="RdBu",
            title="🔎 Wichtigste Einflussfaktoren"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Dieses Modell unterstützt keine Feature Importance.")

# =====================
# TAB 4: Infos
# =====================
with tab4:
    st.header("ℹ️ Infos zum Dashboard")
    st.markdown("""
    Dieses Dashboard ist ein Prototyp für **Customer Churn Prediction**.  

    ### Features:
    - 👤 Einzelkunde Vorhersage  
    - 📂 Batch-Upload für mehrere Kunden  
    - 📊 Visualisierung der wichtigsten Einflussfaktoren  
    - ⚡ Deployment-fähig auf Streamlit Cloud  

    ### Technologien:
    - Python (pandas, scikit-learn, joblib)
    - Streamlit (UI)
    - Plotly (Visualisierung)
    """)
