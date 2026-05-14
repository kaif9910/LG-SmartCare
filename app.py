import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

st.set_page_config(
    page_title="LG SmartCare — Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)

st.markdown("""
    <style>
    .high-risk   { color: #cc0000; font-weight: bold; font-size: 22px; }
    .medium-risk { color: #cc7700; font-weight: bold; font-size: 22px; }
    .low-risk    { color: #1a7a1a; font-weight: bold; font-size: 22px; }
    .info-box {
        background-color: #f0f4ff;
        border-left: 4px solid #3a5fd9;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        font-size: 13px;
        color: #333;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚙️ LG SmartCare — Machine Failure Risk Analyzer")
st.write("Enter appliance parameters below to predict failure risk using our trained XGBoost model.")

@st.cache_resource
def train_model():
    try:
        df = pd.read_csv('ai4i.csv')
    except FileNotFoundError:
        return None, None, "ai4i.csv not found in repo!"

    df = df.rename(columns={
        'Type':                    'appliance_type',
        'Air temperature [K]':     'ambient_temperature',
        'Process temperature [K]': 'process_temperature',
        'Rotational speed [rpm]':  'rotational_speed',
        'Torque [Nm]':             'torque',
        'Tool wear [min]':         'device_age_proxy',
        'Machine failure':         'service_required',
    })

    df = df.drop(columns=['Product ID'], errors='ignore')
    df['temp_diff']         = df['process_temperature'] - df['ambient_temperature']
    df['device_age_months'] = df['device_age_proxy'] / 30
    df = df.drop(columns=['device_age_proxy'])

    cols_to_drop = ['service_required', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'date']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.dropna(subset=['service_required'])

    X = df.drop(columns=cols_to_drop)
    y = df['service_required']

    numeric_features     = ['ambient_temperature', 'process_temperature',
                             'rotational_speed', 'torque',
                             'temp_diff', 'device_age_months']
    categorical_features = ['appliance_type']

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline(steps=[('scaler', StandardScaler())]),     numeric_features),
        ('cat', Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features),
    ])

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_processed = preprocessor.fit_transform(X_train)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_processed, y_train)

    model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                          random_state=42, eval_metric='logloss')
    model.fit(X_resampled, y_resampled)

    return preprocessor, model, None

with st.spinner("🔄 Model training... please wait (only happens once)"):
    preprocessor, model, error = train_model()

if error:
    st.error(f"Error: {error}")
elif preprocessor is not None and model is not None:

    st.success("✅ Model ready!")
    st.header("Input Machine Parameters")

    with st.expander("ℹ️ What do these parameters mean?"):
        st.markdown("""
        | Parameter | Description |
        |---|---|
        | **Appliance Type** | Quality grade: L = Low, M = Medium, H = High |
        | **Ambient Temperature** | Room/environment temperature in Kelvin |
        | **Process Temperature** | Operating temperature inside the machine in Kelvin |
        | **Rotational Speed** | Motor speed in revolutions per minute (rpm) |
        | **Torque** | Rotational force applied by the motor (Newton-metres) |
        | **Tool Wear** | Cumulative usage time (proxy for device age) in minutes |
        """)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        appliance_type = st.selectbox("Appliance Type (Quality Grade)", options=["L", "M", "H"])
        ambient_temperature = st.number_input("Ambient Temperature [K]", min_value=250.0, max_value=350.0, value=298.0, step=0.1)
        process_temperature = st.number_input("Process Temperature [K]", min_value=250.0, max_value=350.0, value=308.0, step=0.1)

    with col2:
        rotational_speed = st.number_input("Rotational Speed [rpm]", min_value=1000, max_value=3000, value=1500, step=10)
        torque = st.number_input("Torque [Nm]", min_value=1.0, max_value=100.0, value=40.0, step=0.5)
        device_age_proxy = st.number_input("Tool Wear / Device Age [minutes]", min_value=0, max_value=300, value=100, step=1)

    temp_diff         = process_temperature - ambient_temperature
    device_age_months = device_age_proxy / 30.0

    st.markdown(
        f'<div class="info-box">Auto-calculated — <b>Temperature Difference:</b> {temp_diff:.1f} K &nbsp;|&nbsp; '
        f'<b>Device Age (months):</b> {device_age_months:.2f}</div>',
        unsafe_allow_html=True
    )

    if process_temperature <= ambient_temperature:
        st.warning("⚠️ Process Temperature should be higher than Ambient Temperature.")

    st.divider()

    if st.button("🔍 Predict Failure Risk", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            "appliance_type":      [appliance_type],
            "ambient_temperature": [ambient_temperature],
            "process_temperature": [process_temperature],
            "rotational_speed":    [rotational_speed],
            "torque":              [torque],
            "temp_diff":           [temp_diff],
            "device_age_months":   [device_age_months],
        })

        try:
            processed_input = preprocessor.transform(input_data)
            risk_prob       = model.predict_proba(processed_input)[0][1]

            st.subheader("Prediction Results")
            st.write(f"**Failure Probability:** {risk_prob * 100:.1f}%")
            st.progress(float(risk_prob))

            if risk_prob > 0.60:
                st.markdown('<p class="high-risk">🔴 HIGH RISK — Immediate service required!</p>', unsafe_allow_html=True)
                st.error("High probability of failure. Schedule a technician visit immediately.")
            elif risk_prob > 0.35:
                st.markdown('<p class="medium-risk">🟠 MEDIUM RISK — Schedule maintenance soon.</p>', unsafe_allow_html=True)
                st.warning("Plan a preventive maintenance visit within 2–4 weeks.")
            else:
                st.markdown('<p class="low-risk">🟢 LOW RISK — Machine is running optimally.</p>', unsafe_allow_html=True)
                st.success("All parameters look healthy. Continue regular monitoring.")

            with st.expander("📋 View input summary"):
                st.dataframe(input_data, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
