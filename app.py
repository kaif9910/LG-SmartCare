import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="LG SmartCare — Predictive Maintenance",
    page_icon="⚙️",
    layout="centered"
)

# CSS for styling
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

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load preprocessor and XGBoost model from disk. Returns (None, None) on failure."""
    try:
        preprocessor = joblib.load("lg_preprocessor.pkl")
        model        = joblib.load("lg_predictive_maintenance_xgb.pkl")
        return preprocessor, model
    except FileNotFoundError as e:
        st.error(
            f"Model file not found: {e}\n\n"
            "Please run **Cell 13** in the Jupyter notebook first to export "
            "`lg_preprocessor.pkl` and `lg_predictive_maintenance_xgb.pkl`."
        )
        return None, None

preprocessor, model = load_models()

# ── Only show the form when models loaded successfully ─────────────────────────
if preprocessor is not None and model is not None:

    st.header("Input Machine Parameters")

    # Feature explanations (collapsible)
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
        appliance_type = st.selectbox(
            "Appliance Type (Quality Grade)",
            options=["L", "M", "H"],
            help="L = Low quality, M = Medium, H = High"
        )
        ambient_temperature = st.number_input(
            "Ambient Temperature [K]",
            min_value=250.0, max_value=350.0,
            value=298.0, step=0.1,
            help="Typical range: 295–305 K"
        )
        process_temperature = st.number_input(
            "Process Temperature [K]",
            min_value=250.0, max_value=350.0,
            value=308.0, step=0.1,
            help="Usually 5–15 K higher than ambient"
        )

    with col2:
        rotational_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=1000, max_value=3000,
            value=1500, step=10,
            help="Typical range: 1200–2800 rpm"
        )
        torque = st.number_input(
            "Torque [Nm]",
            min_value=1.0, max_value=100.0,
            value=40.0, step=0.5,
            help="Typical range: 20–70 Nm"
        )
        device_age_proxy = st.number_input(
            "Tool Wear / Device Age [minutes]",
            min_value=0, max_value=300,
            value=100, step=1,
            help="0 = brand new, 300 = heavily used"
        )

    # ── Derived features (auto-calculated) ────────────────────────────────────
    temp_diff         = process_temperature - ambient_temperature
    device_age_months = device_age_proxy / 30.0

    st.markdown(
        f'<div class="info-box">'
        f'Auto-calculated — <b>Temperature Difference:</b> {temp_diff:.1f} K &nbsp;|&nbsp; '
        f'<b>Device Age (months):</b> {device_age_months:.2f}'
        f'</div>',
        unsafe_allow_html=True
    )

    # Input sanity warning
    if process_temperature <= ambient_temperature:
        st.warning(
            "⚠️ Process Temperature should be higher than Ambient Temperature. "
            "Please check your inputs."
        )

    st.divider()

    # ── Predict ───────────────────────────────────────────────────────────────
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

            # Probability bar
            st.write(f"**Failure Probability:** {risk_prob * 100:.1f}%")
            st.progress(float(risk_prob))

            # Risk category
            if risk_prob > 0.60:
                st.markdown(
                    '<p class="high-risk">🔴 HIGH RISK — Immediate service required!</p>',
                    unsafe_allow_html=True
                )
                st.error(
                    "This appliance has a high probability of failure. "
                    "Schedule a technician visit immediately to avoid costly breakdowns."
                )
            elif risk_prob > 0.35:
                st.markdown(
                    '<p class="medium-risk">🟠 MEDIUM RISK — Schedule maintenance soon.</p>',
                    unsafe_allow_html=True
                )
                st.warning(
                    "Some parameters are outside optimal range. "
                    "Plan a preventive maintenance visit within the next 2–4 weeks."
                )
            else:
                st.markdown(
                    '<p class="low-risk">🟢 LOW RISK — Machine is running optimally.</p>',
                    unsafe_allow_html=True
                )
                st.success(
                    "All parameters look healthy. Continue regular monitoring "
                    "and log the next scheduled inspection."
                )

            # Summary table
            with st.expander("📋 View input summary"):
                summary = input_data.copy()
                summary["temp_diff"]         = temp_diff
                summary["device_age_months"] = round(device_age_months, 2)
                st.dataframe(summary, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info(
                "Make sure the model files were exported from the **same** version "
                "of the notebook that created this app."
            )

else:
    st.info(
        "Models could not be loaded. Please check the error above and ensure "
        "`lg_preprocessor.pkl` and `lg_predictive_maintenance_xgb.pkl` are in "
        "the same folder as this script."
    )
