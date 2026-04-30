import streamlit as st
import pandas as pd
import joblib
import os

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Developer Burnout Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* All text */
    html, body, [class*="css"] { color: #d0daf2; font-family: 'Segoe UI', sans-serif; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(168,85,247,0.15));
        border: 1px solid rgba(139,92,246,0.35);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
    }
    .main-header h1 { color: #f0f4ff; font-size: 26px; font-weight: 700; margin: 0 0 6px; }
    .main-header p  { color: #7890b8; font-size: 14px; margin: 0; }

    /* Section cards */
    .section-card {
        background: linear-gradient(135deg, rgba(22,30,65,0.7), rgba(14,19,50,0.8));
        border: 1px solid rgba(130,145,255,0.18);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }
    .section-title {
        font-size: 13px;
        font-weight: 700;
        color: #a78bfa;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(130,145,255,0.15);
    }

    /* Result boxes */
    .result-low {
        background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(5,150,105,0.12));
        border: 2px solid rgba(52,211,153,0.5);
        border-radius: 16px;
        padding: 28px 36px;
        text-align: center;
    }
    .result-medium {
        background: linear-gradient(135deg, rgba(251,146,60,0.2), rgba(234,88,12,0.12));
        border: 2px solid rgba(251,146,60,0.5);
        border-radius: 16px;
        padding: 28px 36px;
        text-align: center;
    }
    .result-high {
        background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(185,28,28,0.12));
        border: 2px solid rgba(248,113,113,0.5);
        border-radius: 16px;
        padding: 28px 36px;
        text-align: center;
    }
    .result-label { font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; }
    .result-value { font-size: 36px; font-weight: 800; }
    .result-low    .result-label { color: #6ee7b7; }
    .result-low    .result-value { color: #34d399; }
    .result-medium .result-label { color: #fed7aa; }
    .result-medium .result-value { color: #fb923c; }
    .result-high   .result-label { color: #fca5a5; }
    .result-high   .result-value { color: #f87171; }

    /* Info tip box */
    .info-box {
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 12px;
        color: #9090c8;
        margin-bottom: 16px;
    }

    /* Validation warning */
    .warn-box {
        background: rgba(239,68,68,0.1);
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 10px;
        padding: 12px 16px;
        font-size: 12px;
        color: #fca5a5;
        margin-top: 10px;
    }

    /* Streamlit widget overrides */
    .stNumberInput > div > div > input {
        background-color: #1a1f3a !important;
        border: 1px solid rgba(130,145,255,0.25) !important;
        color: #d0daf2 !important;
        border-radius: 8px !important;
    }
    .stSlider > div { color: #d0daf2 !important; }
    div[data-testid="stMetricValue"] { color: #f0f4ff !important; font-size: 28px !important; }
    label[data-testid="stWidgetLabel"] > div > p { color: #b0c0d8 !important; font-size: 13px !important; font-weight: 500 !important; }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        padding: 12px 0 !important;
        width: 100% !important;
        letter-spacing: 0.03em !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 28px rgba(99,102,241,0.6) !important;
        transform: translateY(-1px) !important;
    }
    div[data-testid="column"] { padding: 0 6px !important; }
    .stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── Load models ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all three pkl files once and cache them."""
    errors = []

    for fname in ["burnout_pipeline.pkl", "feature_names.pkl", "label_encoder.pkl"]:
        if not os.path.exists(fname):
            errors.append(fname)

    if errors:
        return None, None, None, errors

    pipeline      = joblib.load("burnout_pipeline.pkl")
    feature_names = joblib.load("feature_names.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return pipeline, feature_names, label_encoder, []


pipeline, feature_names, label_encoder, load_errors = load_models()


# ── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔥 Developer Burnout Predictor</h1>
    <p>Fill in the developer profile below and click <strong>Predict Burnout Level</strong> to get an instant prediction from the trained Logistic Regression model.</p>
</div>
""", unsafe_allow_html=True)

# Show load errors if any pkl files are missing
if load_errors:
    st.error(f"❌ Could not find the following required files in the current directory:\n\n" +
             "\n".join(f"  • `{f}`" for f in load_errors) +
             "\n\nMake sure all three `.pkl` files are in the **same folder** as this script, then run again.")
    st.stop()


# ── Feature config ───────────────────────────────────────────────────────────
# (label, min, max, default, step, help text)
FEATURE_CONFIG = {
    "age": (
        "Age (years)", 18, 65, 30, 1,
        "Developer's age in years. Must be between 18 and 65."
    ),
    "experience_years": (
        "Experience (years)", 0, 40, 5, 1,
        "Years of professional software development experience."
    ),
    "daily_work_hours": (
        "Daily Work Hours", 1.0, 18.0, 8.0, 0.5,
        "Average hours worked per day. Realistic range: 4–14."
    ),
    "sleep_hours": (
        "Sleep Hours / Night", 2.0, 12.0, 7.0, 0.5,
        "Average hours of sleep per night. Healthy range: 6–9."
    ),
    "caffeine_intake": (
        "Caffeine Intake (servings/day)", 0, 15, 3, 1,
        "Number of caffeinated drinks (coffee, energy drinks) per day."
    ),
    "bugs_per_day": (
        "Bugs Encountered / Day", 0, 50, 5, 1,
        "Average number of bugs or issues encountered daily."
    ),
    "commits_per_day": (
        "Commits / Day", 0, 50, 10, 1,
        "Average number of code commits pushed per day."
    ),
    "meetings_per_day": (
        "Meetings / Day", 0, 15, 3, 1,
        "Average number of meetings attended per day."
    ),
    "screen_time": (
        "Screen Time (hours/day)", 1.0, 20.0, 10.0, 0.5,
        "Total daily screen time including work and personal use."
    ),
    "exercise_hours": (
        "Exercise (hours/day)", 0.0, 5.0, 0.5, 0.25,
        "Hours of physical exercise per day. Zero is valid."
    ),
    "stress_level": (
        "Stress Level (0–100)", 0, 100, 50, 1,
        "Self-reported or estimated stress level on a scale of 0 (none) to 100 (extreme)."
    ),
}


# ── Input form ───────────────────────────────────────────────────────────────
st.markdown('<div class="info-box">💡 All fields are required. Values are validated before prediction — no negative numbers allowed where they don\'t make sense.</div>', unsafe_allow_html=True)

inputs = {}
validation_errors = []

# Row 1 — Personal info
st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
for i, key in enumerate(["age", "experience_years"]):
    label, mn, mx, default, step, help_txt = FEATURE_CONFIG[key]
    col = col1 if i == 0 else col2
    with col:
        val = st.number_input(label, min_value=mn, max_value=mx, value=default, step=step, help=help_txt, key=key)
        inputs[key] = val

# Row 2 — Work habits
st.markdown('<div class="section-title" style="margin-top:20px;">💼 Daily Work Habits</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
work_keys = ["daily_work_hours", "meetings_per_day", "screen_time"]
for i, key in enumerate(work_keys):
    label, mn, mx, default, step, help_txt = FEATURE_CONFIG[key]
    col = [col1, col2, col3][i]
    with col:
        val = st.number_input(label, min_value=mn, max_value=mx, value=default, step=step, help=help_txt, key=key)
        inputs[key] = val

# Row 3 — Productivity
st.markdown('<div class="section-title" style="margin-top:20px;">⚙️ Productivity Metrics</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
prod_keys = ["commits_per_day", "bugs_per_day"]
for i, key in enumerate(prod_keys):
    label, mn, mx, default, step, help_txt = FEATURE_CONFIG[key]
    col = col1 if i == 0 else col2
    with col:
        val = st.number_input(label, min_value=mn, max_value=mx, value=default, step=step, help=help_txt, key=key)
        inputs[key] = val

# Row 4 — Wellbeing
st.markdown('<div class="section-title" style="margin-top:20px;">🌙 Health & Wellbeing</div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
well_keys = ["sleep_hours", "exercise_hours", "caffeine_intake", "stress_level"]
for i, key in enumerate(well_keys):
    label, mn, mx, default, step, help_txt = FEATURE_CONFIG[key]
    col = [col1, col2, col3, col4][i]
    with col:
        val = st.number_input(label, min_value=mn, max_value=mx, value=default, step=step, help=help_txt, key=key)
        inputs[key] = val

# ── Validation ───────────────────────────────────────────────────────────────
def validate(inputs):
    errors = []
    if inputs["age"] < 18:
        errors.append("Age must be at least 18.")
    if inputs["experience_years"] > inputs["age"] - 16:
        errors.append("Experience years cannot exceed age minus 16 (earliest realistic start).")
    if inputs["sleep_hours"] + inputs["daily_work_hours"] > 24:
        errors.append("Sleep hours + work hours cannot exceed 24 hours in a day.")
    if inputs["screen_time"] < inputs["daily_work_hours"]:
        errors.append("Screen time should be at least as long as daily work hours.")
    if inputs["stress_level"] < 0 or inputs["stress_level"] > 100:
        errors.append("Stress level must be between 0 and 100.")
    return errors


# ── Predict button ────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
col_btn, col_space = st.columns([1, 2])
with col_btn:
    predict_clicked = st.button("🔮 Predict Burnout Level")

if predict_clicked:
    errors = validate(inputs)

    if errors:
        for e in errors:
            st.markdown(f'<div class="warn-box">⚠️ {e}</div>', unsafe_allow_html=True)
    else:
        # Build DataFrame in exact feature order from feature_names.pkl
        input_df = pd.DataFrame([inputs])[feature_names]

        # Predict
        with st.spinner("Running prediction..."):
            prediction_encoded = pipeline.predict(input_df)[0]
            prediction_label   = label_encoder.inverse_transform([prediction_encoded])[0]
            probabilities      = pipeline.predict_proba(input_df)[0]
            class_labels       = label_encoder.inverse_transform(pipeline.classes_)

        # ── Result display ───────────────────────────────────────────────────
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        st.markdown("---")

        level = prediction_label.strip().lower()
        css_class = {
            "low":    "result-low",
            "medium": "result-medium",
            "high":   "result-high",
        }.get(level, "result-medium")

        emoji = {"low": "🟢", "medium": "🟠", "high": "🔴"}.get(level, "⚪")

        col_result, col_probs = st.columns([1, 1])

        with col_result:
            st.markdown(f"""
            <div class="{css_class}">
                <div class="result-label">Predicted Burnout Level</div>
                <div class="result-value">{emoji} {prediction_label}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_probs:
            st.markdown('<div class="section-title">📊 Prediction Confidence</div>', unsafe_allow_html=True)
            # Sort: Low → Medium → High
            order = ["Low", "Medium", "High"]
            bar_colors = {"Low": "#34d399", "Medium": "#fb923c", "High": "#f87171"}
            sorted_pairs = sorted(
                zip(class_labels, probabilities),
                key=lambda x: order.index(x[0]) if x[0] in order else 99
            )
            for label_name, prob in sorted_pairs:
                pct = round(prob * 100, 1)
                color = bar_colors.get(label_name, "#818cf8")
                st.markdown(f"""
                <div style="margin-bottom:14px">
                    <div style="display:flex;justify-content:space-between;font-size:12px;
                                color:#b0c0d8;font-weight:600;margin-bottom:5px">
                        <span>{label_name}</span><span style="color:{color}">{pct}%</span>
                    </div>
                    <div style="height:10px;border-radius:5px;background:rgba(255,255,255,0.06)">
                        <div style="height:100%;width:{pct}%;border-radius:5px;
                                    background:{color};opacity:0.85;
                                    transition:width .6s ease"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Input summary ────────────────────────────────────────────────────
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        with st.expander("📋 View full input summary used for prediction"):
            summary_df = pd.DataFrame([inputs]).T.reset_index()
            summary_df.columns = ["Feature", "Value"]
            summary_df["Feature"] = summary_df["Feature"].str.replace("_", " ").str.title()
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # ── Interpretation tip ───────────────────────────────────────────────
        tips = {
            "low":    "✅ **Low Burnout** — The developer profile looks healthy. Maintain good sleep, balanced work hours, and regular exercise.",
            "medium": "⚠️ **Medium Burnout** — Some risk factors are present. Consider reducing meetings, improving sleep, and monitoring stress levels closely.",
            "high":   "🚨 **High Burnout** — Significant risk factors detected. Immediate action is recommended: reduce work hours, prioritize sleep, and address stress sources.",
        }
        if level in tips:
            st.info(tips[level])


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:40px; text-align:center; font-size:11px; color:#3d4d66;'>Developer Burnout Analysis · DS & AI Track · Nafath Bootcamp 2026 · Logistic Regression Model</div>", unsafe_allow_html=True)
