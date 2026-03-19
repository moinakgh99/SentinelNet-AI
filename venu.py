# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import accuracy_score, confusion_matrix
from datetime import datetime
import time
import os

# ---------------------------
# CONFIG / ASSET PATHS
# ---------------------------
# Place these files in the same folder as app.py or set full paths / URLs
BACKGROUND_IMAGE = "bg_img.jpeg"   # starry background (local file or URL)
DASHBOARD_IMAGE = "img1_nids.jpg"  # hero image (local file or URL)
MODEL_PATH = "rf_model_tuned.pkl"  # RandomForest model (joblib). Optional.

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="SentinelNet - AI Powered Network Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# CSS + STAR BACKGROUND + STYLING
# ---------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;700&display=swap');

    html, body, .main {{
        background: radial-gradient(circle at 10% 10%, rgba(8,12,22,0.95), rgba(8,12,22,0.99));
        color: #E0E0E0;
        font-family: 'Rajdhani', sans-serif;
    }}

    /* App background image (starry) */
    .stApp {{
        background-image: url("{BACKGROUND_IMAGE}");
        background-size: cover;
        background-attachment: fixed;
    }}

    
    /* Headings */
    h1,h2,h3 {{ font-family: 'Orbitron', monospace !important; text-transform: uppercase; letter-spacing:1.4px; }}
    h1 {{ color: #00E5FF; text-shadow: 0 0 14px rgba(0,229,255,0.25); font-weight:900; }}
    h2 {{ color: #14FFEC; text-shadow: 0 0 10px rgba(20,255,236,0.12); }}

    /* Metric card */
    .metric-card {{
        background: linear-gradient(135deg, rgba(0,229,255,0.03), rgba(127,90,240,0.03));
        border: 1px solid rgba(20,255,236,0.08);
        padding: 18px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.6);
        color: #e6ffff;
    }}
    .metric-card h1 {{ font-size:1.6rem; color: #14FFEC; margin:6px 0; }}
    .metric-card h3 {{ font-size:0.95rem; margin:0; color:#cfeff3; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(6,8,20,0.95), rgba(3,6,18,0.95));
        border-right: 2px solid rgba(20,255,236,0.06);
        display:flex;
        flex-direction:column;
        align-items:center;
        padding-top:12px;
    }}

    /* make sidebar buttons look colorful and full width */
    [data-testid="stSidebar"] .stButton > button {{
        background: linear-gradient(90deg,#14FFEC,#7F5AF0) !important;
        border: none !important;
        color: #00121a !important;
        font-weight: 800 !important;
        width: 92% !important;
        padding: 10px 12px !important;
        border-radius: 10px !important;
        margin: 6px 0 !important;
        box-shadow: 0 8px 20px rgba(127,90,240,0.12) !important;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 28px rgba(127,90,240,0.22) !important;
        background: linear-gradient(90deg,#7F5AF0,#14FFEC) !important;
    }}

    /* center images globally */
    img {{ display:block; margin-left:auto !important; margin-right:auto !important; }}

    /* Footer */
    .footer-note {{ text-align:center; color:#9ff3f0; padding:18px 0 30px 0; font-family: Orbitron; }}

    /* small responsive tweaks */
    @media (max-width: 700px) {{
        .metric-card h1 {{ font-size:1.2rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# SESSION STATE (persistent)
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "model" not in st.session_state:
    st.session_state.model = None
if "model_features" not in st.session_state:
    st.session_state.model_features = None

# Try loading model if available (non-fatal)
if st.session_state.model is None:
    try:
        if os.path.exists(MODEL_PATH):
            st.session_state.model = joblib.load(MODEL_PATH)
            if hasattr(st.session_state.model, "feature_names_in_"):
                st.session_state.model_features = list(st.session_state.model.feature_names_in_)
            else:
                st.session_state.model_features = None
    except Exception:
        st.session_state.model = None
        st.session_state.model_features = None

import base64
import streamlit as st

# Convert local image → base64
def load_image_base64(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

banner_base64 = load_image_base64("banner_2.png")

with st.sidebar:

    # ---------------------------
    # FULL WIDTH / FULL HEIGHT IMAGE BANNER
    # ---------------------------
    st.markdown(
        f"""
        <div style="width:100%; text-align:center; margin-bottom:12px;">
            <img src="data:image/png;base64,{banner_base64}"
                 style="
                    width:250%;
                    height:150px;
                    object-fit:cover;
                    border-radius:12px;
                 ">
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------
    # APP TITLE
    # ---------------------------


    st.markdown("<hr style='border:1px solid #14FFEC;'>", unsafe_allow_html=True)

    # ---------------------------
    # NAV BUTTONS
    # ---------------------------
    if st.button("🏠 Dashboard", key="btn_dashboard"):
        st.session_state.page = "Dashboard"

    if st.button("🔍 Detection Models", key="btn_detection"):
        st.session_state.page = "Detection"

    if st.button("⚠️ Alerts & Logs", key="btn_alerts"):
        st.session_state.page = "Alerts"

    if st.button("📈 Performance Metrics", key="btn_performance"):
        st.session_state.page = "Performance"

    st.markdown("<hr>", unsafe_allow_html=True)




# ---------------------------
# Helper utilities
# ---------------------------
def safe_read_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None

def align_features_for_model(df_features, model):
    """Return a dataframe aligned to model input. Fill missing features with 0, drop extras."""
    if hasattr(model, "feature_names_in_"):
        model_feats = list(model.feature_names_in_)
    else:
        model_feats = list(df_features.columns)

    aligned = pd.DataFrame(index=df_features.index)
    for f in model_feats:
        if f in df_features.columns:
            aligned[f] = df_features[f]
        else:
            aligned[f] = 0
    return aligned

def map_label_to_name(label):
    """Map numeric/string labels to 'Normal' vs 'Suspicious' naming for pie and colors."""
    try:
        # numpy ints
        if isinstance(label, (np.integer, int)):
            label = int(label)
            return {0: "Normal", 1: "Suspicious", 2: "Mischievous"}.get(label, str(label))
        else:
            s = str(label).strip()
            if s.lower() in ["normal", "benign", "0"]:
                return "Normal"
            if s.lower() in ["attack", "anomaly", "mischievous", "malicious", "suspicious", "1", "2"]:
                return "Suspicious"
            return s
    except Exception:
        return str(label)

LABEL_COLOR_MAP = {
    "Normal": "#4CC9F0",
    "Suspicious": "#FF4D6D",
    "Mischievous": "#B5179E",
}

# ---------------------------
# Set current page variable
# ---------------------------
page = st.session_state.page

# ---------------------------
# DASHBOARD PAGE  (project overview only, no metrics)
# ---------------------------
if page == "Dashboard":

    st.markdown("""
    <style>
    .fade-img {
        animation: fadeIn 1.3s ease-in-out;
    }

    @keyframes fadeIn {
        0%   { opacity: 0; transform: scale(1.02); }
        100% { opacity: 1; transform: scale(1.0); }
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------------------------------------
    # 🔵 CLEAN CYBER GRID BACKGROUND (NO STAR ANIMATION)
    # ---------------------------------------
    st.markdown("""
        <style>

        /* Apply background ONLY to Streamlit main container */
        section.main > div {
            background-size: cover;
            position: relative;
            overflow: hidden;
        }

        /* --- CYBER GRID BACKGROUND --- */
        section.main > div::after {
            content: "";
            position: absolute;
            top: 0; left: 0;
            width: 200%;
            height: 200%;
            background-image:
                linear-gradient(rgba(0,255,255,0.15) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0,255,255,0.15) 1px, transparent 1px);
            background-size: 60px 60px;
            animation: gridMove 20s linear infinite;
            z-index: -1;
            opacity: 0.25;
        }

        @keyframes gridMove {
            0% { transform: translate(-20px, -20px); }
            100% { transform: translate(-200px, -350px); }
        }

        </style>
    """, unsafe_allow_html=True)


    # ---------------------------------------
    # 🔵 TITLE + SUBTITLE
    # ---------------------------------------
    st.markdown(
        """
        <h1 style="
            text-align:center;
            color:#14FFEC;
            font-family:'Orbitron';
            letter-spacing:2px;
            font-size:42px;
            text-shadow: 0 0 12px rgba(20,255,236,0.8);
        ">
            SentinelNet - AI Powered Network Intrusion Detection System
        </h1>

        <h3 style="
            text-align:center;
            color:#E0FFFF;
            margin-top:5px;
            font-family:'Rajdhani';
            font-size:22px;
            text-shadow:0 0 10px rgba(0,255,255,0.5);
        ">
            Real-Time Cybersecurity Monitoring
        </h3>

        <p style="
            text-align:center;
            color:#CFFFFF;
            font-size:17px;
            font-family:'Poppins';
            margin-top:10px;
            line-height:1.5;
            text-shadow:0 0 6px rgba(0,255,255,0.3);
        ">
            SentinelNet is an AI-driven Network Intrusion Detection System designed<br>
            to identify malicious activity, analyze real-time traffic, and enhance cyber defense.<br><br>
        </p>
        """,
        unsafe_allow_html=True
    )

    import streamlit as st
    import time
    import os

    # list your images
    images = ["image(i).png", "image(ii).png", "img1_nids.jpg", "bg_img.jpeg"]

    # width in pixels — change this to make images smaller/larger
    DESIRED_WIDTH = 520   # try 480, 520, 600 etc.

    # init index
    if "img_index" not in st.session_state:
        st.session_state.img_index = 0

    # placeholder (center using columns)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        placeholder = st.empty()

    # show image centered with fixed width
    try:
        # load file path (ensure file exists in same folder as app.py)
        img_path = images[st.session_state.img_index % len(images)]
        if not os.path.exists(img_path):
            placeholder.error(f"Image not found: {img_path}")
        else:
            with c2:
                placeholder.image(img_path, width=DESIRED_WIDTH, use_container_width=False)
    except Exception as e:
        placeholder.error(f"Could not show image: {e}")

    # advance and rerun after delay
    time.sleep(2)
    st.session_state.img_index = (st.session_state.img_index + 1) % len(images)

    # rerun to update UI (use experimental_rerun to be explicit)
    st.rerun()







# ---------------------------
# DETECTION MODELS PAGE
# ---------------------------
if page == "Detection":
    st.title("🔍 Detection Models - SentinelNet AI")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload CSV file for Anomaly Detection (Label column optional). The uploaded dataset will be used across pages.", type=["csv"])
    if uploaded_file is not None:
        df = safe_read_csv(uploaded_file)
        if df is not None:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.csv_data = df.copy()
            st.success("CSV uploaded and stored (available across pages).")

    if st.session_state.csv_data is None:
        st.info("Please upload your CSV here (one time). The file will be retained while the session is active.")
    else:
        data = st.session_state.csv_data.copy()

        # If a model is loaded, try to predict and populate 'Prediction'
        if st.session_state.model is not None:
            try:
                feats = data.drop(columns=["Label", "Prediction"], errors="ignore")
                X = align_features_for_model(feats, st.session_state.model)
                preds = st.session_state.model.predict(X)
                # store predictions as strings to avoid dtype mismatch later
                data["Prediction"] = [str(p) for p in preds]
                st.session_state.csv_data = data
                st.success("Model predictions computed and stored in session (Prediction column).")
            except Exception as e:
                st.warning(f"Model prediction failed: {e} — visuals will use 'Label' column if present.")
        else:
            st.info("RandomForest model not found in app folder. Place 'rf_model_tuned.pkl' to enable model predictions.")

        # PIE CHART (Normal vs Suspicious)
        st.markdown("### 🔹 Activity Distribution (Normal vs Suspicious/Malicious)")
        source_col = "Prediction" if "Prediction" in data.columns else ("Label" if "Label" in data.columns else None)
        if source_col is None:
            st.info("No 'Label' or 'Prediction' column present. Pie chart can't be built until one exists.")
        else:
            mapped = data[source_col].apply(lambda x: map_label_to_name(x))
            normal_count = int((mapped == "Normal").sum())
            suspicious_count = int((mapped != "Normal").sum())
            pie_df = pd.DataFrame({"Type": ["Normal", "Suspicious/Malicious"], "Count": [normal_count, suspicious_count]})

            fig_pie = go.Figure(go.Pie(
                labels=pie_df["Type"],
                values=pie_df["Count"],
                hole=0.42,
                sort=False,
                marker=dict(colors=[LABEL_COLOR_MAP.get("Normal"), LABEL_COLOR_MAP.get("Suspicious")]),
                textinfo="label+percent+value"
            ))
            fig_pie.update_layout(plot_bgcolor='rgba(10,14,26,0.8)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#00ffff'), height=380)
            st.plotly_chart(fig_pie, use_container_width=True)

        # 2D dynamic line chart: animate classification per row (x=row index, y=class mapped)
        st.markdown("### 📈 Live Classification Stream (2D dynamic line chart)")
        st.markdown("This animates across dataset rows — x axis = row index (simulated time), y axis = class name (mapped).")

        if source_col is None:
            st.info("Upload CSV with Label column or run the model to see live classification animation.")
        else:
            seq_labels = data[source_col].astype(str).apply(map_label_to_name).tolist()
            # stable ordering of classes
            classes = list(pd.Series(seq_labels).unique())
            y_map = {name: i * 10 + 50 for i, name in enumerate(classes)}  # numeric positions
            # animate   
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", line=dict(width=2), marker=dict(size=6), name="Classification"))
            fig_line.update_layout(
                plot_bgcolor='rgba(10,14,26,0.8)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#00ffff'),
                height=420,
                xaxis_title="Row index (simulated time)",
                yaxis_title="Class (mapped)",
                yaxis=dict(tickmode="array", tickvals=list(y_map.values()), ticktext=list(y_map.keys()))
            )

            # Single placeholder
            placeholder = st.empty()
            placeholder.plotly_chart(fig_line, use_container_width=True)

            # Prepare data for streaming
            max_window = 80
            x_vals, y_vals, text_vals = [], [], []

            for i, cls in enumerate(seq_labels):
                x_vals.append(i)
                y_vals.append(y_map.get(cls, 50) + np.random.uniform(-0.8, 0.8))
                text_vals.append(cls)

                # Keep only last max_window points
                fig_line.data[0].x = x_vals[-max_window:]
                fig_line.data[0].y = y_vals[-max_window:]
                fig_line.data[0].text = text_vals[-max_window:]
                fig_line.data[0].hovertemplate = "Index: %{x}<br>Class: %{text}<extra></extra>"

                # Update chart dynamically
                placeholder.plotly_chart(fig_line, use_container_width=True)
                time.sleep(0.02)  # small pause for smooth animation




            st.success("Live classification animation completed.")
            st.markdown("### Preview (first 10 rows)")
            st.dataframe(data.head(10), use_container_width=True)

# ---------------------------
# ALERTS & LOGS PAGE
# ---------------------------
if page == "Alerts":
    st.title("⚠️ Alerts & Logs")
    st.markdown("---")

    if st.session_state.csv_data is None:
        st.info("Please upload CSV on Detection Models page to populate Alerts & Logs.")
    else:
        data = st.session_state.csv_data.copy()
        source_col = "Prediction" if "Prediction" in data.columns else ("Label" if "Label" in data.columns else None)

        if source_col is None:
            st.warning("No 'Prediction' or 'Label' column found in dataset. Upload CSV or run model first.")
        else:
            # Map labels
            data["MappedLabel"] = data[source_col].astype(str).apply(map_label_to_name)

            # Create logs table
            n = len(data)
            logs = pd.DataFrame({
                "Time": pd.date_range(end=datetime.now(), periods=n, freq="S"),
                "Source IP": [f"192.168.1.{(i % 254)+1}" for i in range(n)],
                "Destination IP": [f"10.0.0.{(i % 254)+1}" for i in range(n)],
                "Label": data["MappedLabel"].values
            })

            # Reorder columns to put Label at the end
            cols = [c for c in logs.columns if c != "Label"] + ["Label"]
            logs = logs[cols]

            # Color mapping for Label
            def label_color(val):
                color = "#4CAF50" if val == "Normal" else "#FF4D4D"
                return f"color: {color}; font-weight: bold"

            # Apply styling
            styled_logs = logs.sort_values(by="Time", ascending=False).head(200).style.applymap(
                label_color, subset=["Label"]
            )

            st.markdown("### Recent Alerts (latest 200 rows)")
            st.dataframe(styled_logs, use_container_width=True, height=320)

            st.markdown("### 📊 Alert Type Distribution (Enhanced Bar Chart)")

            label_counts = logs["Label"].value_counts()

            # Custom colors
            bar_colors = ["#13f28a", "#ff4d6d"]   # neon green, neon red

            fig_bar = go.Figure()

            fig_bar.add_trace(go.Bar(
                x=label_counts.index,
                y=label_counts.values,
                marker=dict(
                    color=bar_colors,
                    line=dict(color="#00ffff", width=2),        # neon border
                    opacity=0.9
                ),
                hovertemplate="<b>%{x}</b>Count: %{y}<extra></extra>",
            ))

            fig_bar.update_layout(
                height=350,
                width=550,
                plot_bgcolor='rgba(10,14,26,0.9)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=40, r=40, t=40, b=40),

                # smaller labels
                font=dict(color='#00ffff', size=12),

                # remove gridlines for modern look
                xaxis=dict(
                    title="",
                    showgrid=False,
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    title="Count",
                    showgrid=False,
                    tickfont=dict(size=12),
                    titlefont=dict(size=12)
                ),

                # rounded bars (hack)
                bargap=0.45,
            )

            # Center using Streamlit columns
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.plotly_chart(fig_bar, use_container_width=False)


# PERFORMANCE PAGE
# ---------------------------
if page == "Performance":
    st.title("📈 System Performance Metrics")
    st.markdown("---")

    # Add small spacing for clean layout
    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.csv_data is None:
        st.info("Upload dataset in Detection Models page to compute model performance.")
    else:
        data = st.session_state.csv_data.copy()

        # compute accuracy & confusion matrix if possible
        if ("Prediction" in data.columns) and ("Label" in data.columns):
            try:
                y_true = data["Label"].astype(str)
                y_pred = data["Prediction"].astype(str)
                acc_val = accuracy_score(y_true, y_pred)
                labels_for_cm = list(np.unique(np.concatenate([y_true, y_pred])))
                conf = confusion_matrix(y_true, y_pred, labels=labels_for_cm)
            except Exception as e:
                st.error(f"Could not compute metrics: {e}")
                acc_val = None
                conf = None
                labels_for_cm = []
        else:
            acc_val = None
            conf = None
            labels_for_cm = []

        # ---------------------------
        # CENTERED ACCURACY CARD
        # ---------------------------
        if acc_val is not None:
            st.markdown(
                """
                <div style="display:flex; justify-content:center; margin-bottom:20px;">
                    <div style="width:420px;">
                        <div class="metric-card">
                            <h3 style="font-size:18px; text-align:center;">Model Accuracy</h3>
                            <h1 style="font-size:52px; text-align:center;">""" + f"{acc_val*100:.2f}%" + """</h1>
                            <p style="text-align:center;">Computed on uploaded CSV (Label vs Prediction)</p>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # extra spacing before confusion matrix
            st.markdown("<br>", unsafe_allow_html=True)

            # ---------------------------
            # CONFUSION MATRIX
            # ---------------------------
            st.markdown("### Confusion Matrix (compact)")
            if conf is not None:
                cm_df = pd.DataFrame(conf, index=labels_for_cm, columns=labels_for_cm)
                fig_cm = px.imshow(
                    cm_df,
                    text_auto=True,
                    color_continuous_scale=["#071122", "#4895EF", "#7209B7"]
                )
                fig_cm.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#00ffff'),
                    height=280
                )
                st.plotly_chart(fig_cm, use_container_width=True)

        else:
            st.info("No ground truth 'Label' available or no model 'Prediction' to compute accuracy and confusion matrix.")

        # ---------------------------
        # RESOURCE SUMMARY CARDS
        # ---------------------------
        st.markdown("<br><br>", unsafe_allow_html=True)  # Extra space before resource cards
        st.markdown("### Resource Usage (summary cards)")

        c1, c2, c3 = st.columns(3)
        metrics = ["CPU Usage", "Memory Usage", "Network Throughput"]
        for col, name in zip([c1, c2, c3], metrics):
            val = np.random.randint(30, 95)
            col.markdown(
                f"""
                <div class="metric-card">
                    <h3 style="font-size:14px">{name}</h3>
                    <h1 style="font-size:28px">{val}%</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------
# FOOTER
# ---------------------------
st.markdown(
    """
    <div class="footer-note">
        &copy; 2025 SentinelNet | AI-Powered NIDS
    </div>
    """,
    unsafe_allow_html=True,
)
