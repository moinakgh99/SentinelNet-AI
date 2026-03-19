# app.py
# SentinelNet merged app — hologram slideshow option B (auto-rotate every 2s)
# Run: streamlit run app.py

import base64
def img_to_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

import io
import json
import os
import re
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(
    page_title="SentinelNet NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# CSS / Styling (single copy) including hologram frame + slideshow animation
# ---------------------------
st.markdown(
    """
    <style>
    
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;700&display=swap');

    * { font-family: 'Rajdhani', sans-serif; }
    body, .main { background: linear-gradient(180deg, #0A0F1F 0%, #1A1037 100%); color: #E0E0E0; }

    h1,h2,h3 { font-family: 'Orbitron', monospace !important; text-transform: uppercase; letter-spacing: 2px; }
    h1 { color: #00E5FF; text-shadow: 0 0 12px rgba(0,229,255,0.45); font-weight:900; animation: fadeIn 0.6s ease-out; }
    @keyframes fadeIn { from { opacity:0; transform: translateY(-10px);} to {opacity:1; transform:translateY(0);} }

    .metric-card {
        background: linear-gradient(135deg, rgba(0,229,255,0.08), rgba(127,90,240,0.08));
        border: 2px solid #00E5FF;
        padding: 20px;
        border-radius: 12px;
        text-align:center;
        transition: all 0.35s ease;
        box-shadow: 0 0 18px rgba(0,229,255,0.18);
    }
    .metric-card:hover { transform: translateY(-6px) scale(1.02); border-color:#7F5AF0; box-shadow: 0 0 28px rgba(127,90,240,0.18); }
    .metric-card h1 { color:#14FFEC; font-size:1.8rem; margin:6px 0; }

    .stButton > button {
        background: linear-gradient(135deg, #14FFEC 0%, #7F5AF0 100%);
        border: none;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight:700;
        letter-spacing:1px;
        transition: transform .2s;
    }
    .stButton > button:hover { transform: translateY(-3px); }

    [data-testid="stSidebar"] {
        background: rgba(10,15,31,0.95);
        border-right: 2px solid rgba(20,255,236,0.06);
        padding-top:8px;
    }
    [data-testid="stSidebar"] .stRadio > label {
        background: rgba(255,255,255,0.02);
        padding:8px 12px;
        border-radius:6px;
        color:#bff7f6;
    }
    .stDataFrame { background: rgba(15,20,40,0.65); border-radius:6px; }

    /* hologram frame + slideshow */
    .hologram-wrap {
        width: 100%;
        max-width: 720px;
        margin: 0 auto;
        position: relative;
        padding: 28px;
        border-radius: 16px;
        background: linear-gradient(180deg, rgba(20,255,236,0.03), rgba(127,90,240,0.02));
        border: 1px solid rgba(20,255,236,0.06);
        box-shadow: 0 20px 60px rgba(0,0,0,0.6);
        display:flex;
        justify-content:center;
        align-items:center;
    }
    .holo-frame {
        width: 100%;
        border-radius: 12px;
        overflow: hidden;
        position: relative;
        padding: 20px;
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        box-shadow: inset 0 0 60px rgba(20,255,236,0.02);
    }

    .holo-stack {
        position: relative;
        width: 100%;
        height: 360px;
        display: block;
        overflow: hidden;
        border-radius: 10px;
        perspective: 1200px;
        -webkit-perspective: 1200px;
    }

    .holo-stack img {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        max-width: 100%;
        max-height: 100%;
        width: auto;
        height: auto;
        opacity: 0;
        object-fit: cover;
        transition: transform 0.9s cubic-bezier(.2,.9,.3,1), opacity 0.9s ease;
        filter: drop-shadow(0 18px 40px rgba(0,0,0,0.6)) saturate(1.05);
    }

    /* create a smooth crossfade loop using keyframes */
    @keyframes holoFade {
        0%   { opacity: 0; transform: translate(-50%, -50%) scale(1.03) translateZ(-20px); }
        8%   { opacity: 1; transform: translate(-50%, -50%) scale(1.00) translateZ(0); }
        42%  { opacity: 1; transform: translate(-50%, -50%) scale(1.00) translateZ(0); }
        50%  { opacity: 0; transform: translate(-50%, -50%) scale(0.98) translateZ(-20px); }
        100% { opacity: 0; transform: translate(-50%, -50%) scale(0.98) translateZ(-20px); }
    }

    /* each image gets the same animation but staggered via animation-delay inlined on the element */
    .holo-stack img {
        animation-name: holoFade;
        animation-duration: 6s; /* full cycle for N images; we'll stagger by delay */
        animation-iteration-count: infinite;
        animation-timing-function: ease-in-out;
    }

    /* holographic overlay lines */
    .holo-overlay {
        position:absolute;
        inset:0;
        pointer-events:none;
        background-image:
           linear-gradient(180deg, rgba(20,255,236,0.03) 1px, transparent 1px),
           linear-gradient(90deg, rgba(127,90,240,0.02) 1px, transparent 1px);
        background-size: 40px 40px;
        mix-blend-mode: screen;
        opacity:0.08;
    }

    /* small responsive tweaks */
    @media (max-width:900px) {
        .holo-stack { height: 260px; }
    }
    @media (max-width:600px) {
        .holo-stack { height: 200px; }
        .hologram-wrap { padding: 12px; }
    }
    /* ========================================
   FINAL WORKING OUTER GLOW FOR PLOTLY CHARTS
   Matches your DOM:  <div class="plot-container plotly">
   ======================================== */

/* ========================================
   MEDIUM–SOFT OUTER GLOW FOR PLOTLY CHARTS
   ======================================== */

.plot-container.plotly {
    position: relative !important;
    overflow: visible !important;
}

.plot-container.plotly::after {
    content: "";
    position: absolute;

    /* balanced spacing */
    inset: -10px;

    border-radius: 14px;

    /* medium–soft glow */
    box-shadow:
        0 0 8px rgba(0,255,255,0.30),
        0 0 15px rgba(0,255,255,0.18),
        0 0 22px rgba(127,90,240,0.22);

    animation: mediumGlow 3s infinite alternate ease-in-out;
    pointer-events: none;
    z-index: 6;
}

@keyframes mediumGlow {
    0% {
        box-shadow:
            0 0 5px rgba(0,255,255,0.20),
            0 0 10px rgba(0,255,255,0.15),
            0 0 18px rgba(127,90,240,0.18);
    }
    100% {
        box-shadow:
            0 0 8px rgba(0,255,255,0.30),
            0 0 15px rgba(0,255,255,0.18),
            0 0 22px rgba(127,90,240,0.22);
    }
}

    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helper functions (kept from original code)
# ---------------------------
def safe_read_csv(uploaded_file):
    try:
        if hasattr(uploaded_file, "read"):
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_csv(str(uploaded_file))
    except Exception:
        try:
            if hasattr(uploaded_file, "read"):
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, engine="python", low_memory=False)
            else:
                return pd.read_csv(str(uploaded_file), engine="python", low_memory=False)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            return None


def align_features_for_model(df_features, model):
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
    try:
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


def df_info_str(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def find_label_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.strip().lower() == "label":
            return c
    return None


# ---------------------------
# Session-state defaults and monitoring logic
# ---------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# monitoring_status: "Active" | "Running" | "Monitoring Done" | "Stopped"
if "monitoring_status" not in st.session_state:
    st.session_state.monitoring_status = "Active"

if "csv_data" not in st.session_state:
    st.session_state.csv_data = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "model" not in st.session_state:
    st.session_state.model = None
if "model_features" not in st.session_state:
    st.session_state.model_features = None
if "img_index" not in st.session_state:
    st.session_state.img_index = 0

# Try load the RandomForest model if present (Venu logic)
MODEL_PATH = "rf_model_tuned.pkl"
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

# ---------------------------
# Sidebar: Navigation (3 main items). When Overview chosen, show 4 sub-pages.
# Remove Total Packets/Active Threats/Uptime metrics from sidebar.
# ---------------------------
with st.sidebar:
    st.image("Gemini_Generated_Image_odeq68odeq68odeq.png", width=190)
    st.markdown("""
    <h2 style='color:#14FFEC;
            margin-top:10px;
            font-family: Orbitron;
            letter-spacing: 2px;
            text-shadow: 0 0 12px rgba(20,255,236,0.45);'>
        SentinelNet-NIDS
    </h2>
    """, unsafe_allow_html=True)

    st.markdown("---")

    nav_choice = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Detection Models", "📘 Overview"],
        index=0,
        key="nav_main"
    )

    # Map the nav to session_state.page (Overview handled below)
    if nav_choice == "🏠 Dashboard":
        st.session_state.page = "Dashboard"
    elif nav_choice == "🔍 Detection Models":
        st.session_state.page = "Detection"
    else:
        # Show sub-navigation for Overview pages
        sub = st.radio(
            "Overview pages",
            ["Overview Home", "🧩 EDA & Cleaning", "🤖 Supervised Models", "🔍 Anomaly & Evaluation", "🚨 Alerts"],
            index=0,
            key="overview_sub",
        )
        if sub == "Overview Home":
            st.session_state.page = "Overview_Home"
        elif sub == "🧩 EDA & Cleaning":
            st.session_state.page = "Overview_EDA"
        elif sub == "🤖 Supervised Models":
            st.session_state.page = "Overview_Supervised"
        elif sub == "🔍 Anomaly & Evaluation":
            st.session_state.page = "Overview_Anomaly"
        else:
            st.session_state.page = "Overview_Alerts"

    st.markdown("---")
    st.markdown("### System Status")

    ms = st.session_state.monitoring_status
    if ms == "Active" or ms == "Running":
        if st.button("🔴 Stop Monitoring"):
            st.session_state.monitoring_status = "Stopped"
    elif ms == "Stopped":
        if st.button("🟢 Start Monitoring"):
            st.session_state.monitoring_status = "Running"
    elif ms == "Monitoring Done":
        if st.button("🔁 Reset Monitoring"):
            st.session_state.monitoring_status = "Active"

    if st.session_state.monitoring_status == "Active":
        st.success("System Active")
    elif st.session_state.monitoring_status == "Running":
        st.info("Monitoring Running")
    elif st.session_state.monitoring_status == "Monitoring Done":
        st.success("Monitoring Done")
    else:
        st.warning("Monitoring Stopped")

    st.markdown("---")
    st.caption("Navigation: Dashboard, Detection Models, Overview (four pages)")

# ---------------------------
# Dashboard page (Venu heading + Fazil overview cards + hologram slideshow)
# ---------------------------
if st.session_state.page == "Dashboard":
    st.markdown(
        """
        <h1 style="text-align:center; color:#14FFEC; font-family:'Orbitron'; letter-spacing:2px; font-size:40px;">
            SentinelNet - AI Powered Network Intrusion Detection System
        </h1>
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
        <h3 style="text-align:center; color:#E0FFFF; margin-top:6px; font-family:'Rajdhani'; font-size:20px;">
            Real-Time Cybersecurity Monitoring
        </h3>

        """,
        unsafe_allow_html=True,
    )

    # Fazil Overview cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><h3>Dataset</h3><h1>CICIDS2017</h1><div class="small">Flow CSVs</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h3>Algorithms</h3><h1>RF / SVM / LR</h1><div class="small">with PCA</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h3>Best Model</h3><h1>Random Forest</h1><div class="small">~96% acc</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><h3>Alerts</h3><h1>Real-Time</h1><div class="small">CSV logging</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------------- 3D HOLOGRAM ROTATING SLIDESHOW ----------------------

    img1 = img_to_base64("image(i).png")
    img2 = img_to_base64("HD-wallpaper-3d-neon-blue-lock-computer-security-digital-technology-blue-digital-background-blue-security-background-lock-concepts-thumbnail.webp")
    img3 = img_to_base64("image(ii).png")

    st.markdown(f"""
    <div class="holo3d-wrap">
        <div class="holo3d-frame">
            <div class="holo3d-slider">
                <img src="data:image/webp;base64,{img1}" class="holo-img" />
                <img src="data:image/webp;base64,{img2}" class="holo-img" />
                <img src="data:image/webp;base64,{img3}" class="holo-img" />
            </div>
        </div>
    </div>

    <style>
    .holo3d-wrap {{
        width: 100%;
        max-width: 900px;
        margin: 0 auto 35px auto;
        padding: 20px;
        border-radius: 18px;
        background: rgba(20,255,236,0.03);
        box-shadow: 0 0 25px rgba(20,255,236,0.25),
                    0 0 50px rgba(127,90,240,0.25);
        perspective: 1200px;
    }}
    .holo3d-frame {{
        width: 100%;
        height: 370px;
        border-radius: 14px;
        overflow: hidden;
        position: relative;
        transform-style: preserve-3d;
    }}
    .holo3d-slider {{
        width: 100%;
        height: 100%;
        position: absolute;
        display: flex;
        animation: rotate3D 18s infinite linear;
        transform-style: preserve-3d;
    }}
    .holo-img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        margin-right: -100%;
        opacity: 0.85;
        border-radius: 14px;
        box-shadow:
            0 0 20px rgba(0,255,255,0.25),
            0 0 35px rgba(127,90,240,0.25);
    }}
    .holo-img:nth-child(1) {{ transform: rotateY(0deg) translateZ(250px); }}
    .holo-img:nth-child(2) {{ transform: rotateY(120deg) translateZ(250px); }}
    .holo-img:nth-child(3) {{ transform: rotateY(240deg) translateZ(250px); }}

    @keyframes rotate3D {{
        0%   {{ transform: rotateY(0deg); }}
        100% {{ transform: rotateY(-360deg); }}
    }}
    </style>
    """, unsafe_allow_html=True)

# ---------------------- END 3D HOLOGRAM SLIDESHOW ----------------------

    st.markdown("---")
    st.markdown("### Dashboard Snapshot")
    row1, row2 = st.columns([2, 3])
    with row1:
        st.markdown('<div class="metric-card"><h3>Current Status</h3><h1>{}</h1></div>'.format(st.session_state.monitoring_status), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="metric-card"><h3>Model Loaded</h3><h1>{}</h1></div>'.format("Yes" if st.session_state.model is not None else "No"), unsafe_allow_html=True)
    with row2:
        st.markdown("#### Recent Alerts (sample)")
        demo_alerts = pd.DataFrame({
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * 4,
            "Source IP": ["192.168.1.10", "192.168.1.15", "10.0.0.5", "172.16.0.2"],
            "Alert": ["Normal", "Suspicious", "Normal", "Suspicious"]
        })
        st.dataframe(demo_alerts, use_container_width=True, height=220)

# ---------------------------
# Detection Models page
# - Venu detection logic (upload, model predict, plots)
# - Alerts & Logs + Performance metrics integrated
# ---------------------------
elif st.session_state.page == "Detection":
    st.title("🔍 Detection Models - SentinelNet AI")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload CSV file for Anomaly Detection (Label column optional). The uploaded dataset will be used across pages.",
        type=["csv"],
        key="detection_uploader",
    )

    if uploaded_file is not None:
        df = safe_read_csv(uploaded_file)
        if df is not None:
            st.session_state.uploaded_file = uploaded_file
            st.session_state.csv_data = df.copy()
            st.session_state.monitoring_status = "Running"
            st.success("CSV uploaded and stored (available across pages). Monitoring started...")

    if st.session_state.csv_data is None:
        st.info("Please upload your CSV here (one time). The file will be retained while the session is active.")
    else:
        data = st.session_state.csv_data.copy()

        if st.session_state.model is not None:
            try:
                feats = data.drop(columns=["Label", "Prediction"], errors="ignore")
                X = align_features_for_model(feats, st.session_state.model)
                preds = st.session_state.model.predict(X)
                data["Prediction"] = [str(p) for p in preds]
                st.session_state.csv_data = data
                st.success("Model predictions computed and stored in session (Prediction column).")
            except Exception as e:
                st.warning(f"Model prediction failed: {e} — visuals will use 'Label' column if present.")
        else:
            st.info("RandomForest model not found in app folder. Place 'rf_model_tuned.pkl' to enable model predictions.")

        # PIE CHART
        st.markdown("### 🔹 Activity Distribution (Normal vs Suspicious/Malicious)")
        source_col = "Prediction" if "Prediction" in data.columns else ("Label" if "Label" in data.columns else None)
        if source_col is None:
            st.info("No 'Label' or 'Prediction' column present. Pie chart can't be built until one exists.")
        else:
            mapped = data[source_col].apply(lambda x: map_label_to_name(x))
            normal_count = int((mapped == "Normal").sum())
            suspicious_count = int((mapped != "Normal").sum())
            pie_df = pd.DataFrame({"Type": ["Normal", "Suspicious/Malicious"], "Count": [normal_count, suspicious_count]})

            fig_pie = go.Figure(
                go.Pie(
                    labels=pie_df["Type"],
                    values=pie_df["Count"],
                    hole=0.42,
                    sort=False,
                    marker=dict(colors=[LABEL_COLOR_MAP.get("Normal"), LABEL_COLOR_MAP.get("Suspicious")]),
                    textinfo="label+percent+value",
                )
            )
            fig_pie.update_layout(
                plot_bgcolor="rgba(10,14,26,0.8)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#00ffff"),
                height=380,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # Live classification animation (capped)
        st.markdown("### 📈 Live Classification Stream (2D dynamic line chart)")
        st.markdown("This animates across dataset rows — x axis = row index (simulated time), y axis = class name (mapped).")

        if source_col is None:
            st.info("Upload CSV with Label column or run the model to see live classification animation.")
        else:
            seq_labels = data[source_col].astype(str).apply(map_label_to_name).tolist()
            classes = list(pd.Series(seq_labels).unique())
            y_map = {name: i * 10 + 50 for i, name in enumerate(classes)}
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", line=dict(width=2), marker=dict(size=6), name="Classification"))
            fig_line.update_layout(
                plot_bgcolor="rgba(10,14,26,0.8)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#00ffff"),
                height=420,
                xaxis_title="Row index (simulated time)",
                yaxis_title="Class (mapped)",
                yaxis=dict(tickmode="array", tickvals=list(y_map.values()), ticktext=list(y_map.keys())),
            )

            placeholder = st.empty()
            placeholder.plotly_chart(fig_line, use_container_width=True)

            max_display = min(len(seq_labels), 500)
            max_window = 80
            x_vals, y_vals, text_vals = [], [], []

            for i, cls in enumerate(seq_labels[:max_display]):
                x_vals.append(i)
                y_vals.append(y_map.get(cls, 50) + np.random.uniform(-0.8, 0.8))
                text_vals.append(cls)

                fig_line.data[0].x = x_vals[-max_window:]
                fig_line.data[0].y = y_vals[-max_window:]
                fig_line.data[0].text = text_vals[-max_window:]
                fig_line.data[0].hovertemplate = "Index: %{x}<br>Class: %{text}<extra></extra>"

                placeholder.plotly_chart(fig_line, use_container_width=True)
                time.sleep(0.01)

            st.success("Live classification animation completed.")
            st.markdown("### Preview (first 10 rows)")
            st.dataframe(data.head(10), use_container_width=True)

        # ---- Alerts & Logs (Venu integrated) ----
        st.markdown("---")
        st.markdown("## ⚠️ Alerts & Logs (Live)")
        if source_col is not None:
            mapped_col = data[source_col].astype(str).apply(map_label_to_name)
            n = len(mapped_col)
            logs = pd.DataFrame({
                "Time": pd.date_range(end=datetime.now(), periods=n, freq="S"),
                "Source IP": [f"192.168.1.{(i % 254)+1}" for i in range(n)],
                "Destination IP": [f"10.0.0.{(i % 254)+1}" for i in range(n)],
                "Label": mapped_col.values
            })
            st.markdown("### Recent Alerts (latest 50 rows)")
            st.dataframe(logs.sort_values(by="Time", ascending=False).head(200), use_container_width=True, height=280)

            label_counts = logs["Label"].value_counts()
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=label_counts.index,
                y=label_counts.values,
                marker=dict(line=dict(color="#00ffff", width=1)),
                hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
            ))
            fig_bar.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)")
            col1, col2 = st.columns([1, 2])
            with col2:
                st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No predictions/labels available for Alerts & Logs visualization yet.")

        # ---- Performance Metrics (Venu integrated) ----
        st.markdown("---")
        st.markdown("## 📈 Performance Metrics")
        st.markdown("Model performance and resource summary (computed from uploaded CSV if labels present).")

        # compute accuracy & confusion if both present
        if ("Prediction" in data.columns) and ("Label" in data.columns):
            try:
                y_true = data["Label"].astype(str)
                y_pred = data["Prediction"].astype(str)
                from sklearn.metrics import accuracy_score, confusion_matrix
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

        if acc_val is not None:
            st.markdown(f"### Model Accuracy: **{acc_val*100:.2f}%**")
            if conf is not None:
                cm_df = pd.DataFrame(conf, index=labels_for_cm, columns=labels_for_cm)
                fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale=["#071122", "#4895EF", "#7209B7"])
                fig_cm.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=320)
                st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("No ground truth 'Label' or model 'Prediction' available to compute accuracy/confusion matrix.")

        st.markdown("### Resource Usage (demo)")
        c1, c2, c3 = st.columns(3)
        for col, name in zip([c1, c2, c3], ["CPU Usage", "Memory Usage", "Network Throughput"]):
            val = np.random.randint(22, 78)
            col.markdown(f'<div class="metric-card"><h3>{name}</h3><h1>{val}%</h1></div>', unsafe_allow_html=True)

        # Mark monitoring done after processing
        st.session_state.monitoring_status = "Monitoring Done"

# ---------------------------
# Overview pages - 4 fully separate pages (Fazil fallback plots)
# ---------------------------
elif st.session_state.page == "Overview_Home":
    st.title("🛡️ Overview — Milestones (Home)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><h3>Dataset</h3><h1>CICIDS2017</h1><div class="small">Flow CSVs</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h3>Algorithms</h3><h1>RF / SVM / LR</h1><div class="small">with PCA</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h3>Best Model</h3><h1>Random Forest</h1><div class="small">~96% acc</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><h3>Alerts</h3><h1>Real-Time</h1><div class="small">CSV logging</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    ### How to Use Overview Section

    - Explore **EDA**, **Supervised Models**, **Anomaly Evaluation**, and **Alerts** using the overview subpages.  
    - When no file is uploaded, the system uses the **Wednesday Working Hours CSV dataset** from **CICIDS2017** as an example to demonstrate how the pipeline works. 
    - How data is cleaned and preprocessed  
    - How features are analyzed  
    - How models behave on real network traffic  
    - How anomalies and alerts are detected  
    """)


elif st.session_state.page == "Overview_EDA":
    st.title("🧩 EDA & Cleaning")
    df = st.session_state.get("df_uploaded") or st.session_state.get("csv_data")
    st.markdown("#### 1) Load Dataset & Preview")
    df = st.session_state.get("df_uploaded")

    if df is None:
        head_cols = [
            "Flow ID", "Source IP", "Source Port", "Destination IP",
            "Destination Port", "Protocol", "Timestamp", "Flow Duration",
            "Total Fwd Packets", "Total Backward Packets", "min_seg_size_forward",
            "Active Mean", "Active Std", "Active Max", "Active Min",
            "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label"
        ]
        head_rows = [
            ["192.168.10.14-209.48.71.168-49459-80-6", "192.168.10.14", 49459, "209.48.71.168", 80, 6, "5/7/2017 8:42", 38308, 1, 1, 20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "BENIGN"],
            ["192.168.10.3-192.168.10.17-389-49453-6", "192.168.10.17", 49453, "192.168.10.3", 389, 6, "5/7/2017 8:42", 479, 11, 5, 32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "BENIGN"],
            ["192.168.10.3-192.168.10.17-88-46124-6", "192.168.10.17", 46124, "192.168.10.3", 88, 6, "5/7/2017 8:42", 1095, 10, 6, 32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "BENIGN"],
            ["192.168.10.3-192.168.10.17-389-49454-6", "192.168.10.17", 49454, "192.168.10.3", 389, 6, "5/7/2017 8:42", 15206, 17, 12, 32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "BENIGN"],
            ["192.168.10.3-192.168.10.17-88-46126-6", "192.168.10.17", 46126, "192.168.10.3", 88, 6, "5/7/2017 8:42", 1092, 9, 6, 32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "BENIGN"],
        ]
        st.dataframe(pd.DataFrame(head_rows, columns=head_cols), use_container_width=True, hide_index=True)
    else:
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.markdown("#### 2) Shape, Columns, Data Types, and Summary")
    if df is None:
        st.write("Shape of dataset: (50000, 85)")
        st.text("<class 'pandas.core.frame.DataFrame'>\nRangeIndex: 50000 entries, 0 to 49999\nData columns: 85\nfloat64(37), int64(43), object(5)")
    else:
        st.write(f"Shape of dataset: {df.shape}")
        st.code(list(df.columns))
        st.text(df_info_str(df))


    st.markdown("#### 3) Class Distribution (Attack vs Normal)")
    if df is None:
        class_counts = pd.Series({"BENIGN": 47339, "DoS slowloris": 2661})
    else:
        label_col = find_label_col(df)
        if label_col is None:
            class_counts = pd.Series({"BENIGN": 47339, "DoS slowloris": 2661})
        else:
            class_counts = df[label_col].value_counts().sort_values(ascending=False)
    st.code(class_counts.to_string())
    st.plotly_chart(
        px.bar(
            x=class_counts.index.astype(str), y=class_counts.values, text=class_counts.values,
            color=class_counts.index.astype(str), color_discrete_sequence=["#00eaff", "#ff4d6d", "#F4A261", "#9b5de5", "#2a9d8f"]
        ).update_layout(plot_bgcolor="rgba(0,0,0,0)"),
        use_container_width=True,
    )

    st.markdown("#### 4) Identify Skewed Features (Top 10)")
    if df is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        st.code(df[num_cols].skew(numeric_only=True).sort_values(ascending=False).head(10).to_string())
    else:
        st.code(
            " Total Backward Packets 66.798718\n Subflow Bwd Packets 66.798718\n Bwd Header Length 66.789454\n Fwd Header Length.1 66.470562\n Fwd Header Length 66.470562\n Total Fwd Packets 66.315142\nSubflow Fwd Packets 66.315142\n act_data_pkt_fwd 66.295197\n Subflow Bwd Bytes 66.071021\n Total Length of Bwd Packets 66.067813"
        )

    st.markdown("#### 5) Outliers — Boxplots ")
    def _m1_box_fallback_local(local_df):
        if local_df is None:
            example = pd.DataFrame({
                "Source Port": np.random.randint(0, 65535, 1000),
                "Destination Port": np.random.randint(0, 65535, 1000),
                "Protocol": np.random.choice([0,6,17], 1000),
                "Flow Duration": np.random.randint(0, 1_000_000, 1000),
                "Total Fwd Packets": np.random.randint(1, 20000, 1000),
            })
            cc = st.columns(5)
            for i, c in enumerate(example.columns):
                fig = px.box(example, y=c, points="outliers", template="plotly_dark")
                fig.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)")
                cc[i].plotly_chart(fig, use_container_width=True)
        else:
            targets = ["Source Port", "Destination Port", "Protocol", "Flow Duration", "Total Fwd Packets"]
            colmap = {c.strip(): c for c in local_df.columns}
            cc = st.columns(5)
            for i, c in enumerate(targets):
                if c in colmap:
                    fig = px.box(local_df, y=colmap[c], points="outliers", template="plotly_dark")
                    fig.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)")
                    cc[i].plotly_chart(fig, use_container_width=True)
    _m1_box_fallback_local(df)

    st.markdown("#### 6) Correlation Heatmap")
    def _m1_heatmap_fallback_local(local_df):
        if local_df is None:
            tmp = np.random.rand(12,12)
            fig_hm = px.imshow(tmp, color_continuous_scale="RdBu", aspect="auto")
        else:
            num_cols = local_df.select_dtypes(include=[np.number]).columns
            corr = local_df[num_cols].corr()
            fig_hm = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
        fig_hm.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=520)
        st.plotly_chart(fig_hm, use_container_width=True)
    _m1_heatmap_fallback_local(df)

    st.markdown("#### 7) Cleaning Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown('<div class="metric-card"><h3>Total Rows</h3><h1>50,000</h1></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><h3>Columns</h3><h1>85 ➜ 75</h1></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><h3>Missing</h3><h1>Flow Bytes/s: 9</h1></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card"><h3>Duplicates</h3><h1>0</h1></div>', unsafe_allow_html=True)
    st.caption("Removed constants include: Bwd/ Fwd URG/PSH Flags, CWE Flag Count, Fwd/Bwd Avg * Bulk ...")
    
    st.markdown("#### 8) Split Shapes")
    st.code("X_train: (35000, 74)\nX_test: (15000, 74)\ny_train: (35000,)\ny_test: (15000,)")

elif st.session_state.page == "Overview_Supervised":
    st.title("🤖 Supervised Models")
    st.markdown("### Feature Selection, PCA & Model Training (Summary)")
    default_metrics = {
        "milestone2": {"numeric_features": 78, "pca_95": 22, "pca_components": {"RF": 22, "SVM": 11, "LR": 14}},
        "acc": {"Random Forest": 96.09, "Linear SVM": 94.87, "Logistic Regression": 92.96},
        "rf_report": {
            "0": {"precision": 0.99, "recall": 0.95, "f1": 0.97, "support": 88006},
            "1": {"precision": 0.55, "recall": 0.98, "f1": 0.71, "support": 2059},
            "2": {"precision": 0.95, "recall": 0.97, "f1": 0.96, "support": 46215},
            "accuracy": 0.9609,
        },
        "rf_top10": [
            ("PC2", 0.176918), ("PC3", 0.121040), ("PC9", 0.089699), ("PC8", 0.086322),
            ("PC13", 0.062612), ("PC15", 0.060300), ("PC20", 0.057858),
            ("PC14", 0.044498), ("PC16", 0.036611), ("PC1", 0.032558),
        ],
    }

    numeric_features = default_metrics["milestone2"]["numeric_features"]
    pca_95 = default_metrics["milestone2"]["pca_95"]
    pca_components = default_metrics["milestone2"]["pca_components"]
    acc_map = default_metrics["acc"]
    rf_rep = default_metrics["rf_report"]
    rf_top10 = default_metrics["rf_top10"]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>Numeric Features</h3><h1>{numeric_features}</h1></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>PCA 95% Var</h3><h1>{pca_95}</h1></div>', unsafe_allow_html=True)
    with c3:
        best = max(acc_map, key=acc_map.get)
        st.markdown(f'<div class="metric-card"><h3>Best Model</h3><h1>{best}</h1></div>', unsafe_allow_html=True)

    st.markdown("#### Accuracy Leaderboard")
    fig = px.bar(pd.DataFrame({"Model": list(acc_map.keys()), "Accuracy": list(acc_map.values())}),
                 x="Model", y="Accuracy", color="Accuracy", text_auto=True, color_continuous_scale="Viridis")
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Random Forest — Detailed Report")
    rows = []
    for k, v in rf_rep.items():
        if k == "accuracy": continue
        rows.append([k, v["precision"], v["recall"], v["f1"], v["support"]])
    st.dataframe(pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1", "Support"]),
                 use_container_width=True, hide_index=True)
    st.caption(f"Overall Accuracy: {rf_rep.get('accuracy', 0.9609):.4f}")


    st.markdown("#### RF Top Features")
    top10 = pd.DataFrame(rf_top10, columns=["Feature", "Importance"])
    fig_imp = px.bar(top10.sort_values("Importance"), x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Blues")
    fig_imp.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("#### PCA Components used per Model")
    st.table(pd.DataFrame([pca_components]).T.rename(columns={0: "n_components"}))


    st.markdown("#### Confusion Matrices")
    cm = np.array([[9850, 120], [30, 1000]])
    fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Normal", "Attack"], y=["Normal", "Attack"],
                    color_continuous_scale=["#e0f3ff", "#4CC9F0", "#7209B7"], text_auto=True)
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=360)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### ROC Curves")
    st.markdown('<div class="img-glow-wrap">', unsafe_allow_html=True)
    st.image("Screenshot 2025-11-22 204720.png", width=None)
    st.markdown('</div>', unsafe_allow_html=True)



elif st.session_state.page == "Overview_Anomaly":
    st.title("🔍 Anomaly & Evaluation")
    km_normal = 685331
    km_anom = 7355
    iso_normal = 685253
    iso_anom = 7433

    st.markdown("#### Anomaly Ratios")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### K-Means")
        fig_k = px.pie(values=[km_normal, km_anom], names=["Normal", "Anomaly"],
                       color_discrete_sequence=["#18ffff", "#ff4d6d"])
        fig_k.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_k, use_container_width=True)
        st.code(f"Normal: {km_normal:,}\nAnomaly: {km_anom:,}\nAnomaly %: {(km_anom/(km_normal+km_anom))*100:.2f}%")
    with c2:
        st.markdown("#### Isolation Forest")
        fig_i = px.pie(values=[iso_normal, iso_anom], names=["Normal", "Anomaly"],
                       color_discrete_sequence=["#00eaff", "#ff5252"])
        fig_i.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_i, use_container_width=True)
        st.code(f"Normal: {iso_normal:,}\nAnomaly: {iso_anom:,}\nAnomaly %: {(iso_anom/(iso_normal+iso_anom))*100:.2f}%")

    st.markdown("### PCA 2D Scatter")
    # ----------- LOAD YOUR TWO IMAGES -----------
    # Replace with your actual file paths
    kmeans_img = "Screenshot 2025-11-22 224905.png"     # your first image
    iso_img    = "Screenshot 2025-11-22 224923.png"  # your second image

    # ----------- DISPLAY SIDE-BY-SIDE -----------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#####  K-Means Anomaly Detection")
        st.image(kmeans_img, use_column_width=True)

    with col2:
        st.markdown("#####  Isolation Forest Anomaly Detection")
        st.image(iso_img, use_column_width=True)


    st.markdown("#### Final Model Evaluation (After PCA)")
    st.plotly_chart(px.bar(pd.DataFrame({"Model": ["Random Forest", "SVM", "Logistic Regression"], "Accuracy": [96.21, 90.07, 90.78]}),
                          x="Model", y="Accuracy", color="Accuracy", text_auto=True).update_layout(plot_bgcolor="rgba(0,0,0,0)"), use_container_width=True)

elif st.session_state.page == "Overview_Alerts":
    st.title("🚨 Alerts (Overview)")
    st.markdown("---")
    if "alerts_df" in st.session_state:
        alerts_df = st.session_state["alerts_df"]
        counts = alerts_df["Alert_Message"].value_counts()
        normal = int(counts.get("✅ Normal Activity", 0))
        intr = int(counts.get("🚨 Intrusion Detected", 0))
        total = normal + intr
    else:
        normal, intr = 120_140, 2_019
        total = normal + intr

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="metric-card"><h3>Test Samples</h3><h1>{total:,}</h1></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><h3>Normal</h3><h1 class="ok">{normal:,}</h1></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><h3>Intrusions</h3><h1 class="warn">{intr:,}</h1></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(" ### Alert Table  ")
    if "alerts_df" in st.session_state:
        alerts_df = st.session_state["alerts_df"]
        st.dataframe(alerts_df.head(200), use_container_width=True)
    else:
        demo = pd.DataFrame({
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * 6,
            "Actual_Label": [0, 0, 2, 0, 0, 2],
            "Predicted_Label": [0, 0, 2, 0, 0, 2],
            "Alert_Message": ["✅ Normal Activity", "✅ Normal Activity", "🚨 Intrusion Detected", "✅ Normal Activity", "✅ Normal Activity", "🚨 Intrusion Detected"]
        })
        st.dataframe(demo, use_container_width=True)
    st.markdown("### Alert Distribution ")
    st.plotly_chart(px.pie(values=[120140, 2019], names=["✅ Normal Activity", "🚨 Intrusion Detected"]).update_layout(plot_bgcolor="rgba(0,0,0,0)"), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Real-Time Simulation (Demo)")
    with st.expander("Run 10 live predictions (demo labels)"):
        for i in range(10):
            lbl = "✅ Normal Activity" if i < 9 else "🚨 Intrusion Detected"
            st.write(f"Sample {i+1}: {lbl}")
            time.sleep(0.15)


# ---------------------------
# Footer (common)
# ---------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #00ffff; padding: 12px 0;'>
        <p style='font-size: 0.95rem; font-weight: 600; text-transform:uppercase; letter-spacing:1px;'>
            🛡️ SentinelNet AI-Powered Network Intrusion Detection System
        </p>
        <p style='font-size: 0.85rem; color:#ff00ff; text-transform:uppercase;'>
            © 2025 SentinelNet | Real-time Threat Detection
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
