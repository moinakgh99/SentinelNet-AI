# =============================================================
# SentinelNet NIDS — Milestones Dashboard (Streamlit, single file)
# Bulk upload your Colab outputs once (CSV / JSON / PNG / JPG / ZIP / IPYNB).
# The app auto-detects + lets you MAP notebook images to the right places,
# then all milestones are populated with your real outputs.
# Run: streamlit run app.py
# =============================================================

import base64
import io
import json
import os
import re
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="SentinelNet NIDS — Milestones",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# Cyberpunk CSS
# -------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;700&display=swap');
* { font-family: 'Rajdhani', sans-serif; }
body, .main { background: linear-gradient(180deg, #0A0F1F 0%, #1A1037 100%); color: #E0E0E0; }
h1, h2, h3 { font-family: 'Orbitron', monospace !important; text-transform: uppercase; letter-spacing: 2px; }
h1 { color: #00E5FF; text-shadow: 0 0 12px rgba(0, 229, 255, 0.45); font-weight: 900; }
h2 { color: #7F5AF0; text-shadow: 0 0 10px rgba(127, 90, 240, 0.4); font-weight: 700; }
h3 { color: #4DD0E1; text-shadow: 0 0 8px rgba(77, 208, 225, 0.4); font-weight: 600; }
.metric-card {
  background: linear-gradient(135deg, rgba(0, 229, 255, 0.08), rgba(127, 90, 240, 0.08));
  border: 2px solid #00E5FF; padding: 18px; border-radius: 12px; text-align: center;
  transition: all 0.35s ease; box-shadow: 0 0 18px rgba(0, 229, 255, 0.3);
}
.metric-card h1 { font-size: 2rem; margin: 8px 0; color: #14FFEC; }
[data-testid="stSidebar"] {
  background: rgba(10, 15, 31, 0.9); border-right: 2px solid rgba(20, 255, 236, 0.4);
  backdrop-filter: blur(10px);
}
.stDataFrame { background: rgba(15, 20, 40, 0.65); border: 1px solid rgba(20, 255, 236, 0.35); border-radius: 6px; }
::-webkit-scrollbar { width: 10px; } ::-webkit-scrollbar-thumb { background: linear-gradient(#14FFEC, #7F5AF0); border-radius: 5px; }
.small { color: #b8d7ff; font-size: 0.9rem; }
.warn { color: #F4A261; } .ok { color: #14FFEC; }
</style>
""",
    unsafe_allow_html=True,
)

# ===============================
# Helpers
# ===============================
SLOTS = [
    # Milestone-1
    ("m1_boxplot", "M1 — Boxplots"),
    ("m1_heatmap", "M1 — Correlation Heatmap"),
    # Milestone-2
    ("m2_acc_image", "M2 — Accuracy Leaderboard"),
    ("m2_feat_image", "M2 — RF Top Features"),
    ("m2_cm_image", "M2 — Confusion Matrices"),
    ("m2_roc_image", "M2 — ROC Curves"),
    # Milestone-3
    ("m3_pca", "M3 — PCA 2D Scatter"),
    ("m3_km", "M3 — KMeans Clusters"),
    ("m3_iso", "M3 — Isolation Forest"),
    # Milestone-4
    ("m4_pie", "M4 — Alert Distribution"),
]

SLOT_KEYS = [k for k, _ in SLOTS]


def df_info_str(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return buf.getvalue()


def find_label_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if c.strip().lower() == "label":
            return c
    return None


def safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(path, engine="python")
        except Exception:
            return None


def write_upload_to_tempfile(uploaded_file) -> Path:
    """Persist an uploaded file to a temp directory and return its Path."""
    tmpdir = Path(tempfile.mkdtemp(prefix="sentinelnet_"))
    path = tmpdir / uploaded_file.name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def extract_zip_to_temp(path: Path) -> list[Path]:
    """Extract a ZIP to a temp dir and return all contained file paths."""
    root = Path(tempfile.mkdtemp(prefix="sn_zip_"))
    with zipfile.ZipFile(path, "r") as z:
        z.extractall(root)
    return [p for p in root.rglob("*") if p.is_file()]


def map_image_to_slot(name: str) -> str | None:
    """Heuristic filename -> slot mapping."""
    n = name.lower()
    if re.search(r"box|outlier|boxplot", n): return "m1_boxplot"
    if re.search(r"heatmap|corr|correlation", n): return "m1_heatmap"
    if re.search(r"acc|accuracy|leaderboard", n): return "m2_acc_image"
    if re.search(r"(feature|importance)", n): return "m2_feat_image"
    if re.search(r"roc", n): return "m2_roc_image"
    if re.search(r"confusion|cmatrix|cm", n): return "m2_cm_image"
    if re.search(r"pca.*scatter|pcascatter|pc2|pc1", n): return "m3_pca"
    if re.search(r"kmeans|k-means|cluster", n): return "m3_km"
    if re.search(r"isoforest|isolation", n): return "m3_iso"
    if re.search(r"alert.*pie|alert.*chart|distribution", n): return "m4_pie"
    return None


def parse_ipynb_images(path: Path) -> list[dict]:
    """
    Parse a .ipynb and extract embedded images (base64) + guess a slot
    using nearby markdown/cell source keywords. Return list of dicts:
      { "path": <saved_png_path>, "guess": <slot or None>, "label": <str> }
    """
    out = []
    try:
        nb = json.loads(Path(path).read_text(encoding="utf-8"))
        cells = nb.get("cells", [])
    except Exception:
        return out

    # create temp dir to store images
    img_dir = Path(tempfile.mkdtemp(prefix="sn_nb_img_"))

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        src_text = " ".join(cell.get("source", []))[:200].lower()
        nearby_text = src_text
        # look back one markdown cell to get context
        if idx > 0 and cells[idx - 1].get("cell_type") == "markdown":
            nearby_text = " ".join(cells[idx - 1].get("source", [])).lower()

        for outp in outputs:
            data = outp.get("data", {})
            if "image/png" in data:
                b64 = data["image/png"]
                try:
                    img_bytes = base64.b64decode(b64)
                    p = img_dir / f"nb_img_{idx}_{len(out)}.png"
                    with open(p, "wb") as f:
                        f.write(img_bytes)
                    # guess slot from context or filename-like hints in text
                    guess = None
                    for k, _title in SLOTS:
                        key = k.replace("_", " ")
                        if key.split()[1] in nearby_text or key.split()[-1] in nearby_text:
                            guess = k
                            break
                    if not guess:
                        guess = map_image_to_slot("".join(nearby_text.split()[:10]))
                    label = "Notebook image"
                    out.append({"path": str(p), "guess": guess, "label": label})
                except Exception:
                    pass
            # also grab JSON-like metrics printed as application/json
            if "application/json" in data:
                try:
                    m = data["application/json"]
                    if isinstance(m, dict):
                        # Merge as metrics
                        st.session_state["m2_metrics"] = {**st.session_state.get("m2_metrics", {}), **m}
                except Exception:
                    pass
            # plain text outputs that look like JSON
            if "text/plain" in data:
                txt = "".join(data["text/plain"])
                if txt.strip().startswith("{") and txt.strip().endswith("}"):
                    try:
                        m = json.loads(txt)
                        if isinstance(m, dict):
                            st.session_state["m2_metrics"] = {**st.session_state.get("m2_metrics", {}), **m}
                    except Exception:
                        pass
    return out


def show_img_or_plot(session_key_or_path: str, fallback_plot_fn, caption=None):
    """Show image if we have a path in session; otherwise call fallback plot."""
    obj = st.session_state.get(session_key_or_path)
    if obj:
        st.image(obj, use_column_width=True, caption=caption)
    else:
        fallback_plot_fn()


def metrics_get(path, default=None):
    """metrics.json dotted getter."""
    m = st.session_state.get("m2_metrics")
    if not m:
        return default
    try:
        cur = m
        for part in path.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return default
        return default if cur is None else cur
    except Exception:
        return default


# ===============================
# Sidebar (Bulk Upload + nav + mapping)
# ===============================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/240/shield.png", width=96)
    st.title("🛡️ SentinelNet NIDS")

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🧩 EDA & Cleaning",
            "🤖 Supervised Models",
            "🔍 Anomaly & Evaluation",
            "🚨 Alerts",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Bulk Upload (All Files Once)")

    up_files = st.file_uploader(
        "Upload multiple Colab files (CSV/JSON/PNG/JPG/IPYNB) or a ZIP",
        accept_multiple_files=True,
        type=["csv", "json", "png", "jpg", "jpeg", "ipynb", "zip"],
        key="bulk_multi",
    )

    # Store extracted notebook images before mapping
    if "nb_images_pool" not in st.session_state:
        st.session_state["nb_images_pool"] = []

    if up_files:
        saved_paths: list[Path] = []
        for up in up_files:
            p = write_upload_to_tempfile(up)
            saved_paths.append(p)
            # expand ZIPs
            if p.suffix.lower() == ".zip":
                saved_paths.extend(extract_zip_to_temp(p))

        # Ingest CSV/JSON/PNGs directly
        csvs = [p for p in saved_paths if p.suffix.lower() == ".csv"]
        jsons = [p for p in saved_paths if p.suffix.lower() == ".json"]
        imgs = [p for p in saved_paths if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
        ipynbs = [p for p in saved_paths if p.suffix.lower() == ".ipynb"]

        # Alerts CSV + Dataset CSV
        if csvs:
            alerts_csvs = [p for p in csvs if re.search(r"alerts|alert_log|alerts_log", p.name, re.I)]
            if alerts_csvs:
                df_alerts = safe_read_csv(alerts_csvs[0])
                if df_alerts is not None:
                    st.session_state["alerts_df"] = df_alerts
            dataset_csvs = [p for p in csvs if p not in alerts_csvs]
            if dataset_csvs:
                df_dataset = safe_read_csv(dataset_csvs[0])
                if df_dataset is not None:
                    df_dataset.columns = [c.strip() for c in df_dataset.columns]
                    st.session_state["df_uploaded"] = df_dataset

        # metrics.json
        for j in jsons:
            try:
                with open(j, "r", encoding="utf-8") as f:
                    st.session_state["m2_metrics"] = json.load(f)
                    break
            except Exception:
                pass

        # direct images (map automatically by filename)
        for img in imgs:
            slot = map_image_to_slot(img.name)
            if slot:
                st.session_state[slot] = str(img)

        # parse .ipynb images/metrics to a pool for mapping
        nb_pool = []
        for nb in ipynbs:
            nb_pool.extend(parse_ipynb_images(nb))
        if nb_pool:
            st.session_state["nb_images_pool"].extend(nb_pool)

        st.success(f"Ingested {len(saved_paths)} file(s). Scroll down to 'Notebook Image Mapping' to assign plots.")

    # Notebook image mapping UI
    if st.session_state["nb_images_pool"]:
        st.markdown("---")
        st.markdown("### Notebook Image Mapping")
        st.caption("Assign extracted notebook plots to dashboard placeholders.")

        new_pool = []
        for i, item in enumerate(st.session_state["nb_images_pool"], start=1):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.image(item["path"], caption=f"Notebook image #{i}", use_column_width=True)
            with c2:
                guess = item.get("guess")
                slot = st.selectbox(
                    "Map to slot",
                    ["(skip)"] + SLOT_KEYS,
                    index=(SLOT_KEYS.index(guess) + 1) if guess in SLOT_KEYS else 0,
                    key=f"map_{i}",
                )
                if slot != "(skip)":
                    st.session_state[slot] = item["path"]
                else:
                    new_pool.append(item)
        st.session_state["nb_images_pool"] = new_pool
        if not new_pool:
            st.success("All notebook images mapped. You can collapse this section.")

# ===============================
# Overview
# ===============================
if page == "🏠 Overview":
    st.title("🛡️ SentinelNet — Network Intrusion Detection (Milestones)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card"><h3>Dataset</h3><h1>CICIDS2017</h1><div class="small">Flow CSVs</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h3>Algorithms</h3><h1>RF / SVM / LR</h1><div class="small">with PCA</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h3>Best Model</h3><h1>Random Forest</h1><div class="small">~96% acc</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card"><h3>Alerts</h3><h1>Real-Time</h1><div class="small">CSV logging</div></div>', unsafe_allow_html=True)

   
# ===============================
# Milestone 1 — EDA & Cleaning
# ===============================
elif page == "🧩 EDA & Cleaning":
    st.title("🧩 Dataset Acquisition, EDA & Preprocessing")

    # 1) Load & Preview
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
        st.caption("Upload dataset CSV (via bulk upload) to replace the preview.")
    else:
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    # 2) Shape/columns/info
    st.markdown("#### 2) Shape, Columns, Data Types, and Summary")
    if df is None:
        st.write("Shape of dataset: (50000, 85)")
        st.text("<class 'pandas.core.frame.DataFrame'>\nRangeIndex: 50000 entries, 0 to 49999\nData columns: 85\nfloat64(37), int64(43), object(5)")
    else:
        st.write(f"Shape of dataset: {df.shape}")
        st.code(list(df.columns))
        st.text(df_info_str(df))

    # 3) Class distribution
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

    # 4) Skewness
    st.markdown("#### 4) Identify Skewed Features (Top 10)")
    if df is not None:
        num_cols = df.select_dtypes(include=[np.number]).columns
        st.code(df[num_cols].skew(numeric_only=True).sort_values(ascending=False).head(10).to_string())
    else:
        st.code(
            " Total Backward Packets 66.798718\n Subflow Bwd Packets 66.798718\n Bwd Header Length 66.789454\n Fwd Header Length.1 66.470562\n Fwd Header Length 66.470562\n Total Fwd Packets 66.315142\nSubflow Fwd Packets 66.315142\n act_data_pkt_fwd 66.295197\n Subflow Bwd Bytes 66.071021\n Total Length of Bwd Packets 66.067813"
        )

    # 5) Boxplots (use mapped image if available)
    st.markdown("#### 5) Outliers — Boxplots")
    def _m1_box_fallback():
        if df is None:
            example = pd.DataFrame(
                {"Source Port": np.random.randint(0, 65535, 1000),
                 "Destination Port": np.random.randint(0, 65535, 1000),
                 "Protocol": np.random.choice([0, 6, 17, 18], 1000),
                 "Flow Duration": np.random.randint(0, 1.2e8, 1000),
                 "Total Fwd Packets": np.random.randint(1, 205000, 1000)}
            )
            cols = list(example.columns)
            cc = st.columns(5)
            for i, c in enumerate(cols):
                fig = px.box(example, y=c, points="outliers", template="plotly_dark")
                fig.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)")
                cc[i].plotly_chart(fig, use_container_width=True)
        else:
            targets = ["Source Port", "Destination Port", "Protocol", "Flow Duration", "Total Fwd Packets"]
            colmap = {c.strip(): c for c in df.columns}
            cc = st.columns(5)
            for i, c in enumerate(targets):
                if c in colmap:
                    fig = px.box(df, y=colmap[c], points="outliers", template="plotly_dark")
                    fig.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)")
                    cc[i].plotly_chart(fig, use_container_width=True)
    show_img_or_plot("m1_boxplot", _m1_box_fallback, caption="Boxplots (Colab)")

    # 6) Heatmap
    st.markdown("#### 6) Correlation Heatmap")
    def _m1_heatmap_fallback():
        if df is None:
            tmp = np.random.rand(12, 12)
            fig_hm = px.imshow(tmp, color_continuous_scale="RdBu", aspect="auto")
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns
            corr = df[num_cols].corr()
            fig_hm = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
        fig_hm.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=520)
        st.plotly_chart(fig_hm, use_container_width=True)
    show_img_or_plot("m1_heatmap", _m1_heatmap_fallback, caption="Correlation Heatmap (Colab)")

    st.markdown("---")
    st.markdown("#### 7) Cleaning Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown('<div class="metric-card"><h3>Total Rows</h3><h1>50,000</h1></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><h3>Columns</h3><h1>85 ➜ 75</h1></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><h3>Missing</h3><h1>Flow Bytes/s: 9</h1></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card"><h3>Duplicates</h3><h1>0</h1></div>', unsafe_allow_html=True)
    st.caption("Removed constants include: Bwd/ Fwd URG/PSH Flags, CWE Flag Count, Fwd/Bwd Avg * Bulk ...")
    st.markdown("#### 8) Split Shapes")
    st.code("X_train: (35000, 74)\nX_test: (15000, 74)\ny_train: (35000,)\ny_test: (15000,)")

# ===============================
# Milestone 2 — Supervised Models
# ===============================
elif page == "🤖 Supervised Models":
    st.title("🤖 Feature Selection, PCA & Model Training")

    default_metrics = {
        "milestone2": {"numeric_features": 78, "pca_95": 22, "pca_components": {"RF": 22, "SVM": 11, "LR": 14}},
        "acc": {"Random Forest": 96.09, "Linear SVM": 94.87, "Logistic Regression": 92.96},
        "rf_report": {
            "0": {"precision": 0.99, "recall": 0.95, "f1": 0.97, "support": 88006},
            "1": {"precision": 0.55, "recall": 0.98, "f1": 0.71, "support": 2059},
            "2": {"precision": 0.95, "recall": 0.97, "f1": 0.96, "support": 46215},
            "3": {"precision": 0.85, "recall": 0.97, "f1": 0.90, "support": 1100},
            "4": {"precision": 0.62, "recall": 0.97, "f1": 0.76, "support": 1159},
            "5": {"precision": 0.50, "recall": 1.00, "f1": 0.67, "support": 2},
            "accuracy": 0.9609,
        },
        "rf_top10": [
            ("PC2", 0.176918), ("PC3", 0.121040), ("PC9", 0.089699), ("PC8", 0.086322),
            ("PC13", 0.062612), ("PC15", 0.060300), ("PC20", 0.057858),
            ("PC14", 0.044498), ("PC16", 0.036611), ("PC1", 0.032558),
        ],
    }

    numeric_features = metrics_get("milestone2.numeric_features", default_metrics["milestone2"]["numeric_features"])
    pca_95 = metrics_get("milestone2.pca_95", default_metrics["milestone2"]["pca_95"])
    pca_components = metrics_get("milestone2.pca_components", default_metrics["milestone2"]["pca_components"])
    acc_map = metrics_get("acc", default_metrics["acc"])
    rf_rep = metrics_get("rf_report", default_metrics["rf_report"])
    rf_top10 = metrics_get("rf_top10", default_metrics["rf_top10"])

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="metric-card"><h3>Numeric Features</h3><h1>{numeric_features}</h1></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><h3>PCA 95% Var</h3><h1>{pca_95}</h1></div>', unsafe_allow_html=True)
    with c3:
        best = max(acc_map, key=acc_map.get)
        st.markdown(f'<div class="metric-card"><h3>Best Model</h3><h1>{best}</h1></div>', unsafe_allow_html=True)

    st.markdown("#### Accuracy Leaderboard")
    def _acc_fallback():
        fig = px.bar(pd.DataFrame({"Model": list(acc_map.keys()), "Accuracy": list(acc_map.values())}),
                     x="Model", y="Accuracy", color="Accuracy", text_auto=True, color_continuous_scale="Viridis")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    show_img_or_plot("m2_acc_image", _acc_fallback, caption="Accuracy (Colab)")

    st.markdown("#### Random Forest — Detailed Report")
    rows = []
    for k, v in rf_rep.items():
        if k == "accuracy": continue
        rows.append([k, v["precision"], v["recall"], v["f1"], v["support"]])
    st.dataframe(pd.DataFrame(rows, columns=["Class", "Precision", "Recall", "F1", "Support"]),
                 use_container_width=True, hide_index=True)
    st.caption(f"Overall Accuracy: {rf_rep.get('accuracy', 0.9609):.4f}")

    st.markdown("#### RF Top Features")
    def _feat_fallback():
        top10 = pd.DataFrame(rf_top10, columns=["Feature", "Importance"])
        fig_imp = px.bar(top10.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues")
        fig_imp.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_imp, use_container_width=True)
    show_img_or_plot("m2_feat_image", _feat_fallback, caption="RF Top Features (Colab)")

    st.markdown("#### PCA Components used per Model")
    st.table(pd.DataFrame([pca_components]).T.rename(columns={0: "n_components"}))

    st.markdown("#### Confusion Matrices")
    def _cm_fallback():
        cm = np.array([[9850, 120], [30, 1000]])
        fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Normal", "Attack"], y=["Normal", "Attack"],
                        color_continuous_scale=["#e0f3ff", "#4CC9F0", "#7209B7"], text_auto=True)
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", height=360)
        st.plotly_chart(fig, use_container_width=True)
    show_img_or_plot("m2_cm_image", _cm_fallback, caption="Confusion Matrices (Colab)")

    st.markdown("#### ROC Curves")
    def _roc_fallback():
        roc_df = pd.DataFrame({"Model": ["Random Forest", "SVM", "Logistic Regression"], "AUC": [0.987, 0.93, 0.91]})
        fig_roc = px.line(x=np.linspace(0, 1, 100), y=np.linspace(0, 1, 100),
                          labels={"x": "False Positive Rate", "y": "True Positive Rate"})
        for mname, auc in roc_df.values:
            x = np.linspace(0, 1, 100)
            y = np.clip(x ** (1 / (auc * 2)), 0, 1)
            fig_roc.add_scatter(x=x, y=y, mode="lines", name=f"{mname} (AUC={auc:.3f})")
        fig_roc.update_traces(line=dict(width=3))
        fig_roc.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_roc, use_container_width=True)
    show_img_or_plot("m2_roc_image", _roc_fallback, caption="ROC Curves (Colab)")

# ===============================
# Milestone 3 — Anomaly & Eval
# ===============================
elif page == "🔍 Anomaly & Evaluation":
    st.title("🔍 Unsupervised Detection & Model Evaluation")

    km_normal = metrics_get("milestone3.kmeans.normal", 685331)
    km_anom = metrics_get("milestone3.kmeans.anomaly", 7355)
    iso_normal = metrics_get("milestone3.isoforest.normal", 685253)
    iso_anom = metrics_get("milestone3.isoforest.anomaly", 7433)

    st.markdown("#### Anomaly Ratios")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### K-Means")
        def _km_fallback():
            fig_k = px.pie(values=[km_normal, km_anom], names=["Normal", "Anomaly"],
                           color_discrete_sequence=["#18ffff", "#ff4d6d"])
            fig_k.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_k, use_container_width=True)
        show_img_or_plot("m3_km", _km_fallback, caption="K-Means (Colab)")
        st.code(f"Normal: {km_normal:,}\nAnomaly: {km_anom:,}\nAnomaly %: {(km_anom/(km_normal+km_anom))*100:.2f}%")
    with c2:
        st.markdown("##### Isolation Forest")
        def _iso_fallback():
            fig_i = px.pie(values=[iso_normal, iso_anom], names=["Normal", "Anomaly"],
                           color_discrete_sequence=["#00eaff", "#ff5252"])
            fig_i.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_i, use_container_width=True)
        show_img_or_plot("m3_iso", _iso_fallback, caption="Isolation Forest (Colab)")
        st.code(f"Normal: {iso_normal:,}\nAnomaly: {iso_anom:,}\nAnomaly %: {(iso_anom/(iso_normal+iso_anom))*100:.2f}%")

    st.markdown("#### PCA 2D Scatter")
    def _pca_fallback():
        st.info("Upload PCA scatter image (via notebook or direct file) to display the exact plot.")
    show_img_or_plot("m3_pca", _pca_fallback, caption="PCA Scatter (Colab)")

    rf_acc = metrics_get("milestone3.final_acc.RF", 96.21)
    svm_acc = metrics_get("milestone3.final_acc.SVM", 90.07)
    lr_acc = metrics_get("milestone3.final_acc.LR", 90.78)
    st.markdown("#### Final Model Evaluation (After PCA)")
    st.plotly_chart(
        px.bar(pd.DataFrame({"Model": ["Random Forest", "SVM", "Logistic Regression"], "Accuracy": [rf_acc, svm_acc, lr_acc]}),
               x="Model", y="Accuracy", color="Accuracy", text_auto=True, color_continuous_scale="Viridis"
               ).update_layout(plot_bgcolor="rgba(0,0,0,0)"),
        use_container_width=True,
    )

# ===============================
# Milestone 4 — Alerts
# ===============================
elif page == "🚨 Alerts":
    st.title("🚨 Real-Time Alert Generation & Logging")

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

    def _alerts_pie_fallback():
        st.plotly_chart(
            px.pie(values=[normal, intr], names=["✅ Normal Activity", "🚨 Intrusion Detected"],
                   color_discrete_sequence=["#4CAF50", "#FF5252"]).update_layout(plot_bgcolor="rgba(0,0,0,0)"),
            use_container_width=True,
        )
    show_img_or_plot("m4_pie", _alerts_pie_fallback, caption="Alert Distribution (Colab)")

    st.markdown("#### Alerts Table")
    if "alerts_df" in st.session_state:
        st.dataframe(st.session_state["alerts_df"].head(100), use_container_width=True, hide_index=True)
    else:
        demo = pd.DataFrame(
            {"Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] * 5,
             "Actual_Label": [0, 0, 2, 0, 0],
             "Predicted_Label": [0, 0, 2, 0, 0],
             "Alert_Message": ["✅ Normal Activity"] * 5}
        )
        st.dataframe(demo, use_container_width=True, hide_index=True)
        st.caption("Include alerts_log.csv in your bulk upload to replace this.")

    st.markdown("---")
    st.markdown("#### Real-Time Simulation (demo)")
    with st.expander("Run 10 live predictions (demo labels)"):
        for i in range(10):
            lbl = "✅ Normal Activity" if i < 9 else "🚨 Intrusion Detected"
            st.write(f"Sample {i+1}: {lbl}")
            time.sleep(0.15)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #00ffff; padding: 6px 0;'>
  © 2025 SentinelNet | ML-powered NIDS | Milestones Dashboard
</div>
""",
    unsafe_allow_html=True,
)