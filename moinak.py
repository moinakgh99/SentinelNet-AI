import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="SentinelNet NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cyberpunk Neon CSS Styling
st.markdown("""
    <style>
/* Load premium fonts */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;700&display=swap');

/* General */
* {
    font-family: 'Rajdhani', sans-serif;
}

body, .main {
    background: linear-gradient(180deg, #0A0F1F 0%, #1A1037 100%);
    background-attachment: fixed;
    color: #E0E0E0;
}

/* ===== HEADINGS ===== */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    text-transform: uppercase;
    letter-spacing: 2px;
}

h1 {
    color: #00E5FF;
    text-shadow: 0 0 12px rgba(0, 229, 255, 0.45);
    font-weight: 900;
    animation: fadeIn 0.6s ease-out;
}

h2 {
    color: #7F5AF0;
    text-shadow: 0 0 10px rgba(127, 90, 240, 0.4);
    font-weight: 700;
}

h3 {
    color: #4DD0E1;
    text-shadow: 0 0 8px rgba(77, 208, 225, 0.4);
    font-weight: 600;
}

/* Smooth entry animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ===== Metric Cards ===== */
.metric-card {
    background: linear-gradient(135deg, rgba(0, 229, 255, 0.08), rgba(127, 90, 240, 0.08));
    border: 2px solid #00E5FF;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    transition: all 0.35s ease;
    box-shadow: 0 0 18px rgba(0, 229, 255, 0.3);
}

.metric-card:hover {
    transform: translateY(-6px) scale(1.02);
    border-color: #7F5AF0;
    box-shadow: 0 0 28px rgba(127, 90, 240, 0.55);
}

.metric-card h1 {
    font-size: 2.5rem;
    margin: 10px 0;
    color: #14FFEC;
}

/* Threat Variants (New Sophisticated Colors) */
.threat-high {
    border-color: #FFB703;   /* amber instead of pink */
    box-shadow: 0 0 25px rgba(255, 183, 3, 0.35);
}

.threat-medium {
    border-color: #7F5AF0;
}

.threat-low {
    border-color: #4DD0E1;
}

/* ===== Buttons ===== */
.stButton > button {
    background: linear-gradient(135deg, #14FFEC 0%, #7F5AF0 100%);
    border: none;
    padding: 12px 25px;
    border-radius: 8px;
    color: #0A0F1F;
    font-family: 'Orbitron';
    font-weight: 700;
    letter-spacing: 1px;
    transition: all 0.25s ease;
    box-shadow: 0 0 15px rgba(20, 255, 236, 0.35);
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow: 0 0 25px rgba(127, 90, 240, 0.5);
    background: linear-gradient(135deg, #7F5AF0 0%, #14FFEC 100%);
}

/* ===== Sidebar ===== */
[data-testid="stSidebar"] {
    background: rgba(10, 15, 31, 0.9);
    border-right: 2px solid rgba(20, 255, 236, 0.4);
    backdrop-filter: blur(10px);
}

[data-testid="stSidebar"] .stRadio > label {
    background: rgba(255, 255, 255, 0.03);
    padding: 10px 18px;
    border: 1px solid rgba(20, 255, 236, 0.3);
    border-radius: 6px;
    margin: 5px 0;
    color: #14FFEC;
    transition: 0.3s ease;
}

[data-testid="stSidebar"] .stRadio > label:hover {
    border-color: #7F5AF0;
    color: #7F5AF0;
    background: rgba(127, 90, 240, 0.12);
}

/* ===== DataFrames ===== */
.stDataFrame {
    background: rgba(15, 20, 40, 0.65);
    border: 1px solid rgba(20, 255, 236, 0.35);
    border-radius: 6px;
}

/* Inputs */
.stSelectbox, .stMultiSelect, .stTextInput {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 6px;
    border: 1px solid rgba(20, 255, 236, 0.3) !important;
    color: #E0E0E0 !important;
    transition: 0.3s ease;
}

.stSelectbox:hover, .stMultiSelect:hover, .stTextInput:hover {
    border-color: #7F5AF0 !important;
}

/* Alerts */
.stAlert {
    background: rgba(255, 255, 255, 0.06);
    border-left: 4px solid #14FFEC;
}

/* Smooth Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-thumb {
    background: linear-gradient(#14FFEC, #7F5AF0);
    border-radius: 5px;
}

</style>

""", unsafe_allow_html=True)

# Initialize session state
if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False
if 'alert_count' not in st.session_state:
    st.session_state.alert_count = 0

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/240/shield.png", width=100)

    st.title("🛡️ SentinelNet NIDS")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📊 Traffic Analysis", "🔍 Detection Models", "⚠️ Alerts & Logs", "📈 Performance Metrics", "⚙️ Settings"],
        index=0
    )

    st.markdown("---")
    st.markdown("### System Status")

    if st.button("🟢 Start Monitoring" if not st.session_state.monitoring_active else "🔴 Stop Monitoring"):
        st.session_state.monitoring_active = not st.session_state.monitoring_active

    if st.session_state.monitoring_active:
        st.success("System Active")
    else:
        st.warning("System Inactive")

    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Total Packets", "1,234,567")
    st.metric("Active Threats", "12")
    st.metric("Uptime", "99.8%")

# Generate sample data
def generate_traffic_data():
    np.random.seed(42)
    hours = pd.date_range(end=datetime.now(), periods=24, freq='H')
    return pd.DataFrame({
        'timestamp': hours,
        'normal': np.random.randint(1000, 5000, 24),
        'suspicious': np.random.randint(50, 500, 24),
        'malicious': np.random.randint(10, 100, 24)
    })

def generate_attack_types():
    return pd.DataFrame({
        'Attack Type': ['DDoS', 'Port Scan', 'SQL Injection', 'Brute Force', 'Malware', 'Other'],
        'Count': [45, 32, 18, 25, 12, 8],
        'Severity': ['Critical', 'High', 'Critical', 'Medium', 'Critical', 'Low']
    })

# ---------------------------
# Page: Dashboard
# ---------------------------
if page == "🏠 Dashboard":
    st.title("🛡️ SentinelNet AI-Powered Network Intrusion Detection System")
    st.markdown("### Real-Time Network Security Monitoring Dashboard")
    st.markdown("---")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
            <div class="metric-card">
                <h3>Total Traffic</h3>
                <h1>1.2M</h1>
                <p>Packets Analyzed</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="metric-card threat-high">
                <h3>Threats Detected</h3>
                <h1>140</h1>
                <p>Last 24 Hours</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="metric-card threat-medium">
                <h3>Model Accuracy</h3>
                <h1>98.7%</h1>
                <p>Detection Rate</p>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="metric-card threat-low">
                <h3>System Health</h3>
                <h1>Optimal</h1>
                <p>All Systems Go</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Real-time traffic visualization
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📈 Real-Time Traffic Flow")
        traffic_data = generate_traffic_data()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=traffic_data['timestamp'], y=traffic_data['normal'],
                                mode='lines', name='Normal Traffic',
                                line=dict(color='#4895EF', width=3),
                                fill='tozeroy'))
        fig.add_trace(go.Scatter(x=traffic_data['timestamp'], y=traffic_data['suspicious'],
                                mode='lines', name='Suspicious',
                                line=dict(color='#4CC9F0', width=3),
                                fill='tozeroy'))
        fig.add_trace(go.Scatter(x=traffic_data['timestamp'], y=traffic_data['malicious'],
                                mode='lines', name='Malicious',
                                line=dict(color='#7209B7', width=3),
                                fill='tozeroy'))

        fig.update_layout(
            plot_bgcolor='rgba(10,14,26,0.8)',
            paper_bgcolor='rgba(10,14,26,0)',
            font=dict(color='#00ffff'),
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🎯 Attack Distribution")
        attack_data = generate_attack_types()

        fig = px.pie(attack_data, values='Count', names='Attack Type',
                     color_discrete_sequence=[
                    '#4CC9F0', '#4895EF', '#7209B7',
                    '#B5179E', '#2A9D8F', '#F4A261'
                        ],hole=0.4)
        fig.update_layout(
            plot_bgcolor='rgba(10,14,26,0.8)',
            paper_bgcolor='rgba(10,14,26,0)',
            font=dict(color='#00ffff'),
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recent alerts
    st.markdown("### ⚠️ Recent Security Alerts")
    alerts = pd.DataFrame({
        'Timestamp': pd.date_range(end=datetime.now(), periods=5, freq='15min')[::-1],
        'Alert Type': ['DDoS Attack', 'Port Scan', 'SQL Injection', 'Brute Force', 'Malware Detected'],
        'Source IP': ['192.168.1.105', '10.0.0.23', '172.16.0.45', '192.168.1.78', '10.0.0.89'],
        'Severity': ['🔴 Critical', '🟡 Medium', '🔴 Critical', '🟡 Medium', '🔴 Critical'],
        'Status': ['Blocked', 'Monitoring', 'Blocked', 'Blocked', 'Quarantined']
    })

    st.dataframe(alerts, use_container_width=True, hide_index=True)

# ---------------------------
# Page: Traffic Analysis
# ---------------------------
elif page == "📊 Traffic Analysis":
    st.title("📊 Network Traffic Analysis")
    st.markdown("### Comprehensive Traffic Patterns and Behavior Analysis")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Packets/sec", "15,432", "+12%")
    with col2:
        st.metric("Bandwidth Usage", "1.2 GB/s", "+5%")
    with col3:
        st.metric("Active Connections", "4,567", "-3%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Protocol distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Protocol Distribution")
        protocols = pd.DataFrame({
            'Protocol': ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS'],
            'Percentage': [45, 25, 10, 8, 10, 2]
        })

        fig = px.bar(protocols, x='Protocol', y='Percentage',
                     color='Percentage',
                     color_continuous_scale=[[0, '#bde7fb'], [0.5, '#4895EF'], [1, '#7209B7']]
)
        fig.update_layout(
            plot_bgcolor='rgba(10,14,26,0.8)',
            paper_bgcolor='rgba(10,14,26,0)',
            font=dict(color='#00ffff'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Traffic by Port")
        ports = pd.DataFrame({
            'Port': ['80 (HTTP)', '443 (HTTPS)', '22 (SSH)', '3389 (RDP)', '53 (DNS)', 'Other'],
            'Traffic': [35000, 45000, 5000, 3000, 8000, 12000]
        })

        fig = px.treemap(ports, path=['Port'], values='Traffic',
                         color='Traffic',
                         color_continuous_scale=[[0, '#bde7fb'], [0.5, '#4CC9F0'], [1, '#2A9D8F']]
)
        fig.update_layout(
            plot_bgcolor='rgba(10,14,26,0.8)',
            paper_bgcolor='rgba(10,14,26,0)',
            font=dict(color='#00ffff'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Geographic distribution
    st.markdown("### 🌍 Geographic Traffic Distribution")
    geo_data = pd.DataFrame({
        'Country': ['USA', 'China', 'Russia', 'India', 'Germany', 'UK', 'Brazil', 'Japan'],
        'Requests': [45000, 23000, 18000, 15000, 12000, 10000, 8000, 7000],
        'Threats': [120, 340, 280, 90, 45, 38, 62, 25]
    })

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(geo_data, x='Country', y='Requests',
                     color='Requests',
                     color_continuous_scale=[[0, '#bde7fb'], [1, '#4895EF']]
)
        fig.update_layout(
            plot_bgcolor='rgba(10,14,26,0.8)',
            paper_bgcolor='rgba(10,14,26,0)',
            font=dict(color='#00ffff'),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(geo_data, x='Country', y='Threats',
                     color='Threats',
                     color_continuous_scale=[[0, '#F4A261'], [1, '#7209B7']]
)
        fig.update_layout(
            plot_bgcolor='rgba(10,14,26,0.8)',
            paper_bgcolor='rgba(10,14,26,0)',
            font=dict(color='#00ffff'),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Page: Detection Models
# ---------------------------
elif page == "🔍 Detection Models":
    st.title("🔍 AI Detection Models")
    st.markdown("### Machine Learning Model Performance & Comparison")
    st.markdown("---")

    # Model comparison
    models = pd.DataFrame({
        'Model': ['Random Forest', 'Decision Tree', 'SVM', 'Neural Network', 'XGBoost'],
        'Accuracy': [98.7, 95.3, 96.8, 97.9, 98.2],
        'Precision': [98.2, 94.8, 96.2, 97.5, 97.8],
        'Recall': [97.9, 95.1, 96.5, 97.3, 97.9],
        'F1-Score': [98.0, 95.0, 96.4, 97.4, 97.9],
        'Training Time (s)': [45, 12, 120, 180, 60]
    })

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📊 Model Performance Comparison")

        fig = go.Figure()
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#4CC9F0', '#4895EF', '#7209B7', '#2A9D8F']


        for metric, color in zip(metrics, colors):
            fig.add_trace(go.Bar(
                name=metric,
                x=models['Model'],
                y=models[metric],
                text=models[metric],
                textposition='auto',
                marker_color=color
            ))

        fig.update_layout(
            barmode='group',
            plot_bgcolor='rgba(10,14,26,0.8)',
            paper_bgcolor='rgba(10,14,26,0)',
            font=dict(color='#00ffff'),
            height=400,
            yaxis=dict(range=[90, 100])
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🏆 Best Model")
        st.markdown("""
            <div class="metric-card">
                <h2>Random Forest</h2>
                <h1>98.7%</h1>
                <p>Highest Accuracy</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("**Selected Model:** Random Forest")
        st.success("**Status:** Active & Deployed")

    st.markdown("---")

    # Detailed metrics table
    st.markdown("### 📋 Detailed Model Metrics")
    st.dataframe(models, use_container_width=True, hide_index=True)

    # Feature importance
    st.markdown("### 🎯 Feature Importance (Random Forest)")
    features = pd.DataFrame({
        'Feature': ['Packet Size', 'Duration', 'Protocol Type', 'Port Number',
                    'Flag Count', 'Byte Rate', 'Packet Rate', 'Service Type'],
        'Importance': [0.185, 0.165, 0.145, 0.125, 0.115, 0.105, 0.095, 0.065]
    }).sort_values('Importance', ascending=True)

    fig = px.bar(features, x='Importance', y='Feature', orientation='h',
                 color='Importance',
                 color_continuous_scale=[[0, '#bde7fb'], [0.5, '#4895EF'], [1, '#7209B7']]
)
    fig.update_layout(
        plot_bgcolor='rgba(10,14,26,0.8)',
        paper_bgcolor='rgba(10,14,26,0)',
        font=dict(color='#00ffff'),
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Page: Settings
# ---------------------------
elif page == "⚙️ Settings":
    st.title("⚙️ System Settings & Configuration")
    st.markdown("### Configure Detection Parameters and System Preferences")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Detection", "🔔 Notifications", "🔐 Security", "📊 Data"])

    with tab1:
        st.markdown("### Detection Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Model Selection")
            selected_model = st.selectbox(
                "Active Detection Model",
                ["Random Forest", "Decision Tree", "SVM", "Neural Network", "XGBoost"],
                index=0
            )

            st.markdown("#### Sensitivity Settings")
            sensitivity = st.slider("Detection Sensitivity", 0, 100, 75)

            threshold = st.slider("Alert Threshold", 0.0, 1.0, 0.85, 0.05)

            st.markdown("#### Detection Modes")
            real_time = st.checkbox("Real-time Detection", value=True)
            batch_mode = st.checkbox("Batch Processing", value=False)
            deep_scan = st.checkbox("Deep Packet Inspection", value=True)

        with col2:
            st.markdown("#### Traffic Filters")

            protocols = st.multiselect(
                "Monitor Protocols",
                ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "DNS", "FTP", "SSH"],
                default=["TCP", "UDP", "HTTP", "HTTPS"]
            )

            ports = st.text_input("Monitor Specific Ports (comma-separated)",
                                  "80,443,22,3389")

            st.markdown("#### Anomaly Detection")
            anomaly_detection = st.checkbox("Enable Anomaly Detection", value=True)
            behavioral_analysis = st.checkbox("Behavioral Analysis", value=True)

            st.markdown("#### Action on Detection")
            auto_block = st.checkbox("Automatically Block Threats", value=True)
            quarantine = st.checkbox("Quarantine Suspicious Traffic", value=False)

        if st.button("💾 Save Detection Settings", type="primary"):
            st.success("✅ Detection settings saved successfully!")

    with tab2:
        st.markdown("### Notification Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Email Notifications")
            email_enabled = st.checkbox("Enable Email Alerts", value=True)
            email_address = st.text_input("Alert Email Address",
                                         "admin@sentinelnet.com")

            email_severity = st.multiselect(
                "Send Emails for Severity Levels",
                ["Critical", "High", "Medium", "Low"],
                default=["Critical", "High"]
            )

            st.markdown("#### SMS Notifications")
            sms_enabled = st.checkbox("Enable SMS Alerts", value=False)
            phone_number = st.text_input("Phone Number", "+1-XXX-XXX-XXXX")

        with col2:
            st.markdown("#### Dashboard Notifications")
            desktop_notif = st.checkbox("Desktop Notifications", value=True)
            sound_alerts = st.checkbox("Sound Alerts", value=True)

            st.markdown("#### Alert Frequency")
            alert_freq = st.selectbox(
                "Maximum Alert Frequency",
                ["Immediate", "Every 5 minutes", "Every 15 minutes", "Hourly"],
                index=0
            )

            st.markdown("#### Notification Channels")
            slack_enabled = st.checkbox("Slack Integration", value=False)
            teams_enabled = st.checkbox("Microsoft Teams Integration", value=False)
            webhook_enabled = st.checkbox("Custom Webhook", value=False)

        if st.button("💾 Save Notification Settings", type="primary"):
            st.success("✅ Notification settings saved successfully!")

    with tab3:
        st.markdown("### Security Configuration")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Access Control")
            require_auth = st.checkbox("Require Authentication", value=True)
            two_factor = st.checkbox("Two-Factor Authentication", value=True)
            session_timeout = st.number_input("Session Timeout (minutes)",
                                             min_value=5, max_value=480, value=30)

            st.markdown("#### API Security")
            api_key_rotation = st.selectbox(
                "API Key Rotation",
                ["Never", "Every 30 days", "Every 90 days", "Every 180 days"],
                index=2
            )

            rate_limiting = st.checkbox("Enable Rate Limiting", value=True)

        with col2:
            st.markdown("#### Logging & Audit")
            audit_log = st.checkbox("Enable Audit Logging", value=True)
            log_retention = st.number_input("Log Retention (days)",
                                           min_value=7, max_value=365, value=90)

            st.markdown("#### Encryption")
            encrypt_data = st.checkbox("Encrypt Stored Data", value=True)
            encrypt_transit = st.checkbox("Encrypt Data in Transit", value=True)

            st.markdown("#### Backup")
            auto_backup = st.checkbox("Automatic Backups", value=True)
            backup_freq = st.selectbox(
                "Backup Frequency",
                ["Daily", "Weekly", "Monthly"],
                index=0
            )

        if st.button("💾 Save Security Settings", type="primary"):
            st.success("✅ Security settings saved successfully!")

    with tab4:
        st.markdown("### Data Management")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Data Collection")
            collect_metadata = st.checkbox("Collect Traffic Metadata", value=True)
            collect_payload = st.checkbox("Collect Payload Data", value=False)
            anonymize_data = st.checkbox("Anonymize IP Addresses", value=True)

            st.markdown("#### Storage Settings")
            storage_limit = st.slider("Storage Limit (GB)", 10, 1000, 500)
            compression = st.checkbox("Enable Data Compression", value=True)

            st.markdown("#### Data Export")
            export_format = st.selectbox(
                "Default Export Format",
                ["CSV", "JSON", "Parquet", "Excel"],
                index=0
            )

        with col2:
            st.markdown("#### Database Settings")
            db_type = st.selectbox(
                "Database Type",
                ["PostgreSQL", "MongoDB", "InfluxDB", "Elasticsearch"],
                index=0
            )

            connection_pool = st.number_input("Connection Pool Size",
                                             min_value=5, max_value=100, value=20)

            st.markdown("#### Data Retention")
            traffic_retention = st.number_input("Traffic Data Retention (days)",
                                               min_value=1, max_value=365, value=30)
            alert_retention = st.number_input("Alert Data Retention (days)",
                                             min_value=1, max_value=365, value=90)

            st.markdown("#### Cleanup")
            auto_cleanup = st.checkbox("Automatic Data Cleanup", value=True)

        if st.button("💾 Save Data Settings", type="primary"):
            st.success("✅ Data settings saved successfully!")

        st.markdown("---")
        st.markdown("### 🗑️ Danger Zone")
        st.warning("⚠️ These actions cannot be undone!")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Reset All Settings"):
                st.error("Settings reset initiated...")
        with col2:
            if st.button("🗑️ Clear All Logs"):
                st.error("All logs will be deleted...")
        with col3:
            if st.button("⚠️ Factory Reset"):
                st.error("System will be reset to factory defaults...")

# ---------------------------
# Page: Alerts & Logs
# ---------------------------
elif page == "⚠️ Alerts & Logs":
    st.title("⚠️ Security Alerts & System Logs")
    st.markdown("### Real-Time Threat Detection and Incident Response")
    st.markdown("---")

    # Alert summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Critical Alerts", "23", "+5")
    with col2:
        st.metric("High Priority", "45", "+12")
    with col3:
        st.metric("Medium Priority", "67", "+8")
    with col4:
        st.metric("Low Priority", "89", "+3")

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        severity_filter = st.selectbox("Filter by Severity",
                                      ["All", "Critical", "High", "Medium", "Low"])
    with col2:
        time_filter = st.selectbox("Time Range",
                                  ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"])
    with col3:
        attack_filter = st.selectbox("Attack Type",
                                    ["All", "DDoS", "Port Scan", "SQL Injection", "Brute Force", "Malware"])

    # Alert timeline
    st.markdown("### 📅 Alert Timeline")
    timeline_data = pd.DataFrame({
        'Time': pd.date_range(end=datetime.now(), periods=20, freq='30min')[::-1],
        'Alerts': np.random.randint(1, 15, 20)
    })

    fig = px.area(timeline_data, x='Time', y='Alerts',
                  color_discrete_sequence=['#4895EF']
)
    fig.update_layout(
        plot_bgcolor='rgba(10,14,26,0.8)',
        paper_bgcolor='rgba(10,14,26,0)',
        font=dict(color='#00ffff'),
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    # Detailed alerts table
    st.markdown("### 📋 Detailed Alert Log")
    alerts_detailed = pd.DataFrame({
        'Timestamp': pd.date_range(end=datetime.now(), periods=15, freq='10min')[::-1],
        'Alert ID': [f"ALT-{1000+i}" for i in range(15)],
        'Severity': np.random.choice(['🔴 Critical', '🟠 High', '🟡 Medium', '🟢 Low'], 15),
        'Attack Type': np.random.choice(['DDoS', 'Port Scan', 'SQL Injection', 'Brute Force', 'Malware'], 15),
        'Source IP': [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(15)],
        'Destination IP': [f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(15)],
        'Port': np.random.choice([80, 443, 22, 3389, 8080, 3306], 15),
        'Action Taken': np.random.choice(['Blocked', 'Quarantined', 'Monitoring', 'Logged'], 15)
    })

    st.dataframe(alerts_detailed, use_container_width=True, hide_index=True, height=400)

    # Export options
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📥 Export to CSV"):
            st.success("Alert log exported successfully!")
    with col2:
        if st.button("📄 Generate Report"):
            st.success("Security report generated!")
    with col3:
        if st.button("📧 Send Email Alert"):
            st.success("Email notification sent!")

# ---------------------------
# Page: Performance Metrics
# ---------------------------
elif page == "📈 Performance Metrics":
    st.title("📈 System Performance Metrics")
    st.markdown("### Comprehensive System Health and Performance Analysis")
    st.markdown("---")

    # System health metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("CPU Usage", "45%", "-5%")
    with col2:
        st.metric("Memory Usage", "62%", "+3%")
    with col3:
        st.metric("Network I/O", "1.2 GB/s", "+15%")
    with col4:
        st.metric("Disk Usage", "78%", "+2%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Performance over time
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🖥️ CPU & Memory Usage")
        time_data = pd.date_range(end=datetime.now(), periods=50, freq='1min')
        cpu_data = 40 + 10 * np.sin(np.linspace(0, 4*np.pi, 50)) + np.random.randn(50) * 2
        mem_data = 60 + 5 * np.cos(np.linspace(0, 4*np.pi, 50)) + np.random.randn(50) * 2

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_data, y=cpu_data, mode='lines',
                                 name='CPU Usage (%)', line=dict(color='#4CC9F0', width=2)))
        fig.add_trace(go.Scatter(x=time_data, y=mem_data, mode='lines',
                                 name='Memory Usage (%)', line=dict(color='#7209B7', width=2)))

        fig.update_layout(
            plot_bgcolor='rgba(10,14,26,0.8)',
            paper_bgcolor='rgba(10,14,26,0)',
            font=dict(color='#00ffff'),
            height=350,
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📡 Network Throughput")
        throughput_data = 1.0 + 0.3 * np.sin(np.linspace(0, 4*np.pi, 50)) + np.random.randn(50) * 0.1

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_data, y=throughput_data, mode='lines',
                                 name='Throughput (GB/s)',
                                 line=dict(color='#4895EF', width=2),
                                 fill='tozeroy'))

        fig.update_layout(
            plot_bgcolor='rgba(10,14,26,0.8)',
            paper_bgcolor='rgba(10,14,26,0)',
            font=dict(color='#00ffff'),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

    # Detection performance
    st.markdown("### 🎯 Detection Performance Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Processing Speed")
        speed_data = pd.DataFrame({
            'Metric': ['Packets/sec', 'Detection Latency', 'Alert Generation'],
            'Current': [15432, 0.023, 0.005],
            'Average': [14500, 0.025, 0.006],
            'Unit': ['packets', 'seconds', 'seconds']
        })
        st.dataframe(speed_data, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Model Performance")
        perf_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=98.7,
            domain={'x': [0, 1], 'y': [0, 1]},
            delta={'reference': 98.0},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#00ffff"},
                'steps': [
                    {'range': [0, 70], 'color': "rgba(255, 0, 128, 0.3)"},
                    {'range': [70, 90], 'color': "rgba(255, 255, 0, 0.3)"},
                    {'range': [90, 100], 'color': "rgba(0, 255, 0, 0.3)"}
                ],
                'threshold': {
                    'line': {'color': "#ff00ff", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            },
            title={'text': "Accuracy (%)"}
        ))

        perf_gauge.update_layout(
            plot_bgcolor='rgba(10,14,26,0.8)',
            paper_bgcolor='rgba(10,14,26,0)',
            font=dict(color='#00ffff'),
            height=250
        )
        st.plotly_chart(perf_gauge, use_container_width=True)

    with col3:
        st.markdown("#### System Uptime")
        uptime_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=99.8,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#00ff00"},
                'steps': [
                    {'range': [0, 95], 'color': "rgba(255, 0, 128, 0.3)"},
                    {'range': [95, 99], 'color': "rgba(255, 255, 0, 0.3)"},
                    {'range': [99, 100], 'color': "rgba(0, 255, 0, 0.3)"}
                ],
            },
            title={'text': "Uptime (%)"}
        ))

        uptime_gauge.update_layout(
            plot_bgcolor='rgba(10,14,26,0.8)',
            paper_bgcolor='rgba(10,14,26,0)',
            font=dict(color='#00ffff'),
            height=250
        )
        st.plotly_chart(uptime_gauge, use_container_width=True)

    # Historical performance
    st.markdown("### 📊 30-Day Performance History")

    days = pd.date_range(end=datetime.now(), periods=30, freq='D')
    history_data = pd.DataFrame({
        'Date': days,
        'Packets Processed': np.random.randint(900000, 1500000, 30),
        'Threats Detected': np.random.randint(100, 200, 30),
        'False Positives': np.random.randint(5, 20, 30),
        'System Uptime': np.random.uniform(99.5, 99.9, 30)
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_data['Date'], y=history_data['Packets Processed'],
                             mode='lines+markers', name='Packets Processed',
                             line=dict(color='#4CC9F0', width=2)))
    fig.add_trace(go.Scatter(x=history_data['Date'], y=history_data['Threats Detected']*5000,
                             mode='lines+markers', name='Threats Detected (scaled)',
                             line=dict(color='#7209B7', width=2)))

    fig.update_layout(
        plot_bgcolor='rgba(10,14,26,0.8)',
        paper_bgcolor='rgba(10,14,26,0)',
        font=dict(color='#00ffff'),
        height=350,
        yaxis_title="Packets / Threats (scaled)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Confusion Matrix & ROC Curve
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Confusion Matrix")
        confusion_matrix = np.array([[9850, 120], [30, 1000]])
        fig = px.imshow(confusion_matrix,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=['Normal', 'Attack'],
                       y=['Normal', 'Attack'],
                      color_continuous_scale=['#e0f3ff', '#4CC9F0', '#7209B7']
,
                       text_auto=True)
        fig.update_layout(
            plot_bgcolor='rgba(10,14,26,0.8)',
            paper_bgcolor='rgba(10,14,26,0)',
            font=dict(color='#00ffff'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### ROC Curve")
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 0.1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                 name='ROC Curve (AUC=0.987)',
                                 line=dict(color='#4895EF', width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                 name='Random Classifier',
                                 line=dict(color='#7209B7', width=2, dash='dash')))

        fig.update_layout(
            plot_bgcolor='rgba(10,14,26,0.8)',
            paper_bgcolor='rgba(10,14,26,0)',
            font=dict(color='#00ffff'),
            height=400,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Show historical metrics table
    st.markdown("### 📋 Detailed 30-Day Metrics")
    st.dataframe(history_data, use_container_width=True, hide_index=True)

# ---------------------------
# Footer (common)
# ---------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #00ffff; padding: 20px 0; animation: neonGlow 3s ease-in-out infinite;'>
        <p style='font-size: 1rem; font-weight: 600; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 2px; font-family: "Orbitron", monospace;'>
            🛡️ SentinelNet AI-Powered Network Intrusion Detection System v1.0
        </p>
        <p style='font-size: 0.9rem; color: #ff00ff; text-transform: uppercase; letter-spacing: 1px;'>
            © 2025 SentinelNet | Powered by Machine Learning | Real-time Threat Detection
        </p>
    </div>
""", unsafe_allow_html=True)