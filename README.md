# 🛡️ SentinelNet — AI-Powered Network Intrusion Detection System  
### Real-Time Threat Monitoring • Machine Learning Models • Interactive Dashboard • Cyberpunk UI

<p align="center">
  <img src="https://img.shields.io/badge/SentinelNet-AI%20NIDS-0A0F1F?style=for-the-badge&logo=shieldcheck&logoColor=00E5FF&labelColor=1A1037" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Enabled-1A1037?style=for-the-badge&logo=python&logoColor=14FFEC&labelColor=0A0F1F" />
  <img src="https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-7F5AF0?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Cybersecurity-Threat%20Detection-7209B7?style=for-the-badge&logo=shield&logoColor=white" />
</p>


---

## 🌐 Live Demo

<p align="center">

<a href="https://sentinelnet-ai.streamlit.app/">
<img src="https://img.shields.io/badge/🛡️_Try_SentinelNet_Live-00FFFF?style=for-the-badge&logo=streamlit&logoColor=black"/>
</a>

</p>

---

## 🧪 Quick Testing Guide

Want to test SentinelNet instantly?

### Step 1

Download the sample dataset:

<a href="https://github.com/moinakgh99/SentinelNet/raw/main/sample_test_50.csv">
<img src="https://img.shields.io/badge/📥_Download_sample50.csv-14FFEC?style=for-the-badge&logo=databricks&logoColor=black"/>
</a>

### Step 2

Launch the application.

### Step 3

Open **Live Analysis**.

### Step 4

Upload the downloaded CSV file.

### Step 5

Explore:

* 🚨 Threat Detection
* 📊 Network Analytics
* 🤖 ML Predictions
* ⚠️ Security Alerts

---



## 🚀 Overview

**SentinelNet** is an advanced **Network Intrusion Detection System (NIDS)** designed to detect malicious network behavior using **Machine Learning** and **Anomaly Detection** techniques.

It analyzes real-time network traffic, predicts threat categories, visualizes metrics, and displays alerts through a futuristic **cyberpunk-themed UI** built in Streamlit.

This system demonstrates a complete end-to-end ML pipeline:

- 📥 Data ingestion  
- 🧹 Cleaning & preprocessing  
- 🤖 Model training (RF, SVM, LR)  
- 🧪 Anomaly detection  
- 📊 Real-time dashboard visualization  
- 🚨 Threat alerting  

---


# 🖼️ **Screenshots**

### 🔹 **Dashboard Overview**
![Dashboard](Screenshots/Dashboard%201.png)
![Dashboard](Screenshots/Dashboard%201.1.png)

### 🔹 **Detection Model - Live Analysis**
![Live Analysis](Screenshots/Live%20Analysis%201.png)
![Live Analysis](Screenshots/Live%20Analysis%201.1.png)
![Live Analysis](Screenshots/Live%20Analysis%201.2.png)
![Live Analysis](Screenshots/Live%20Analysis%201.3.png)

### 🔹 **Overview Page**
![Overview](Screenshots/Overview%20Section.png)

### 🔹 **EDA & Cleaning Page**
![EDA](Screenshots/EDA%20%26%20Cleaning.png)

### 🔹 **Supervised Models**
![Models](Screenshots/Supervised%20Models.png)

### 🔹 **Anomaly Detection (PCA, K-Means, Isolation Forest)**
![Anomaly Detection](Screenshots/Anomaly%20Detection.png)

### 🔹 **Live Alerts Page**
![Alerts](Screenshots/Alerts.png)

---

# ✨ **Features of SentinelNet**

## 🔍 **1. Real-Time Traffic Analysis**
- Accepts uploaded CSV or live data stream  
- Cleans & preprocesses data automatically  
- Aligns features to match ML model structure  

## 🚨 **2. Attack Classification**
Predicts:
- **Normal**
- **Suspicious**

Uses Random Forest as the primary detection model.

## 📊 **3. Interactive Visual Analytics**
Includes:
- Pie charts  
- Dynamic line charts  
- Confusion matrix  
- Feature importance  
- PCA scatter plots  
- Correlation heatmaps  

## 🤖 **4. Machine Learning Models Used**
| Model | Purpose |
|-------|---------|
| Random Forest | Best accuracy (≈96%) |
| SVM | Secondary classifier |
| Logistic Regression | Comparison model |
| PCA | Dimensionality reduction |
| K-Means | Unsupervised anomaly detection |
| Isolation Forest | Outlier detection |

## 🕹️ **5. Futuristic Streamlit UI**
- Cyberpunk glow theme  
- Hologram slideshow  
- Animated stats cards  
- Sidebar navigation  
- Live monitoring controls  

---

# ⚙️ **Tech Stack**

### ✔ Machine Learning  
- Python, Pandas, NumPy  
- Scikit-Learn  
- PCA, Isolation Forest, K-Means  

### ✔ Dashboard / UI  
- Streamlit  
- Plotly  
- Custom CSS (glow effects + animations)

### ✔ Dataset  
- **CICIDS 2017 – Wednesday Working Hours Dataset**  
Why Wednesday?  
- Balanced normal + DoS traffic  
- Less corrupted data  
- Ideal for training + evaluation  

---

# 🧪 **How the System Works**

## **1️⃣ Data Processing**
- Reads CSV  
- Handles missing values  
- Removes skewed features  
- Applies PCA (optional)  

## **2️⃣ Model Prediction**
- Random Forest predicts traffic category  
- Predictions appended to dataframe  
- Alerts generated instantly  

## **3️⃣ Visualization**
- Updated charts displayed in real time  
- Alerts logged  
- System performance monitored  

---

# ⭐ Final Note

This repository contains:

- Full Streamlit dashboard  
- Machine learning models  
- Real-time analytics  
- A complete demonstration video  
- Clean and professional documentation  
