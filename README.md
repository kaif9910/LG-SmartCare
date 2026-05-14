# ⚙️ LG SmartCare — Predictive Maintenance AI

> An end-to-end Machine Learning project that predicts appliance failure risk before it happens — inspired by LG's real-world SmartCare platform.

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat-square)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red?style=flat-square&logo=streamlit)
![PowerBI](https://img.shields.io/badge/Dashboard-PowerBI-yellow?style=flat-square&logo=powerbi)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

---

## 🚀 Live Demo

🔗 **[Open Live App → lg-smartcare-kaif9910.streamlit.app](https://lg-smartcare-kaif9910.streamlit.app)**

---

## 💡 Problem Statement

Industrial machines and home appliances fail **unexpectedly** — causing expensive repairs and downtime.

This project uses **real sensor data** to predict failure **before it happens**, enabling timely preventive maintenance.

| | Reactive Maintenance | Preventive Maintenance |
|---|---|---|
| 💸 Cost | ₹5,000 per incident | ₹1,500 per service |
| ⏱️ Downtime | High | Minimal |
| 📉 Risk | Unplanned failure | Controlled |

> **Savings per machine: ₹3,500** — multiply that across thousands of devices! 💰

---

## 🎯 What This Project Does

```
Raw Sensor Data  →  ML Model  →  Failure Risk Score  →  Actionable Alert
```

- 🟢 **Low Risk** → Machine is healthy, continue monitoring
- 🟠 **Medium Risk** → Schedule maintenance in 2–4 weeks  
- 🔴 **High Risk** → Immediate technician visit required

---

## 📁 Project Structure

```
LG-SmartCare/
│
├── 📓 LG_SMARTCARE.ipynb      — ML pipeline, EDA, training, KPIs
├── 🌐 app.py                  — Streamlit web application
├── 📊 LG_SMARTCARE.pbix       — Power BI interactive dashboard
├── 📄 ai4i.csv                — Machine sensor dataset (10,000 records)
├── 📄 ape.csv                 — Appliance energy dataset
├── 📄 lg_smartcare_data.csv   — Final merged dataset
├── 📄 requirements.txt        — Python dependencies
└── 📄 README.md
```

---

## 🔄 Project Phases

### 🔧 Phase 1 — ML Pipeline (Predictive Maintenance)
- Cleaned & renamed columns to match **LG SmartCare schema**
- Feature engineering — `temp_diff`, `device_age_months`
- Dropped sub-failure columns (TWF, HDF, PWF, OSF, RNF) to **prevent data leakage**
- Fixed class imbalance using **SMOTE** — applied only on training data
- Trained **XGBoost Classifier** with custom threshold (0.35) for better recall
- Evaluated using ROC-AUC Score & Classification Report

### ⚡ Phase 2 — Energy Analysis
- Extracted time-based features: hour, day of week, weekend flag
- Aggregated daily energy consumption (kWh) and peak usage hours
- Simulated regional segmentation (North / South / East / West)

### 📈 Phase 3 — Business KPIs & Dashboard
- Calculated failure rate, high-risk device count, average risk score
- Estimated cost savings from preventive vs reactive maintenance
- Built **interactive Power BI dashboard** for business stakeholders

---

## 🤖 Model Details

| Detail | Value |
|---|---|
| Algorithm | XGBoost Classifier |
| Dataset | AI4I 2020 — 10,000 records |
| Imbalance Fix | SMOTE (on training data only) |
| Decision Threshold | 0.35 (optimized for recall) |
| Evaluation | ROC-AUC Score + Classification Report |
| Features | 7 sensor + derived features |

---

## 📊 Power BI Dashboard

Interactive dashboard with regional filtering, KPI cards, and energy trends.

**To explore:**
1. Download `LG_SMARTCARE.pbix` from this repo
2. Open in **Power BI Desktop** — [Free Download](https://powerbi.microsoft.com/desktop)
3. Click any region → all charts update instantly ✅

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/kaif9910/LG-SmartCare.git
cd LG-SmartCare

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch app
streamlit run app.py
```

> ✅ No `.pkl` files needed — model trains automatically from `ai4i.csv`

---

## 🧠 Key Learnings

- Handling **class imbalance** with SMOTE in real-world datasets
- Preventing **data leakage** in ML pipelines
- **Custom threshold tuning** for business-specific tradeoffs
- Building a complete ML product — **data → model → deployment**
- Creating **Power BI dashboards** for non-technical stakeholders
- Translating ML predictions into **real business value**

---

## 📌 Dataset Sources

- [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset) — UCI ML Repository
- [Appliance Energy Prediction Dataset](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction) — UCI ML Repository

---

## 👤 Author

**Mohamad Kaif** — 3rd Year B.Tech Student

🐙 [GitHub — kaif9910](https://github.com/kaif9910)

---

## 📝 Note

Regional energy segmentation (North/South/East/West) is **simulated for demonstration** to showcase business analytics capability. In production, this data would come from real device registration records.
