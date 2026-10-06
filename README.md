# 🎓 EduAid: Early-Warning Student Dropout Predictor
> **A Community-First Machine Learning System to Identify Vulnerable Students & Deliver Early Interventions.**

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-87.6%25-brightgreen.svg)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.936-blueviolet.svg)

---

## 🌟 Overview
**EduAid** is an end-to-end Machine Learning web application designed to empower community schools, NGOs, and non-profit learning centers. By analyzing behavioral and logistical indicators (attendance trends, commute obstacles, digital access, study habits), EduAid flags at-risk students **months before final exams**, providing educators with an actionable, personalized intervention checklist.

---

## 🚀 Key Features

* **🔍 Single Student Triage**: Real-time slider-based assessment providing instantaneous risk classification (`On-Track` vs. `At-Risk`) and exact percentage confidence.
* **📋 Tailored Intervention Engine**: Automatically generates personalized recommendations (e.g., subsidized bus passes for long commutes, offline study packs for students lacking internet, peer tutoring).
* **📁 Batch Class Scanner**: Upload a full class/grade CSV spreadsheet to scan, rank, and export high-priority students for counselor follow-ups.
* **📈 Explainable AI (XAI)**: Visualizes feature importance rankings and confusion matrix diagnostics to build educator trust.
* **⚡ 100% Offline & Private**: Zero data leaves the local machine, ensuring full privacy and compliance with student data protections.

---

## 📊 Model Performance Comparison

| Model Architecture | Accuracy | Safety Recall | Precision | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **87.6%** | **75.9%** | **80.3%** | **78.0%** | **0.936** |
| **Random Forest** | **84.2%** | 58.6% | **81.7%** | 68.3% | **0.900** |
| **Decision Tree** | 76.0% | 54.5% | 59.4% | 56.8% | 0.794 |

---

## 🛠️ Project Structure

```text
eduaid-student-dropout-predictor/
├── app.py                      # Interactive Streamlit Web Dashboard
├── run_app.bat                 # 1-Click Windows Batch Launcher
├── requirements.txt            # Python dependencies
├── STUDY_NOTES.md              # In-depth ML concepts & interview prep
├── README.md                   # Project documentation
├── data/
│   ├── generate_data.py        # Synthetic dataset generator
│   ├── students_dataset.csv    # 2,500 student training records
│   └── sample_batch_students.csv # 30-student demo roster
├── src/
│   └── train_model.py          # Preprocessing, training & evaluation
└── models/
    ├── best_model.pkl          # Trained Random Forest model
    ├── decision_tree_model.pkl # Interpretable Decision Tree model
    ├── scaler.pkl              # StandardScaler pipeline
    └── model_metrics.json      # Benchmark diagnostics & feature weights
```

---

## ⚙️ Quick Start

### 1. Prerequisites
Ensure Python is installed on your computer.

### 2. Launch with 1-Click (Windows)
Double-click `run_app.bat` or run:
```bash
run_app.bat
```

### 3. Or Launch via Terminal
```bash
# Optional: Retrain models
python src/train_model.py

# Launch Web Application
python -m streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 📜 Ethical AI Principles
* **Supportive, Not Punitive**: Predictions are intended solely to allocate supportive educational resources (scholarships, bus cards, mentoring).
* **Bias Mitigation**: Uses logistical and study indicators rather than demographic attributes to ensure fair, unbiased triage.
