# 🛡️ Cyber Attack / Intrusion Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.6.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Dataset-UNSW--NB15-00b4d8?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Model-LinearSVC-6a0dad?style=for-the-badge" />
</p>

<p align="center">
  A machine learning–based Network Intrusion Detection System (NIDS) built on the <strong>UNSW-NB15</strong> dataset.<br>
  Uses a full <strong>scikit-learn Pipeline</strong> (RobustScaler → SelectKBest → PCA → LinearSVC) and is deployed as an interactive <strong>Streamlit</strong> web application.
</p>

<p align="center">
  <b>Alexandria National University &nbsp;·&nbsp; Data Computation — Spring '26</b>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Model Pipeline](#-model-pipeline)
- [Results](#-results)
- [Streamlit App](#-streamlit-app)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Technologies Used](#-technologies-used)
- [Team](#-team)

---

## 🔍 Overview

This project implements a full end-to-end binary intrusion detection system that classifies network connections as **Normal (0)** or **Attack (1)**. It fulfills all requirements of the Data Computation Spring '26 final project:

- ✅ Dataset with 82,332 records and 45 features
- ✅ Train/Test split **before** any preprocessing (no data leakage)
- ✅ Thorough EDA with unique, non-repetitive visualisations
- ✅ Data cleaning (outlier winsorisation, no records dropped)
- ✅ Dimensionality reduction via SelectKBest + PCA
- ✅ Tuned SVM model with GridSearchCV (Stratified 5-Fold)
- ✅ Complete sklearn Pipeline from raw data to prediction
- ✅ **Bonus:** Streamlit deployment with two interactive modes

---

## 📊 Dataset

**UNSW-NB15** — Created by the Cyber Range Lab at the Australian Centre for Cyber Security (ACCS, UNSW).

| Property | Details |
|---|---|
| **Total Records** | 82,332 |
| **Features** | 45 (42 used after dropping `id` and `attack_cat`) |
| **Task** | Binary Classification: Normal (0) vs Attack (1) |
| **Missing Values** | None found |
| **Class Balance** | ~45% Normal / ~55% Attack |

### Attack Category Distribution (Training Set)

| Attack Type | Count | Description |
|---|---|---|
| **Normal** | 37,000 | Benign network traffic |
| **Generic** | 18,871 | Generic cipher-breaking attacks |
| **Exploits** | 11,132 | Known vulnerability exploitation |
| **Fuzzers** | 6,062 | Random data injection |
| **DoS** | 4,089 | Denial of Service |
| **Reconnaissance** | 3,496 | Scanning and probing |
| **Analysis** | 677 | Port scans, HTML file analysis |
| **Backdoor** | 583 | Unauthorized persistent access |
| **Shellcode** | 378 | Injected shell-level code |
| **Worms** | 44 | Self-replicating malware |

### Feature Categories

| Category | Features |
|---|---|
| **Flow** | `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate` |
| **Protocol** | `proto`, `service`, `state` |
| **Content** | `sload`, `dload`, `sinpkt`, `dinpkt`, `sjit`, `djit` |
| **TCP Timing** | `sttl`, `dttl`, `tcprtt`, `synack`, `ackdat`, `swin`, `dwin` |
| **Connection Count** | `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_ltm`, ... |
| **Application** | `trans_depth`, `response_body_len`, `ct_ftp_cmd`, ... |

> **Source:** [UNSW-NB15 Official Page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

---

## 🔬 Project Workflow

```
Raw Dataset (82,332 × 45)
         │
         ▼
 ⚠️  Step 1: Train/Test Split FIRST (80/20, stratified, random_state=42)
         │                    │
         ▼                    ▼
  Train: 65,865          Test: 16,467
         │
         ▼
 Step 2: EDA (on training set only — 6 unique visualisations)
         │
         ▼
 Step 3: Data Cleaning
   ├─ No missing values found
   └─ Outlier winsorisation: IQR clipping (1st–99th percentile)
      [No records dropped — fully documented]
         │
         ▼
 Step 4: Preprocessing (inside Pipeline)
   ├─ RobustScaler    → numerical features
   └─ OrdinalEncoder  → proto, service, state
         │
         ▼
 Step 5: Dimensionality Reduction
   ├─ SelectKBest (f_classif, k=30)  →  42 → 30 features
   └─ PCA (95% variance retained)   →  30 → 21 components
         │
         ▼
 Step 6: SVM Model
   └─ LinearSVC (C=1.0, loss=squared_hinge)
      Tuned via GridSearchCV — Stratified 5-Fold CV
         │
         ▼
 Step 7: Evaluation → Accuracy / F1 / ROC-AUC / Confusion Matrix
         │
         ▼
 Step 8: Save Pipeline → model_artifacts/
         │
         ▼
 Step 9: Streamlit Deployment (Bonus ✅)
```

---

## ⚙️ Model Pipeline

The entire process is encapsulated in a **single scikit-learn Pipeline** to prevent data leakage and ensure reproducibility:

```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', RobustScaler(),        numerical_columns),
        ('cat', OrdinalEncoder(...),   ['proto', 'service', 'state'])
    ])),
    ('feature_selection', SelectKBest(score_func=f_classif, k=30)),
    ('pca',        PCA(n_components=21, random_state=42)),
    ('classifier', LinearSVC(C=1.0, loss='squared_hinge', max_iter=3000))
])
```

### Hyperparameter Tuning

Tuned via `GridSearchCV` with Stratified 5-Fold cross-validation:

| Parameter | Values Searched | Best Value |
|---|---|---|
| `C` | [0.01, 0.1, 1.0, 10] | **1.0** |
| `class_weight` | [None, 'balanced'] | **None** |
| `loss` | squared_hinge | squared_hinge |

### Dimensionality Reduction Summary

| Step | Input Dims | Output Dims | Method |
|---|---|---|---|
| SelectKBest | 42 | 30 | ANOVA F-score |
| PCA | 30 | **21** | 95% variance retained |
| **Total reduction** | **42** | **21** | **2× compression** |

---

## 📈 Results

### Final Pipeline Performance on Test Set (16,467 samples)

| Metric | Score |
|---|---|
| **Accuracy** | **84.24%** |
| **Precision** | **84.92%** |
| **Recall** | **86.78%** |
| **F1-Score** | **85.84%** |

### Confusion Matrix Summary

| | Predicted Normal | Predicted Attack |
|---|---|---|
| **Actual Normal** | True Negatives ✅ | False Positives ⚠️ |
| **Actual Attack** | False Negatives ⚠️ | True Positives ✅ |

### EDA Insights

- **sttl**, **dttl**, **rate**, and **sbytes** show the strongest separation between Normal and Attack traffic
- Strong correlations exist between `sbytes`/`smean` and `dbytes`/`dmean`
- Extreme outliers in `rate`, `sload`, and `sbytes` are typical of attack traffic — retained and winsorised
- UDP and TCP dominate protocol distribution; unknown service (`-`) is the most frequent service value

---

## 🖥️ Streamlit App

The deployed app (`app.py`) provides a dark-themed cybersecurity dashboard with **two prediction modes**:

### Mode 1 — Manual Input
- Enter all 42 network traffic features manually via grouped form inputs
- Organised into: Protocol & Connection, Traffic Volume, Load & Jitter, TTL & TCP Timing, Connection Count (ct_*), Application & Flags
- Outputs: prediction label, decision function score, sigmoid-based confidence bar

### Mode 2 — Sample from Test Data
- Fetch a random record from the held-out 20% test split (`random_state=42`)
- Three sample options: **Random**, **Normal only**, **Attack only**
- Reveals true label after prediction to verify correctness in real time

### Dashboard Metrics (Header)

The app displays live from `pipeline_metadata.pkl`:

| Metric | Value |
|---|---|
| Model | LinearSVC |
| Accuracy | 84.24% |
| F1-Score | 85.84% |
| PCA Dims | 21 |

---

## 📁 Project Structure

```
Cyber_Attack_Detection/
│
├── app.py                            # Streamlit web application
├── cyber_intrusion_detection.ipynb   # Full project notebook (EDA → Pipeline → Evaluation)
├── UNSW_NB15_training-set.csv        # Dataset (used for both training and test sampling)
├── requirements.txt                  # Python dependencies
│
└── model_artifacts/
    ├── svm_intrusion_pipeline.pkl    # Trained full sklearn pipeline
    └── pipeline_metadata.pkl         # Metrics, feature columns, PCA dims, best params
```

---

## ⚙️ Installation & Usage

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Mahmoud-Mansour5/Cyber_Attack_Detection.git
cd Cyber_Attack_Detection

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
streamlit
pandas
numpy
joblib
scikit-learn==1.6.0
```

### Run the App

```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

> **Note:** The dataset file `UNSW_NB15_training-set.csv` and the `model_artifacts/` folder must be present in the same directory as `app.py`.

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.8+ | Core language |
| **scikit-learn** | 1.6.0 | Pipeline, SVM, PCA, preprocessing, GridSearchCV |
| **Streamlit** | latest | Interactive web application |
| **Pandas** | latest | Data manipulation |
| **NumPy** | latest | Numerical operations |
| **Matplotlib / Seaborn** | latest | EDA visualisations |
| **Joblib** | latest | Model serialisation |

---

## 👥 Team

**Course:** Data Computation — Spring '26  
**Institution:** Alexandria National University — Faculty of Computers and Data Science  

| Name | Role |
|---|---|
| Mahmoud Mansour | Team Member |
| *(Team member 2)* | *(Role)* |
| *(Team member 3)* | *(Role)* |
| *(Team member 4)* | *(Role)* |

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

- **UNSW-NB15 Dataset:**  
  Moustafa, N., & Slay, J. (2015). *UNSW-NB15: a comprehensive data set for network intrusion detection systems.* MilCIS, IEEE.  
  Moustafa, N., & Slay, J. (2016). *The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set.* Information Security Journal, A Global Perspective.

- Australian Centre for Cyber Security (ACCS) for providing the dataset.
