 VitalAI — Healthcare Risk Analytics & Multi-Agent AI Decision Support System

> A comprehensive end-to-end AI/ML project that combines real machine learning models with a multi-agent AI pipeline to analyze cardiovascular risk and provide actionable medical recommendations.

---

Project Overview

VitalAI is a **multi-agent cardiovascular risk analysis system** built as part of the Microsoft Elevate Program internship capstone project. It goes beyond traditional data analytics by integrating real trained ML models with a 3-agent AI pipeline that interprets predictions, generates recommendations, and produces structured medical reports — all accessible through a professional web interface.

Problem Statement

Healthcare institutions struggle with:
- Delayed diagnosis due to manual analysis of patient records
- Lack of actionable insights from raw medical predictions
- No unified system connecting ML predictions with clinical recommendations

Solution

VitalAI solves this by creating a complete pipeline:

```
Patient Data → Real ML Models → 3 AI Agents → Medical Report + Recommendations
```

---

Tech Stack

| Component | Technology |
|---|---|
| Data Analysis | Python, Pandas, SQL |
| Machine Learning | Scikit-learn (Logistic Regression + Random Forest) |
| AI Agents | Flowise (3-agent pipeline) |
| LLM | Groq API (LLaMA 3.3 70B) |
| Backend API | Flask (Python) |
| Frontend | HTML, CSS, JavaScript |
| Dashboard | Power BI |
| Dataset | Kaggle — Cardiovascular Disease (1,000 records) |

---

Project Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Patient Input Form                    │
│         (Age, BP, Cholesterol, Heart Rate etc.)         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   Flask ML API                          │
│     Logistic Regression + Random Forest Models          │
│     Returns: lr_prob, rf_prob, avg_prob, risk_level     │
└─────────────────────┬───────────────────────────────────┘
                      │ Real ML Predictions
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Agent 1 — Risk Analyzer                    │
│   Interprets real ML output → Classifies risk           │
│   Output: HIGH/MEDIUM/LOW + Risk Score + Analysis       │
└─────────────────────┬───────────────────────────────────┘
                      │ Agent 1 Output
                      ▼
┌─────────────────────────────────────────────────────────┐
│             Agent 2 — Health Advisor                    │
│   Receives Agent 1 output → Generates recommendations  │
│   Output: Preventive actions + Lifestyle + Drug check   │
└─────────────────────┬───────────────────────────────────┘
                      │ Agent 2 Output
                      ▼
┌─────────────────────────────────────────────────────────┐
│            Agent 3 — Report Generator                   │
│   Compiles full structured VitalAI Medical Report       │
│   Output: Complete downloadable medical report          │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Web Interface + Chat                       │
│   Displays report + allows patient Q&A in plain English │
└─────────────────────────────────────────────────────────┘
```

---
 ML Models

 Dataset
- Source:Kaggle — Cardiovascular Disease Dataset
- Size:1,000 patient records
- Features:12 cardiovascular indicators
- Target:Heart disease present (1) or absent (0)
- Split: 800 training / 200 testing

 Features Used
| Feature | Description |

| age | Patient age |
| gender | 0 = Female, 1 = Male |
| chest_pain_type | 0-3 (typical angina to asymptomatic) |
| resting_bp | Resting blood pressure (mmHg) |
| serum_chol | Serum cholesterol (mg/dL) |
| fasting_blood_sugar | 1 = above 120 mg/dL, 0 = normal |
| resting_ecg | 0 = normal, 1 = ST abnormality, 2 = LVH |
| max_heart_rate | Maximum heart rate achieved |
| exercise_angina | 1 = yes, 0 = no |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| major_vessels | Number of major vessels (0-3) |

Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |

| Logistic Regression | 96.5% | 96.6% | 97.4% | 97.0% |
| Random Forest | 99.0% 

Both models are trained, saved as `.pkl` files and served via Flask API for real-time predictions.



AI Agent Pipeline

 Agent 1 — Risk Analyzer
- Role: Receives real ML model predictions + patient data
- Input: Patient details + Logistic Regression probability + Random Forest probability
- Output:Risk classification (HIGH/MEDIUM/LOW) + Risk score + Detailed analysis of concerning values

 Agent 2 — Health Advisor
- Role:Receives Agent 1's risk analysis
- Input:Agent 1 output (cannot run without Agent 1)
- Output: Preventive actions + Lifestyle recommendations + Drug interaction warnings + When to see a doctor

 Agent 3 — Report Generator
- Role: Receives Agent 2's recommendations
- Input:Full Agent 2 output (cannot run without Agent 2)
- Output: Structured VitalAI Medical Report with all sections compiled

 What Makes This Unique
- Dependent agents — Agent 2 cannot run without Agent 1's output. Agent 3 cannot run without Agent 2's output. This is a genuine multi-agent dependency chain.
- Real ML integration — Agents receive actual trained model probability scores, not guesses
- Drug interaction detection — Agent 2 checks for cardiovascular drug interactions if medications are mentioned
- Patient Q&A — After report generation, patients can ask follow-up questions in plain language

---

 Project Structure

```
VitalAI/
├── Healthcare_Risk_Analytics_ML.ipynb  # Complete ML notebook
│                                        # EDA, preprocessing, model training,
│                                        # evaluation, feature importance
├── dataset.csv.csv                      # Kaggle cardiovascular dataset
├── app.py                               # Flask API (serves ML models)
├── requirements.txt                     # Python dependencies
├── index.html                           # VitalAI web frontend
├── VitalAI Chatflow.json               # Flowise 3-agent configuration
├── healthcare_dashboard.pbix            # Power BI dashboard
├── dashboard_preview.png               # Dashboard screenshot
└── README.md                           # This file
```

---

 How to Run

 Prerequisites
- Python 3.8+
- Flowise Cloud account
- Groq API key (free at console.groq.com)

 Step 1 — Train and save ML models
Open `Healthcare_Risk_Analytics_ML.ipynb` in Google Colab and run all cells. This trains both models and saves `logistic_model.pkl` and `random_forest_model.pkl`.

 Step 2 — Run Flask API
```bash
pip install flask flask-cors joblib numpy scikit-learn
python app.py
```
API runs at `http://localhost:5000`

 Step 3 — Set up Flowise agents
1. Import `VitalAI Chatflow.json` into Flowise Cloud
2. Add your Groq API key to both Groq nodes
3. Deploy the chatflow

 Step 4 — Open frontend
Open `index.html` in browser. Update the API URL if needed.

---

Features

- **Real ML predictions** — Logistic Regression + Random Forest both run on every submission
- **3 dependent AI agents** — each agent builds on the previous one's output
- **Natural language input** — doctors enter patient data in plain language
- **Structured medical report** — professionally formatted, downloadable as text
- **Drug interaction warning** — detects cardiovascular medication conflicts
- **Patient Q&A chat** — patients ask follow-up questions in plain language
- **Technical view** — evaluators can see real ML probability scores
- **Power BI dashboard** — population-level cardiovascular risk analytics

---

 Data Analytics Phase

The `Healthcare_Risk_Analytics_ML.ipynb` notebook covers:

1. Exploratory Data Analysis (EDA)
   - Dataset shape, data types, null value check
   - Target distribution (58% heart disease, 42% no heart disease)
   - Statistical summary of all features

2. Data Preprocessing
   - Dropped `patient_id` column (not a feature)
   - 80/20 train-test split with random state 42

3. **Model Training**
   - Logistic Regression (max_iter=1000)
   - Random Forest Classifier (n_estimators=100)

4. Model Evaluation
   - Accuracy, Precision, Recall, F1 Score
   - Confusion Matrix visualization
   - Feature Importance chart

5. Predictions
   - Added `Predicted_Risk` column to dataset
   - Exported `Healthcare_Risk_Predictions_Final.csv`

6. Power BI Dashboard
   - Risk distribution by age and gender
   - Key health indicator analysis
   - Interactive filters

---

 Disclaimer

VitalAI is built for **educational purposes only** as part of an internship capstone project. It is not a substitute for professional medical advice. Always consult a qualified physician for medical decisions.

---

 Built With

- Microsoft Elevate Program — Fice Education Internship
- Flowise — Multi-agent AI pipeline
- Groq — Free LLM API (LLaMA 3.3 70B)
- Scikit-learn — ML model training
- Flask — REST API
- Power BI — Business intelligence dashboard

---

*VitalAI — Turning cardiovascular data into actionable medical intelligence*
