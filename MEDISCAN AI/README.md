# 🩻 Medi Scan AI — AI-Powered Medical Image Analysis & Health Platform

**Medi Scan AI** is a modern, responsive, professional healthcare technology platform combining Groq-powered AI vision analysis, evidence-aware clinical reasoning, an interactive conversational health assistant, persistent scan archives, clinical lab document OCR screening, and real trained machine learning risk estimators.

> 🛡️ **Medical & Clinical Safety Advisory:** Medi Scan AI is designed exclusively for **educational, clinical reference, and informational screening purposes**. It does **NOT** provide certified medical diagnoses, replace clinical evaluation by a physician or radiologist, or prescribe medication. In medical emergencies, immediately contact local emergency services (e.g. 911 / 112).

---

## 🌟 Key Capabilities

1. **AI Medical Image Scan & Vision Reasoning (Groq Vision)**
   - Analyzes radiographs (X-rays), CT scans, MRIs, ultrasounds, and dermatoscopy images.
   - Powered by official Groq Vision models (`llama-3.2-11b-vision-preview` or `llama-3.2-90b-vision-preview`).
   - Generates structured, evidence-aware outputs distinguishing between **visible observations** and **possible differential findings**.
   - Identifies technical imaging limitations, artifacts, and missing projections.
   - Provides a plain-language layperson medical terminology glossary and tailored questions to ask a doctor.

2. **Persistent Scan History Archives**
   - Automatically saves analyses locally in `data/scan_history.json`.
   - Search by keyword or anatomy, filter by imaging modality.
   - Review past scans, export JSON reports, or reload scans directly into discussion.
   - Privacy-first controls: delete individual scans or clear complete history at any time.

3. **Interactive Conversational AI Assistant**
   - Context-aware health assistant powered by Groq (`llama-3.3-70b-versatile`).
   - Answers follow-up questions grounded in the actively inspected scan.
   - Includes one-click suggested questions for patient preparation.
   - Implements safety rails to prevent hallucinations, emergency misdirection, or unauthorized medication prescriptions.

4. **Clinical Lab Report OCR & Biomarker Screening**
   - Upload PDF or image lab reports (CBC, metabolic panels).
   - Extracts key biomarkers: Hemoglobin, Blood Glucose, Total Cholesterol, and TSH.
   - Compares findings against reference intervals with biological sex adaptability.
   - Generates statistical baselines and algorithmic triage flags.
   - Pre-fills detected values directly into the ML risk predictors.

5. **Trained Clinical ML Risk Predictors**
   - **Diabetes Risk:** XGBoost Classifier (8 features from benchmark Pima cohort).
   - **Anemia Risk:** LightGBM Classifier (5 CBC features).
   - **Thyroid Condition:** XGBoost 9-class Classifier (21 clinical features with LabelEncoder).
   - All models loaded and verified from `models/` with calibrated preprocessing scalers.

---

## 📂 Project Architecture

```text
ClarityHealth-Lab-Analyzer/
├── app.py                  # Main Streamlit application and navigation controller
├── groq_service.py         # Groq Vision & Chat completions, validation, and safety prompts
├── history_service.py      # Persistent JSON scan history manager (CRUD, search, filter)
├── ml_service.py           # Trained ML models loader and inference (Diabetes, Anemia, Thyroid)
├── ocr_service.py          # Document OCR extraction (pytesseract, pypdf) and reference triage
├── ui_components.py        # Modern healthcare design system, cards, meters, banners, CSS
├── models/                 # Pretrained scikit-learn, XGBoost, and LightGBM model artifacts
│   ├── xgb_diabetes.pkl
│   ├── scaler_diabetes.pkl
│   ├── lgb_anemia.pkl
│   ├── scaler_anemia.pkl
│   ├── xgb_thyroid.pkl
│   ├── scaler_thyroid.pkl
│   └── thyroid_target_encoder.pkl
├── data/                   # Persistent local scan history (auto-created)
│   └── scan_history.json
├── requirements.txt        # Python dependency manifest
├── .env.example            # Template for environment variables
└── README.md               # Complete platform documentation
```

---

## ⚙️ Setup & Installation

### 1. Clone & Enter Project Directory
```bash
git clone <repository-url>
cd ClarityHealth-Lab-Analyzer
```

### 2. Python Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Groq API Key
Obtain a free Groq API Key from [console.groq.com/keys](https://console.groq.com/keys).

Create a `.env` file (copying `.env.example`):
```bash
cp .env.example .env
```
Add your key:
```env
GROQ_API_KEY=gsk_your_actual_key_here
GROQ_VISION_MODEL=llama-3.2-11b-vision-preview
GROQ_CHAT_MODEL=llama-3.3-70b-versatile
```
*(Alternatively, enter and save your key directly in the application's **Settings** tab at runtime).*

---

## 🚀 Running the Application

Launch Medi Scan AI with Streamlit:

```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 🧪 Verification & Testing

Run the automated service and model self-tests:

```bash
# Test ML models loading and inference
python -c "import ml_service; print(ml_service.test_models())"

# Test Groq configuration and vision service status
python -c "import groq_service; print(groq_service.test_service())"

# Test persistent history storage
python -c "import history_service; print(history_service.test_history())"
```

---

## 🔒 Privacy & Ethical AI Safeguards

- **Server-Side API Key Management:** Keys are held server-side or in local `.env` and are never embedded in client bundles.
- **Local Data Retention:** Scan metadata and thumbnails are stored exclusively on your local filesystem under `data/scan_history.json`.
- **Permanent Deletion:** You can delete individual scans or erase the entire database with a single click.
- **Prompt Injection Defense:** Strict system prompt framing isolates user-provided clinical notes and disallows unauthorized instruction override.
