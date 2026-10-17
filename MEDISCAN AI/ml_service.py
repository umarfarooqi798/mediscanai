"""
Mediscan AI — Clinical ML Models Service & Explainability
Loads and executes the 3 trained machine learning classifiers:
1. Diabetes Risk (XGBoost Classifier — 8 features)
2. Anemia Risk (LightGBM Classifier — 5 features)
3. Thyroid Condition (XGBoost 9-class Classifier — 21 features + LabelEncoder)
Computes feature contribution breakdowns for explainability.
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

MODEL_FILES = {
    "model_diabetes": os.path.join(MODEL_DIR, "xgb_diabetes.pkl"),
    "scaler_diabetes": os.path.join(MODEL_DIR, "scaler_diabetes.pkl"),
    "model_anemia": os.path.join(MODEL_DIR, "lgb_anemia.pkl"),
    "scaler_anemia": os.path.join(MODEL_DIR, "scaler_anemia.pkl"),
    "model_thyroid": os.path.join(MODEL_DIR, "xgb_thyroid.pkl"),
    "scaler_thyroid": os.path.join(MODEL_DIR, "scaler_thyroid.pkl"),
    "thyroid_target_encoder": os.path.join(MODEL_DIR, "thyroid_target_encoder.pkl"),
}

_MODELS_CACHE = None
_ERRORS_CACHE = {}


def load_ml_models() -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Loads all model and scaler pickles with caching."""
    global _MODELS_CACHE, _ERRORS_CACHE
    if _MODELS_CACHE is not None:
        return _MODELS_CACHE, _ERRORS_CACHE

    loaded = {}
    errors = {}

    for key, path in MODEL_FILES.items():
        if not os.path.exists(path):
            errors[key] = f"File missing: {os.path.basename(path)}"
            continue
        try:
            loaded[key] = joblib.load(path)
        except Exception as exc:
            errors[key] = f"Error loading {os.path.basename(path)}: {str(exc)}"

    _MODELS_CACHE = loaded
    _ERRORS_CACHE = errors
    return loaded, errors


def get_model_readiness() -> Dict[str, bool]:
    models, _ = load_ml_models()
    return {
        "diabetes": "model_diabetes" in models and "scaler_diabetes" in models,
        "anemia": "model_anemia" in models and "scaler_anemia" in models,
        "thyroid": "model_thyroid" in models and "scaler_thyroid" in models and "thyroid_target_encoder" in models,
    }


def predict_diabetes(
    pregnancies: int,
    glucose: float,
    blood_pressure: float,
    skin_thickness: float,
    insulin: float,
    bmi: float,
    pedigree: float,
    age: int
) -> Dict[str, Any]:
    models, _ = load_ml_models()
    if not ("model_diabetes" in models and "scaler_diabetes" in models):
        return {"success": False, "error": "Diabetes model or scaler is not loaded."}

    try:
        raw_vals = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]
        feature_names = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Pedigree Function", "Age"]
        raw_input = np.array([raw_vals], dtype=float)
        scaled = models["scaler_diabetes"].transform(raw_input)
        pred = int(models["model_diabetes"].predict(scaled)[0])
        prob = None
        if hasattr(models["model_diabetes"], "predict_proba"):
            prob = float(np.max(models["model_diabetes"].predict_proba(scaled)[0]))

        # Calculate feature contributions for explainability
        imp = models["model_diabetes"].feature_importances_
        scaled_dev = scaled[0]
        contrib = scaled_dev * imp
        norm_factor = np.sum(np.abs(contrib)) or 1.0
        norm_contrib = (contrib / norm_factor) * 100.0

        breakdown = [
            {"feature": name, "value": val, "contribution": round(float(c), 1)}
            for name, val, c in zip(feature_names, raw_vals, norm_contrib)
        ]
        breakdown.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return {
            "success": True,
            "prediction": pred,
            "is_elevated_risk": (pred == 1),
            "label": "Elevated Risk Profile" if pred == 1 else "Standard Risk Profile",
            "confidence": prob,
            "feature_breakdown": breakdown
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def predict_anemia(
    gender: str,  # "Male" or "Female"
    hemoglobin: float,
    mch: float,
    mchc: float,
    mcv: float
) -> Dict[str, Any]:
    models, _ = load_ml_models()
    if not ("model_anemia" in models and "scaler_anemia" in models):
        return {"success": False, "error": "Anemia model or scaler is not loaded."}

    try:
        gender_val = 1 if gender.strip().lower() == "male" else 0
        raw_vals = [gender_val, hemoglobin, mch, mchc, mcv]
        feature_names = ["Gender", "Hemoglobin", "MCH", "MCHC", "MCV"]
        raw_input = np.array([raw_vals], dtype=float)
        scaled = models["scaler_anemia"].transform(raw_input)
        pred = int(models["model_anemia"].predict(scaled)[0])
        prob = None
        if hasattr(models["model_anemia"], "predict_proba"):
            prob = float(np.max(models["model_anemia"].predict_proba(scaled)[0]))

        # Calculate feature contributions
        imp = np.array(models["model_anemia"].feature_importances_, dtype=float)
        imp = imp / (np.sum(imp) or 1.0)
        scaled_dev = scaled[0]
        contrib = scaled_dev * imp
        norm_factor = np.sum(np.abs(contrib)) or 1.0
        norm_contrib = (contrib / norm_factor) * 100.0

        breakdown = [
            {"feature": name, "value": val, "contribution": round(float(c), 1)}
            for name, val, c in zip(feature_names, raw_vals, norm_contrib)
        ]
        breakdown.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return {
            "success": True,
            "prediction": pred,
            "is_elevated_risk": (pred == 1),
            "label": "Elevated Anemia Risk" if pred == 1 else "Normal Range Probability",
            "confidence": prob,
            "feature_breakdown": breakdown
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def predict_thyroid(features: Dict[str, Any]) -> Dict[str, Any]:
    models, _ = load_ml_models()
    if not ("model_thyroid" in models and "scaler_thyroid" in models):
        return {"success": False, "error": "Thyroid model or scaler is not loaded."}

    try:
        sex_val = 1 if features.get("sex", "male").lower() == "male" else 0
        feature_names = [
            "Age", "Sex", "On Thyroxine", "Query on Thyroxine", "On Antithyroid Med",
            "Sick", "Pregnant", "Thyroid Surgery", "I-131 Treatment",
            "Query Hypothyroid", "Query Hyperthyroid", "Lithium", "Goitre", "Tumor",
            "Hypopituitary", "Psych", "TSH", "T3", "TT4", "T4U", "FTI"
        ]
        data_row = [
            float(features.get("age", 35)),
            sex_val,
            int(features.get("on_thyroxine", False)),
            int(features.get("query_on_thyroxine", False)),
            int(features.get("on_antithyroid_medication", False)),
            int(features.get("sick", False)),
            int(features.get("pregnant", False)),
            int(features.get("thyroid_surgery", False)),
            int(features.get("i131_treatment", False)),
            int(features.get("query_hypothyroid", False)),
            int(features.get("query_hyperthyroid", False)),
            int(features.get("lithium", False)),
            int(features.get("goitre", False)),
            int(features.get("tumor", False)),
            int(features.get("hypopituitary", False)),
            int(features.get("psych", False)),
            float(features.get("tsh", 2.0)),
            float(features.get("t3", 1.8)),
            float(features.get("tt4", 100.0)),
            float(features.get("t4u", 1.0)),
            float(features.get("fti", 100.0))
        ]

        raw_input = np.array([data_row], dtype=float)
        scaled = models["scaler_thyroid"].transform(raw_input)
        raw_pred = models["model_thyroid"].predict(scaled)[0]
        prob = None
        if hasattr(models["model_thyroid"], "predict_proba"):
            prob = float(np.max(models["model_thyroid"].predict_proba(scaled)[0]))

        encoder = models.get("thyroid_target_encoder")
        if encoder:
            decoded_label = encoder.inverse_transform([int(raw_pred)])[0]
        else:
            decoded_label = str(raw_pred)

        is_negative = str(decoded_label).lower() == "negative"

        # Calculate top lab contributors (TSH, T3, TT4, T4U, FTI)
        imp = models["model_thyroid"].feature_importances_
        scaled_dev = scaled[0]
        contrib = scaled_dev * imp
        norm_factor = np.sum(np.abs(contrib)) or 1.0
        norm_contrib = (contrib / norm_factor) * 100.0

        breakdown = [
            {"feature": name, "value": val, "contribution": round(float(c), 1)}
            for name, val, c in zip(feature_names, data_row, norm_contrib)
        ]
        breakdown.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        return {
            "success": True,
            "raw_prediction": int(raw_pred),
            "label": decoded_label.title(),
            "is_negative": is_negative,
            "confidence": prob,
            "feature_breakdown": breakdown[:8]  # top 8 contributors
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def test_models() -> Dict[str, Any]:
    readiness = get_model_readiness()
    d_res = predict_diabetes(1, 120, 70, 20, 80, 25.0, 0.47, 33)
    a_res = predict_anemia("Female", 12.0, 27.0, 32.0, 85.0)
    t_res = predict_thyroid({"age": 35, "sex": "female", "tsh": 2.0, "t3": 1.8, "tt4": 100.0, "t4u": 1.0, "fti": 100.0})
    return {
        "readiness": readiness,
        "diabetes_test": d_res.get("success"),
        "anemia_test": a_res.get("success"),
        "thyroid_test": t_res.get("success"),
        "diabetes_explainability": bool(d_res.get("feature_breakdown"))
    }


if __name__ == "__main__":
    print("ML Service Test Results:", test_models())
