"""
Mediscan AI — Clinical Lab Report OCR & Biomarker Evaluation Engine
Advanced document preprocessing using OpenCV, multithreaded text extraction,
regex-based biomarker parsing, sanity filtering, reference interval evaluation,
and plain-language clinical insights.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    import pypdf
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# ============================================================
# COMPREHENSIVE LABORATORY REFERENCE INTERVALS
# ============================================================
REFERENCE_RANGES = {
    # Complete Blood Count (CBC)
    "hemoglobin": {"name": "Hemoglobin", "unit": "g/dL", "male": (13.5, 17.5), "female": (12.0, 15.5), "category": "CBC"},
    "wbc": {"name": "White Blood Cells", "unit": "x10³/µL", "range": (4.0, 11.0), "category": "CBC"},
    "platelets": {"name": "Platelets", "unit": "x10³/µL", "range": (150.0, 450.0), "category": "CBC"},
    "rbc": {"name": "Red Blood Cells", "unit": "x10⁶/µL", "male": (4.5, 5.9), "female": (4.0, 5.2), "category": "CBC"},
    "hematocrit": {"name": "Hematocrit (HCT)", "unit": "%", "male": (41.0, 50.0), "female": (36.0, 44.0), "category": "CBC"},
    "mcv": {"name": "MCV", "unit": "fL", "range": (80.0, 100.0), "category": "CBC"},
    "mch": {"name": "MCH", "unit": "pg", "range": (27.0, 33.0), "category": "CBC"},
    "mchc": {"name": "MCHC", "unit": "g/dL", "range": (32.0, 36.0), "category": "CBC"},

    # Metabolic, Glycemic & Lipids
    "glucose": {"name": "Fasting Blood Glucose", "unit": "mg/dL", "range": (70.0, 99.0), "category": "Metabolic"},
    "hba1c": {"name": "HbA1c", "unit": "%", "range": (4.0, 5.6), "category": "Metabolic"},
    "cholesterol": {"name": "Total Cholesterol", "unit": "mg/dL", "range": (125.0, 199.0), "category": "Lipids"},
    "hdl": {"name": "HDL ('Good') Cholesterol", "unit": "mg/dL", "male": (40.0, 70.0), "female": (50.0, 85.0), "category": "Lipids"},
    "ldl": {"name": "LDL ('Bad') Cholesterol", "unit": "mg/dL", "range": (50.0, 99.0), "category": "Lipids"},
    "triglycerides": {"name": "Triglycerides", "unit": "mg/dL", "range": (40.0, 149.0), "category": "Lipids"},

    # Renal & Electrolytes
    "creatinine": {"name": "Serum Creatinine", "unit": "mg/dL", "male": (0.7, 1.3), "female": (0.5, 1.1), "category": "Renal"},
    "urea_bun": {"name": "BUN / Urea", "unit": "mg/dL", "range": (7.0, 20.0), "category": "Renal"},
    "calcium": {"name": "Serum Calcium", "unit": "mg/dL", "range": (8.5, 10.2), "category": "Electrolytes"},
    "potassium": {"name": "Potassium", "unit": "mmol/L", "range": (3.5, 5.0), "category": "Electrolytes"},
    "sodium": {"name": "Sodium", "unit": "mmol/L", "range": (135.0, 145.0), "category": "Electrolytes"},

    # Endocrine / Thyroid
    "tsh": {"name": "TSH (Thyroid Stimulating Hormone)", "unit": "mIU/L", "range": (0.4, 4.0), "category": "Endocrine"},
    "t3": {"name": "Total T3", "unit": "nmol/L", "range": (1.2, 3.1), "category": "Endocrine"},
    "tt4": {"name": "Total T4 (Thyroxine)", "unit": "nmol/L", "range": (66.0, 181.0), "category": "Endocrine"},
    "fti": {"name": "Free Thyroxine Index (FTI)", "unit": "", "range": (60.0, 140.0), "category": "Endocrine"},
    "t4u": {"name": "T4 Uptake", "unit": "", "range": (0.8, 1.3), "category": "Endocrine"},

    # Diabetes & Vitals ML Parameters
    "insulin": {"name": "Fasting Insulin", "unit": "µIU/mL", "range": (2.6, 24.9), "category": "Metabolic"},
    "bmi": {"name": "Body Mass Index (BMI)", "unit": "kg/m²", "range": (18.5, 24.9), "category": "Metabolic"}
}

# ============================================================
# PLAIN-LANGUAGE BIOMARKER KNOWLEDGE BASE
# ============================================================
BIOMARKER_EXPLANATIONS = {
    "hemoglobin": {
        "what_is": "The oxygen-carrying protein inside red blood cells.",
        "high_meaning": "May indicate dehydration, chronic smoking, or living at high altitudes.",
        "low_meaning": "Suggests anemia, blood loss, or iron/vitamin deficiency. Common symptoms include fatigue and pallor."
    },
    "wbc": {
        "what_is": "Cells of the immune system defending against infection and foreign substances.",
        "high_meaning": "Often reflects bacterial infection, systemic inflammation, physical stress, or immune activation.",
        "low_meaning": "May point to viral infections, bone marrow suppression, or medication side effects."
    },
    "platelets": {
        "what_is": "Tiny blood cell fragments essential for normal blood clotting.",
        "high_meaning": "Can occur reactively with inflammation, iron deficiency, or marrow disorders.",
        "low_meaning": "Increases bleeding and bruising risk. Requires clinical evaluation."
    },
    "rbc": {
        "what_is": "Red blood cells responsible for oxygen transport throughout the body.",
        "high_meaning": "Associated with polycythemia, severe dehydration, or chronic hypoxia.",
        "low_meaning": "Consistent with anemia or red blood cell destruction."
    },
    "hematocrit": {
        "what_is": "The percentage of total blood volume made up of red blood cells.",
        "high_meaning": "Indicates concentrated blood, often from dehydration.",
        "low_meaning": "Reflects a reduced red cell mass, consistent with anemia."
    },
    "mcv": {
        "what_is": "The average physical size (volume) of red blood cells.",
        "high_meaning": "Macrocytosis, frequently linked to Vitamin B12 or folate deficiency.",
        "low_meaning": "Microcytosis, characteristic of iron-deficiency anemia or thalassemia trait."
    },
    "mch": {
        "what_is": "The average amount of hemoglobin inside each red blood cell.",
        "high_meaning": "Seen alongside macrocytic anemias.",
        "low_meaning": "Seen in hypochromic microcytic anemias."
    },
    "mchc": {
        "what_is": "The concentration of hemoglobin per volume of packed red blood cells.",
        "high_meaning": "Possible red cell dehydration or hereditary spherocytosis.",
        "low_meaning": "Consistent with pronounced iron deficiency."
    },
    "glucose": {
        "what_is": "The primary sugar circulating in blood, providing energy to cells.",
        "high_meaning": "Elevated fasting glucose (≥100 mg/dL pre-diabetes, ≥126 mg/dL diabetes concern) warrants glycemic review.",
        "low_meaning": "Hypoglycemia (<70 mg/dL), causing shakiness, dizziness, or sweating."
    },
    "hba1c": {
        "what_is": "Glycated hemoglobin reflecting your average blood sugar over the preceding 2–3 months.",
        "high_meaning": "Values ≥5.7% indicate prediabetes; ≥6.5% indicate diabetes. Useful for long-term glycemic monitoring.",
        "low_meaning": "May reflect frequent hypoglycemia or altered red blood cell turnover."
    },
    "cholesterol": {
        "what_is": "A waxy lipid substance essential for cell membranes and hormone synthesis.",
        "high_meaning": "Levels ≥200 mg/dL increase long-term cardiovascular plaque risk.",
        "low_meaning": "Generally optimal; very low levels may occur in malabsorption or hyperthyroidism."
    },
    "hdl": {
        "what_is": "High-Density Lipoprotein ('good' cholesterol) that carries cholesterol away from arteries back to the liver.",
        "high_meaning": "Cardioprotective. Higher levels (≥50–60 mg/dL) are generally beneficial.",
        "low_meaning": "Below 40 mg/dL in men or 50 mg/dL in women represents increased cardiovascular risk."
    },
    "ldl": {
        "what_is": "Low-Density Lipoprotein ('bad' cholesterol) that deposits excess cholesterol in arterial walls.",
        "high_meaning": "Primary contributor to arterial atherosclerosis. Lifestyle or medical management recommended.",
        "low_meaning": "Desirable for cardiovascular risk reduction."
    },
    "triglycerides": {
        "what_is": "The most common form of stored fat in the body.",
        "high_meaning": "Elevated with diets high in refined carbs/sugars, obesity, or metabolic syndrome.",
        "low_meaning": "Generally harmless; seen with very lean diets."
    },
    "creatinine": {
        "what_is": "A waste product produced by muscle metabolism, filtered entirely by the kidneys.",
        "high_meaning": "Direct biomarker of reduced kidney glomerular filtration rate.",
        "low_meaning": "Often seen in low muscle mass, advanced age, or hyperfiltration."
    },
    "urea_bun": {
        "what_is": "Waste product from protein breakdown cleared by the kidneys.",
        "high_meaning": "Seen in dehydration, high-protein intake, or impaired renal clearance.",
        "low_meaning": "Seen in liver disease or low protein intake."
    },
    "calcium": {
        "what_is": "Essential mineral for bone health, cardiac rhythm, and neuromuscular signaling.",
        "high_meaning": "Hypercalcemia; can relate to parathyroid conditions or vitamin D excess.",
        "low_meaning": "Hypocalcemia; can cause muscle spasms or numbness."
    },
    "potassium": {
        "what_is": "Key electrolyte regulating cardiac electrical stability and muscle contraction.",
        "high_meaning": "Hyperkalemia; critical electrolyte imbalance affecting heart rhythm.",
        "low_meaning": "Hypokalemia; often from diuretic use or gastrointestinal loss."
    },
    "sodium": {
        "what_is": "Major extracellular electrolyte regulating blood volume and fluid balance.",
        "high_meaning": "Hypernatremia; typically associated with water deficit or dehydration.",
        "low_meaning": "Hyponatremia; fluid overload or sodium loss."
    },
    "tsh": {
        "what_is": "Hormone secreted by the pituitary gland regulating thyroid metabolism.",
        "high_meaning": "Primary indicator of underactive thyroid (hypothyroidism).",
        "low_meaning": "Primary indicator of overactive thyroid (hyperthyroidism)."
    },
    "t3": {
        "what_is": "Triiodothyronine, the active thyroid hormone governing cellular metabolic rate.",
        "high_meaning": "Associated with hyperthyroidism or thyroid toxicosis.",
        "low_meaning": "Associated with hypothyroidism or severe non-thyroidal systemic illness."
    },
    "tt4": {
        "what_is": "Total Thyroxine, the primary circulating hormone produced by the thyroid gland.",
        "high_meaning": "Elevated in hyperthyroidism or high estrogen/pregnancy states.",
        "low_meaning": "Reduced in primary hypothyroidism or pituitary insufficiency."
    },
    "fti": {
        "what_is": "Free Thyroxine Index, estimated active unbound thyroxine adjusted for carrier proteins.",
        "high_meaning": "Hyperthyroid state.",
        "low_meaning": "Hypothyroid state."
    },
    "t4u": {
        "what_is": "Thyroid Hormone Binding Ratio / T4 Uptake measuring thyroid-binding globulin capacity.",
        "high_meaning": "Elevated binding capacity or anabolic steroid effects.",
        "low_meaning": "Elevated estrogen or pregnancy."
    },
    "insulin": {
        "what_is": "Pancreatic hormone enabling cellular glucose uptake and energy storage.",
        "high_meaning": "Hyperinsulinemia, strongly indicative of insulin resistance and metabolic syndrome.",
        "low_meaning": "Insufficient pancreatic beta-cell production (e.g. Type 1 diabetes)."
    },
    "bmi": {
        "what_is": "Body Mass Index, a ratio of weight to height indicating overall body mass classification.",
        "high_meaning": "Overweight or obesity range, increasing metabolic and cardiovascular risk factors.",
        "low_meaning": "Underweight classification, warranting nutritional assessment."
    }
}


def preprocess_image_cv(image: Image.Image) -> Image.Image:
    """
    Applies OpenCV grayscale, denoising, and CLAHE adaptive contrast enhancement
    to significantly increase Tesseract OCR recognition accuracy.
    """
    if not OPENCV_AVAILABLE:
        return image

    try:
        # Convert PIL Image to RGB numpy array
        img_np = np.array(image.convert("RGB"))
        
        # Convert to Grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Denoise using fast median filter
        denoised = cv2.medianBlur(gray, 3)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return Image.fromarray(enhanced)
    except Exception:
        return image


def extract_text_from_image(image: Image.Image) -> str:
    """Extract text from an image using preprocessed OpenCV pipeline + pytesseract."""
    if not PYTESSERACT_AVAILABLE:
        return "pytesseract library is not installed."
    try:
        preprocessed = preprocess_image_cv(image)
        return pytesseract.image_to_string(preprocessed)
    except Exception as e:
        err = str(e)
        if "tesseract is not installed" in err.lower() or "not in your path" in err.lower():
            return "Tesseract OCR engine not found in system PATH. Please install Tesseract-OCR to enable image OCR."
        return f"OCR Error: {err}"


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF using pypdf or PyMuPDF with fallbacks."""
    all_text = ""
    # Try pypdf first (pure Python, reliable)
    if PYPDF_AVAILABLE:
        try:
            reader = pypdf.PdfReader(pdf_file)
            for idx, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted and extracted.strip():
                    all_text += extracted + "\n"
            if all_text.strip():
                return all_text
        except Exception:
            pass

    # Try PyMuPDF if installed
    if FITZ_AVAILABLE:
        try:
            pdf_file.seek(0)
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page in doc:
                text = page.get_text()
                if text.strip():
                    all_text += text + "\n"
                elif PYTESSERACT_AVAILABLE:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    all_text += extract_text_from_image(img) + "\n"
            return all_text
        except Exception as exc:
            return f"PDF extraction error: {str(exc)}"

    return all_text if all_text.strip() else "No readable text could be parsed from this PDF document."


def clean_number(value: str) -> Optional[float]:
    try:
        val = value.replace(",", "").replace("O", "0").replace("o", "0").strip()
        num = float(val)
        return num if num >= 0 else None
    except (ValueError, AttributeError):
        return None


def extract_patient_demographics(text: str) -> Dict[str, Any]:
    """
    Intelligently extracts patient biological sex ('female' or 'male') and age (years)
    from medical laboratory reports to eliminate manual gender misconfiguration.
    Handles explicit labels, slash shorthands (e.g., '35Y/F', '45/Female'), and honorifics.
    """
    text_lower = text.lower()
    demographics = {"sex": None, "age": None}

    # 1. Explicit Sex / Gender Labels (e.g. "Sex: Female", "Gender: F", "Sex: Male", "Gender: Male")
    sex_match = re.search(r"(?:patient\s+)?(?:gender|sex)\s*[:\-–]?\s*(female|male|fem|f\b|m\b)", text_lower)
    if sex_match:
        val = sex_match.group(1).strip()
        if val in ["female", "fem", "f"]:
            demographics["sex"] = "female"
        elif val in ["male", "m"]:
            demographics["sex"] = "male"

    # 2. Age/Sex compound patterns (e.g. "35 Y / F", "42/Female", "Male / 55 Yrs")
    if not demographics["sex"]:
        slash1 = re.search(r"\b(\d{1,3})\s*(?:y|yr|yrs|years)?\s*[/|\\]\s*(female|male|fem|f\b|m\b)", text_lower)
        if slash1:
            demographics["age"] = int(slash1.group(1))
            s_val = slash1.group(2).strip()
            demographics["sex"] = "female" if s_val in ["female", "fem", "f"] else "male"
        else:
            slash2 = re.search(r"\b(female|male|fem|f\b|m\b)\s*[/|\\]\s*(\d{1,3})\s*(?:y|yr|yrs|years)?\b", text_lower)
            if slash2:
                s_val = slash2.group(1).strip()
                demographics["sex"] = "female" if s_val in ["female", "fem", "f"] else "male"
                demographics["age"] = int(slash2.group(2))

    # 3. Honorifics (Mrs., Ms., Miss -> female; Mr. -> male)
    if not demographics["sex"]:
        if re.search(r"\b(mrs\.|ms\.|miss)\s+[a-z]+", text_lower):
            demographics["sex"] = "female"
        elif re.search(r"\bmr\.\s+[a-z]+", text_lower):
            demographics["sex"] = "male"

    # 4. Standalone Age extraction if not yet found
    if not demographics["age"]:
        age_match = re.search(r"(?:patient\s+)?(?:age)\s*[:\-–]?\s*(\d{1,3})\s*(?:y|yr|yrs|years)?\b", text_lower)
        if age_match:
            a_val = int(age_match.group(1))
            if 1 <= a_val <= 120:
                demographics["age"] = a_val

    return demographics


def extract_lab_values(text: str) -> Dict[str, Optional[float]]:
    """
    Comprehensive regex biomarker extraction across CBC, Metabolic, Lipid, Renal, and Endocrine panels.
    """
    text_lower = text.lower()
    values = {k: None for k in REFERENCE_RANGES.keys()}
    values["blood_pressure"] = None

    patterns = {
        "hemoglobin": [
            r"(?:hemoglobin|haemoglobin|hgb|hb)\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"(?:total\s+hemoglobin)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "wbc": [
            r"(?:white\s+blood\s+cell[s]?|wbc|total\s+leukocyte\s+count|tlc)\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"(?:leukocytes)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "platelets": [
            r"(?:platelet[s]?\s+count|platelet[s]?|plt)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "rbc": [
            r"(?:red\s+blood\s+cell[s]?|rbc|total\s+rbc|erythrocyte[s]?)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "hematocrit": [
            r"(?:hematocrit|haematocrit|hct|packed\s+cell\s+volume|pcv)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "mcv": [
            r"(?:mean\s+corpuscular\s+volume|mcv)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "mch": [
            r"(?:mean\s+corpuscular\s+hemoglobin|mch)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "mchc": [
            r"(?:mean\s+corpuscular\s+hemoglobin\s+concentration|mchc)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "glucose": [
            r"(?:blood\s+glucose|fasting\s+glucose|glucose|blood\s+sugar|fasting\s+blood\s+sugar|fbs)\s*[:\-]?\s*(\d+(?:\.\d+)?)",
            r"(?:random\s+blood\s+sugar|rbs)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "hba1c": [
            r"(?:glycated\s+hemoglobin|glycohemoglobin|hba1c|a1c)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "cholesterol": [
            r"(?:total\s+cholesterol|serum\s+cholesterol|cholesterol)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "hdl": [
            r"(?:hdl\s+cholesterol|hdl|high\s+density\s+lipoprotein)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "ldl": [
            r"(?:ldl\s+cholesterol|ldl|low\s+density\s+lipoprotein)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "triglycerides": [
            r"(?:triglycerides|serum\s+triglycerides|tg)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "creatinine": [
            r"(?:serum\s+creatinine|creatinine|creat)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "urea_bun": [
            r"(?:blood\s+urea\s+nitrogen|bun|blood\s+urea|urea)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "calcium": [
            r"(?:serum\s+calcium|calcium|ca)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "potassium": [
            r"(?:serum\s+potassium|potassium|k\+?)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "sodium": [
            r"(?:serum\s+sodium|sodium|na\+?)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "tsh": [
            r"(?:thyroid\s+stimulating\s+hormone|tsh|ultra\s+tsh)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "t3": [
            r"(?:total\s+t3|serum\s+t3|triiodothyronine|t3)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "tt4": [
            r"(?:total\s+t4|serum\s+t4|total\s+thyroxine|thyroxine|tt4)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "fti": [
            r"(?:free\s+thyroxine\s+index|fti)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "t4u": [
            r"(?:t4\s+uptake|t4u)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "insulin": [
            r"(?:fasting\s+insulin|serum\s+insulin|insulin)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ],
        "bmi": [
            r"(?:body\s+mass\s+index|bmi)\s*[:\-]?\s*(\d+(?:\.\d+)?)"
        ]
    }

    # Extract with physiological sanity checks
    bounds = {
        "hemoglobin": (2.0, 30.0),
        "wbc": (0.5, 100.0),
        "platelets": (10.0, 1500.0),
        "rbc": (1.0, 10.0),
        "hematocrit": (10.0, 75.0),
        "mcv": (40.0, 160.0),
        "mch": (10.0, 60.0),
        "mchc": (15.0, 50.0),
        "glucose": (20.0, 800.0),
        "hba1c": (3.0, 20.0),
        "cholesterol": (40.0, 600.0),
        "hdl": (10.0, 150.0),
        "ldl": (20.0, 400.0),
        "triglycerides": (20.0, 1500.0),
        "creatinine": (0.1, 20.0),
        "urea_bun": (2.0, 200.0),
        "calcium": (4.0, 20.0),
        "potassium": (1.5, 10.0),
        "sodium": (90.0, 180.0),
        "tsh": (0.01, 100.0),
        "t3": (0.1, 25.0),
        "tt4": (5.0, 350.0),
        "fti": (5.0, 350.0),
        "t4u": (0.1, 5.0),
        "insulin": (0.5, 500.0),
        "bmi": (10.0, 80.0)
    }

    for name, pats in patterns.items():
        for pattern in pats:
            match = re.search(pattern, text_lower, flags=re.IGNORECASE)
            if match:
                num = clean_number(match.group(1))
                if num is not None:
                    low_b, high_b = bounds.get(name, (0, 99999))
                    if low_b <= num <= high_b:
                        values[name] = num
                        break

    # Also detect blood pressure (diastolic for ML models)
    bp_match = re.search(r"(?:blood\s+pressure|bp)\s*[:\-]?\s*(\d{2,3})\s*[/]\s*(\d{2,3})", text_lower)
    if bp_match:
        diastolic = clean_number(bp_match.group(2))
        if diastolic and 40 <= diastolic <= 200:
            values["blood_pressure"] = diastolic

    return values


def analyze_biomarker(test_key: str, value: Optional[float], sex: str = "male") -> Dict[str, Any]:
    """Evaluates an extracted value against reference intervals with biological sex adaptation."""
    if value is None:
        return {"status": "Not detected", "message": "Biomarker not detected in report."}

    ref = REFERENCE_RANGES[test_key]
    low, high = ref.get(sex.lower(), ref.get("range", (0, 100)))

    if value < low:
        status = "Low"
    elif value > high:
        status = "High"
    else:
        status = "Normal"

    explanation = BIOMARKER_EXPLANATIONS.get(test_key, {})

    return {
        "name": ref["name"],
        "category": ref["category"],
        "value": value,
        "status": status,
        "unit": ref["unit"],
        "reference_low": low,
        "reference_high": high,
        "message": f"{value} {ref['unit']} vs reference ({low} – {high} {ref['unit']})",
        "explanation": explanation
    }


def statistical_analysis(values: Dict[str, Optional[float]]) -> Optional[Dict[str, float]]:
    numeric = [v for v in values.values() if v is not None]
    if not numeric:
        return None
    arr = np.array(numeric, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "minimum": float(np.min(arr)),
        "maximum": float(np.max(arr))
    }


def calculate_rule_based_risk(values: Dict[str, Optional[float]], sex: str = "male") -> Tuple[str, List[str]]:
    """Multi-organ algorithmic screening triage across Hematology, Glycemic, Lipids, Renal, and Endocrine."""
    points = 0
    reasons = []

    # 1. Glycemic
    glu = values.get("glucose")
    a1c = values.get("hba1c")
    if a1c is not None:
        if a1c >= 6.5:
            points += 3
            reasons.append(f"HbA1c is {a1c}% (≥6.5%), meeting standard diagnostic criteria threshold for diabetes.")
        elif a1c >= 5.7:
            points += 1
            reasons.append(f"HbA1c is {a1c}% (5.7–6.4%), consistent with an increased risk for diabetes (prediabetes).")
    elif glu is not None:
        if glu >= 126:
            points += 2
            reasons.append(f"Fasting blood glucose is elevated ({glu} mg/dL ≥126 mg/dL), warranting clinical evaluation for diabetes.")
        elif glu >= 100:
            points += 1
            reasons.append(f"Fasting blood glucose is in the borderline elevated prediabetic range ({glu} mg/dL).")

    # 2. Hematology / Anemia
    hb = values.get("hemoglobin")
    hb_low = 13.5 if sex.lower() == "male" else 12.0
    if hb is not None:
        if hb < 9.5:
            points += 3
            reasons.append(f"Hemoglobin is markedly low ({hb} g/dL), indicating moderate-to-severe anemia risk.")
        elif hb < hb_low:
            points += 1
            reasons.append(f"Hemoglobin ({hb} g/dL) is below reference baseline ({hb_low} g/dL).")

    # 3. Cardiovascular / Lipids
    chol = values.get("cholesterol")
    ldl = values.get("ldl")
    tg = values.get("triglycerides")
    if ldl is not None and ldl >= 160:
        points += 2
        reasons.append(f"LDL cholesterol is substantially elevated ({ldl} mg/dL), representing higher atherogenic risk.")
    elif chol is not None and chol >= 240:
        points += 2
        reasons.append(f"Total cholesterol is high ({chol} mg/dL ≥240 mg/dL).")

    if tg is not None and tg >= 200:
        points += 1
        reasons.append(f"Triglycerides are elevated ({tg} mg/dL ≥200 mg/dL).")

    # 4. Renal Function
    creat = values.get("creatinine")
    creat_high = 1.3 if sex.lower() == "male" else 1.1
    if creat is not None and creat > creat_high:
        points += 2
        reasons.append(f"Serum creatinine is elevated ({creat} mg/dL), which may signal reduced renal clearance.")

    # 5. Thyroid
    tsh = values.get("tsh")
    if tsh is not None:
        if tsh > 4.5:
            points += 1
            reasons.append(f"TSH is elevated ({tsh} mIU/L), suggesting subclinical or primary hypothyroidism.")
        elif tsh < 0.3:
            points += 1
            reasons.append(f"TSH is suppressed ({tsh} mIU/L), raising consideration for hyperthyroidism.")

    # 6. Platelets & WBC
    wbc = values.get("wbc")
    plt = values.get("platelets")
    if wbc is not None and (wbc > 13.0 or wbc < 3.5):
        points += 1
        reasons.append(f"WBC count ({wbc} x10³/µL) is outside standard reference limits, reflecting possible infection or stress.")
    if plt is not None and plt < 100.0:
        points += 2
        reasons.append(f"Thrombocytopenia flag: Platelet count ({plt} x10³/µL) is noticeably reduced.")

    if points >= 4:
        level = "Higher concern"
    elif points >= 2:
        level = "Moderate concern"
    else:
        level = "Optimal Baseline"

    return level, reasons


def generate_health_suggestions(values: Dict[str, Optional[float]], sex: str = "male") -> Dict[str, List[str]]:
    """
    Synthesizes comprehensive, automated clinical lifestyle suggestions categorized into:
    - what_to_do: Positive evidence-backed proactive habits for better health
    - what_to_avoid: Harmful foods, substances, habits, or activities to steer clear of
    - what_to_ignore: Sensationalized myths, internet panic, or premature self-diagnoses to disregard
    """
    what_to_do: List[str] = []
    what_to_avoid: List[str] = []
    what_to_ignore: List[str] = []

    # 1. Glycemic / Blood Sugar
    glu = values.get("glucose")
    a1c = values.get("hba1c")
    if (a1c is not None and a1c >= 5.7) or (glu is not None and glu >= 100):
        what_to_do.append("Prioritize high-fiber complex carbohydrates (oats, legumes, chia seeds, broccoli) to slow glucose absorption.")
        what_to_do.append("Incorporate a 15-20 minute gentle walk immediately following major meals to improve muscle insulin sensitivity.")
        what_to_do.append("Establish a consistent sleep schedule (7-8 hours nightly), as fragmented sleep acutely raises cortisol and fasting blood glucose.")
        what_to_avoid.append("Avoid sugary beverages, sweetened sodas, fruit juices, and refined morning pastries that cause severe postprandial glucose spikes.")
        what_to_avoid.append("Avoid late-night carbohydrate-heavy snacking within 3 hours of sleep.")
        what_to_ignore.append("Ignore 'instant cure' diabetes herbs, extreme vinegar flushes, or zero-carb crash diets that risk electrolyte disturbances.")
        what_to_ignore.append("Do not panic over a single elevated fasting glucose; stress, poor sleep, or an incomplete fast can temporarily elevate readings.")

    # 2. Hematology / Anemia
    hb = values.get("hemoglobin")
    hb_low = 13.5 if sex.lower() == "male" else 12.0
    if hb is not None and hb < hb_low:
        what_to_do.append("Pair non-heme iron foods (spinach, lentils, fortified grains) with Vitamin C (lemon, bell peppers, tomatoes) to multiply absorption up to 3x.")
        what_to_do.append("Discuss serum ferritin, total iron binding capacity (TIBC), and vitamin B12/folate testing with your doctor.")
        what_to_do.append("Allow adequate pacing during workouts and listen to your body if lightheadedness or fatigue occurs.")
        what_to_avoid.append("Avoid drinking black tea, green tea, or coffee within 60 minutes of iron-rich meals (tannins and polyphenols inhibit iron uptake).")
        what_to_avoid.append("Avoid taking calcium supplements or high-calcium dairy at the exact same meal as your iron sources.")
        what_to_avoid.append("Avoid strenuous high-intensity interval exhaustion training until your red blood cell indices stabilize.")
        what_to_ignore.append("Ignore internet advice advocating megadose iron pills without a confirmed doctor's prescription, as iron overload causes organ toxicity.")

    # 3. Cardiovascular / Lipid Health
    chol = values.get("cholesterol")
    ldl = values.get("ldl")
    tg = values.get("triglycerides")
    hdl = values.get("hdl")
    if (chol is not None and chol >= 200) or (ldl is not None and ldl >= 130) or (tg is not None and tg >= 150):
        what_to_do.append("Emphasize monounsaturated and omega-3 fatty acids: extra virgin olive oil, wild salmon, walnuts, and flaxseeds.")
        what_to_do.append("Target at least 150 minutes of moderate aerobic activity weekly (brisk walking, cycling, or swimming) to boost protective HDL.")
        what_to_do.append("Increase soluble viscous fiber intake (psyllium husk, beans, apples) which binds intestinal bile acids and lowers circulating LDL.")
        what_to_avoid.append("Avoid artificial trans fats, partially hydrogenated vegetable oils, deep-fried fast foods, and commercially baked goods.")
        what_to_avoid.append("Avoid processed meats (bacon, sausages, hot dogs) which combine saturated fats with inflammatory preservatives.")
        what_to_avoid.append("Strictly limit excess alcohol and high-fructose corn syrup, both of which stimulate rapid hepatic triglyceride synthesis.")
        what_to_ignore.append("Ignore extreme claims that all dietary fats are toxic or that cholesterol can be eliminated overnight by drinking celery juice.")
        what_to_ignore.append("Do not stop doctor-prescribed statin therapy on your own based on unverified online forums.")

    # 4. Renal Function & Electrolytes
    creat = values.get("creatinine")
    bun = values.get("urea_bun")
    creat_high = 1.3 if sex.lower() == "male" else 1.1
    if (creat is not None and creat > creat_high) or (bun is not None and bun > 20):
        what_to_do.append("Maintain optimal daily hydration (2.0 to 2.5 liters of clean water daily, unless on a medically supervised fluid restriction).")
        what_to_do.append("Monitor blood pressure regularly at home, as vascular hypertension is the primary driver of progressive glomerular strain.")
        what_to_avoid.append("Avoid routine or unprescribed use of over-the-counter NSAIDs (ibuprofen, naproxen), which constrict renal afferent arterioles.")
        what_to_avoid.append("Avoid extreme high-dose protein powders, creatine monohydrate loading, or unverified pre-workout stim cocktails.")
        what_to_avoid.append("Avoid excessive dietary sodium (>2,300 mg/day) from canned soups, salty snacks, and fast foods.")
        what_to_ignore.append("Ignore aggressive 'kidney detox teas' or colon hydrotherapy fads—healthy kidneys detoxify naturally with adequate hydration.")
        what_to_ignore.append("A mild bump in serum creatinine can be caused simply by vigorous resistance training or mild dehydration the day before the test.")

    # 5. Thyroid Health
    tsh = values.get("tsh")
    if tsh is not None and (tsh > 4.5 or tsh < 0.4):
        what_to_do.append("Track energy levels, cold/heat sensitivity, heart rate, and hair/skin changes in a daily health log for your endocrinologist.")
        what_to_do.append("Ensure balanced dietary micronutrient intake including dietary selenium (Brazil nuts), zinc, and iodized salt.")
        what_to_avoid.append("Avoid self-prescribing high-dose kelp, bladderwrack, or unregulated thyroid glandular supplements without laboratory monitoring.")
        what_to_avoid.append("Avoid taking biotin (Vitamin B7) within 48–72 hours prior to future thyroid blood draws, as biotin causes severe lab assay interference.")
        what_to_ignore.append("Ignore sensationalized blogs claiming thyroid dysfunction can be cured entirely by eliminating all gluten or cruciferous vegetables.")

    # 6. General Baseline Optimization (if optimal or as general foundation)
    if not what_to_do:
        what_to_do.append("Maintain your current balanced diet emphasizing diverse whole colorful vegetables, lean proteins, and complex carbohydrates.")
        what_to_do.append("Continue regular physical activity targeting 150 minutes of aerobic exercise plus 2 resistance training sessions weekly.")
        what_to_do.append("Prioritize preventive wellness: stay hydrated, maintain social connections, and schedule regular annual health reviews.")

    if not what_to_avoid:
        what_to_avoid.append("Avoid prolonged sedentary sitting; take a 2-minute movement break every 45–60 minutes during work.")
        what_to_avoid.append("Avoid chronic sleep deprivation (<6 hours) which disrupts metabolic and immune homeostasis.")
        what_to_avoid.append("Avoid tobacco use and limit recreational alcohol consumption.")

    if not what_to_ignore:
        what_to_ignore.append("Ignore alarmist medical content on social media that promotes anxiety rather than clinical evidence.")
        what_to_ignore.append("Ignore unscientific detox cleanses—your liver, kidneys, and lungs continuously maintain natural homeostasis.")

    return {
        "what_to_do": what_to_do,
        "what_to_avoid": what_to_avoid,
        "what_to_ignore": what_to_ignore
    }


def generate_report(values: Dict[str, Optional[float]], analyses: Dict[str, Dict[str, Any]], sex: str = "male") -> str:
    lines = [
        "==================================================================",
        "          MEDISCAN AI — CLINICAL LABORATORY REPORT ANALYSIS       ",
        "==================================================================",
        f"Reference Profile Biological Sex: {sex.title()}",
        "",
        "------------------------------------------------------------------",
        "1. DETECTED BIOMARKER VALUES & STATUS",
        "------------------------------------------------------------------"
    ]

    detected_count = 0
    for key, item in analyses.items():
        val = item.get("value")
        if val is not None:
            detected_count += 1
            lines.append(f"• {item['name']:<32} : {val:>6} {item['unit']:<8} [ {item['status'].upper()} ] (Ref: {item['reference_low']} - {item['reference_high']})")

    if detected_count == 0:
        lines.append("• No numerical biomarkers detected in uploaded document.")

    risk_level, reasons = calculate_rule_based_risk(values, sex)

    lines.extend([
        "",
        "------------------------------------------------------------------",
        f"2. OVERALL SCREENING TRIAGE ASSESSMENT: [ {risk_level.upper()} ]",
        "------------------------------------------------------------------"
    ])

    if reasons:
        lines.append("Key Clinical Observations:")
        for r in reasons:
            lines.append(f"  ▲ {r}")
    else:
        lines.append("  ✓ All extracted biomarkers align with expected healthy reference intervals.")

    # 3. Actionable Health Suggestions
    suggestions = generate_health_suggestions(values, sex)
    lines.extend([
        "",
        "------------------------------------------------------------------",
        "3. ACTIONABLE HEALTH GUIDANCE (WHAT TO DO / WHAT TO AVOID / IGNORE)",
        "------------------------------------------------------------------"
    ])
    lines.append("PROACTIVE HEALTH STEPS (WHAT TO DO):")
    for s in suggestions["what_to_do"]:
        lines.append(f"  ✓ {s}")

    lines.append("\nPRECAUTIONARY MEASURES (WHAT TO AVOID):")
    for a in suggestions["what_to_avoid"]:
        lines.append(f"  ✗ {a}")

    lines.append("\nMISCONCEPTIONS & MYTHS (WHAT TO IGNORE):")
    for m in suggestions["what_to_ignore"]:
        lines.append(f"  ℹ {m}")

    lines.extend([
        "",
        "------------------------------------------------------------------",
        "4. RECOMMENDED QUESTIONS TO DISCUSS WITH YOUR PHYSICIAN",
        "------------------------------------------------------------------",
        "• Do any of my abnormal biomarker values require confirmatory repeat testing?",
        "• Could my recent diet, hydration, exercise, or medications have influenced these numbers?",
        "• What lifestyle or nutritional modifications are most appropriate for my profile?",
        "",
        "------------------------------------------------------------------",
        "CLINICAL SAFETY DISCLAIMER:",
        "This report was automatically synthesized by Mediscan AI for educational",
        "and informational reference only. It is NOT a medical diagnosis or",
        "clearance. Always review your laboratory findings with a licensed clinician.",
        "=================================================================="
    ])

    return "\n".join(lines)
