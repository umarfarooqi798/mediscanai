"""
Medi Scan AI — Scan History Service
Provides persistent, local JSON-based storage for medical scan analyses,
supporting search, filtering, detailed review, and privacy-first deletion.
"""

import os
import json
import base64
import io
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from PIL import Image

HISTORY_DIR = os.path.join(os.path.dirname(__file__), "data")
HISTORY_FILE = os.path.join(HISTORY_DIR, "scan_history.json")


def _ensure_history_dir():
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR, exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)


def _generate_thumbnail(image_bytes: bytes, max_dim: int = 160) -> str:
    """Generate a lightweight base64 thumbnail for fast history listing."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.thumbnail((max_dim, max_dim))
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=75)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception:
        return ""


def load_all_scans() -> List[Dict[str, Any]]:
    """Load all stored scans, sorted by timestamp descending."""
    _ensure_history_dir()
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            scans = json.load(f)
            if not isinstance(scans, list):
                return []
            return sorted(scans, key=lambda x: x.get("timestamp_epoch", 0), reverse=True)
    except Exception:
        return []


def save_scan(
    filename: str,
    file_size_kb: float,
    image_bytes: bytes,
    analysis_data: Dict[str, Any],
    model_used: str,
    user_notes: str = ""
) -> Dict[str, Any]:
    """Persist a newly analyzed scan into history."""
    _ensure_history_dir()
    scans = load_all_scans()

    now = datetime.now()
    scan_id = f"scan_{now.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000}"
    thumbnail = _generate_thumbnail(image_bytes)

    data = analysis_data.get("data", {}) if "data" in analysis_data else analysis_data

    record = {
        "id": scan_id,
        "filename": filename or "scan_image.jpg",
        "file_size_kb": round(file_size_kb, 1),
        "timestamp_epoch": time.time(),
        "timestamp_str": now.strftime("%Y-%m-%d %H:%M:%S"),
        "thumbnail_b64": thumbnail,
        "modality": data.get("modality", "Medical Imaging"),
        "anatomical_region": data.get("anatomical_region", "Unspecified"),
        "summary": data.get("summary", "No summary available."),
        "model_used": model_used,
        "user_notes": user_notes,
        "analysis": data
    }

    scans.insert(0, record)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(scans, f, indent=2, ensure_ascii=False)

    return record


def save_lab_report(
    filename: str,
    file_size_kb: float,
    patient_sex: str,
    patient_age: Optional[int],
    values: Dict[str, Any],
    analyses: Dict[str, Any],
    triage_level: str,
    triage_reasons: List[str],
    health_guidance: Dict[str, Any],
    executive_summary: Optional[str] = None,
    raw_text: str = ""
) -> Dict[str, Any]:
    """Persist an analyzed laboratory blood report into history."""
    _ensure_history_dir()
    scans = load_all_scans()

    # Avoid duplicate saves for the exact same file if already stored
    for s in scans:
        if s.get("filename") == filename and s.get("type") == "lab_report":
            return s

    now = datetime.now()
    scan_id = f"lab_{now.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000}"
    detected_count = len([v for v in values.values() if v is not None])
    summary_text = executive_summary or f"Laboratory panel with {detected_count} biomarkers evaluated. Triage level: {triage_level}."

    record = {
        "id": scan_id,
        "type": "lab_report",
        "filename": filename or "laboratory_report.pdf",
        "file_size_kb": round(file_size_kb, 1),
        "timestamp_epoch": time.time(),
        "timestamp_str": now.strftime("%Y-%m-%d %H:%M:%S"),
        "thumbnail_b64": "",
        "modality": "Laboratory Blood Panel",
        "anatomical_region": "Systemic Hematology & Biochemistry",
        "summary": summary_text,
        "patient_sex": patient_sex,
        "patient_age": patient_age,
        "triage_level": triage_level,
        "triage_reasons": triage_reasons,
        "model_used": "MediScan Clinical OCR Engine",
        "user_notes": f"Sex: {patient_sex.title()}, Age: {patient_age or 'N/A'}",
        "analysis": {
            "summary": summary_text,
            "executive_summary": summary_text,
            "detected_biomarkers": {k: v for k, v in values.items() if v is not None},
            "analyses": analyses,
            "triage_assessment": triage_level,
            "clinical_flags": triage_reasons,
            "health_guidance": health_guidance,
            "raw_text": raw_text[:2000]
        }
    }

    scans.insert(0, record)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(scans, f, indent=2, ensure_ascii=False)

    return record


def get_scan_by_id(scan_id: str) -> Optional[Dict[str, Any]]:
    """Find a specific scan record by its ID."""
    scans = load_all_scans()
    for s in scans:
        if s.get("id") == scan_id:
            return s
    return None


def search_scans(
    query: Optional[str] = None,
    modality: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search and filter scan records."""
    scans = load_all_scans()
    filtered = []

    q_lower = query.strip().lower() if query else None
    m_lower = modality.strip().lower() if modality and modality != "All Modalities" else None

    for s in scans:
        if m_lower and m_lower not in s.get("modality", "").lower():
            continue
        if q_lower:
            text_blob = f"{s.get('filename', '')} {s.get('modality', '')} {s.get('anatomical_region', '')} {s.get('summary', '')}".lower()
            if q_lower not in text_blob:
                continue
        filtered.append(s)

    return filtered


def delete_scan(scan_id: str) -> bool:
    """Permanently delete a specific scan from history."""
    _ensure_history_dir()
    scans = load_all_scans()
    initial_len = len(scans)
    scans = [s for s in scans if s.get("id") != scan_id]
    if len(scans) < initial_len:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(scans, f, indent=2)
        return True
    return False


def clear_all_history() -> bool:
    """Clear all scan history for privacy."""
    _ensure_history_dir()
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2)
    return True


def test_history() -> Dict[str, Any]:
    """Test history storage functionality."""
    _ensure_history_dir()
    scans = load_all_scans()
    return {
        "status": "ready",
        "total_scans": len(scans),
        "history_file": HISTORY_FILE
    }


if __name__ == "__main__":
    print("History Service Test:", test_history())
