"""
Medi Scan AI — UI Components & Liquid Glass Styling Module
High-end Liquid Glassmorphism design system for modern medical technology.
"""

import streamlit as st
from typing import Dict, Any, List, Optional

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #0f172a;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 800 !important;
    color: #0b1f3a !important;
    letter-spacing: -0.03em;
}

/* ============================================================
   LIQUID GLASS CANVAS BACKGROUND
   ============================================================ */
.stApp {
    background: 
        radial-gradient(ellipse 65% 55% at 15% 10%, rgba(186, 230, 253, 0.55) 0%, transparent 70%),
        radial-gradient(ellipse 60% 50% at 88% 18%, rgba(153, 246, 228, 0.5) 0%, transparent 65%),
        radial-gradient(ellipse 70% 60% at 50% 90%, rgba(224, 231, 255, 0.45) 0%, transparent 70%),
        radial-gradient(ellipse 50% 40% at 30% 60%, rgba(240, 249, 255, 0.8) 0%, transparent 70%),
        linear-gradient(145deg, #f0f7ff 0%, #e8f3fc 50%, #f4fafb 100%) !important;
    background-attachment: fixed !important;
    min-height: 100vh;
}

/* ============================================================
   FROSTED LIQUID GLASS SIDEBAR
   ============================================================ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.88) 0%, rgba(240, 249, 255, 0.92) 100%) !important;
    backdrop-filter: blur(28px) saturate(220%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(220%) !important;
    border-right: 1.5px solid rgba(255, 255, 255, 0.85) !important;
    box-shadow: 4px 0 30px rgba(0, 80, 135, 0.05) !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(186, 230, 253, 0.6) !important;
}

/* Sidebar Radio Pills */
[data-testid="stSidebar"] .stRadio > div {
    gap: 6px !important;
}

[data-testid="stSidebar"] .stRadio label {
    background: rgba(255, 255, 255, 0.75) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.85) !important;
    border-radius: 12px !important;
    padding: 10px 14px !important;
    margin-bottom: 2px !important;
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    box-shadow: 0 2px 6px rgba(0, 80, 135, 0.03), inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255, 255, 255, 0.98) !important;
    border-color: #38bdf8 !important;
    transform: translateX(3px) !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.12) !important;
}

/* ============================================================
   LIQUID GLASS HEADER BANNER
   ============================================================ */
.ms-header-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, rgba(0, 75, 135, 0.94) 0%, rgba(13, 148, 136, 0.88) 100%) !important;
    backdrop-filter: blur(24px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(200%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.45) !important;
    border-radius: 22px !important;
    padding: 26px 34px !important;
    color: #ffffff !important;
    margin-bottom: 22px !important;
    box-shadow: 
        0 18px 45px -10px rgba(0, 80, 135, 0.25),
        inset 0 1.5px 2px 0 rgba(255, 255, 255, 0.6) !important;
    position: relative;
    overflow: hidden;
}

.ms-header-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 50%;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.22) 0%, transparent 100%);
    pointer-events: none;
}

.ms-header-title {
    font-size: 27px;
    font-weight: 800;
    margin: 0;
    color: #ffffff !important;
    letter-spacing: -0.03em;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
}

.ms-header-subtitle {
    font-size: 14.5px;
    color: #e0f2fe;
    margin-top: 5px;
    font-weight: 500;
    text-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
}

.ms-badge {
    background: rgba(255, 255, 255, 0.24) !important;
    backdrop-filter: blur(14px) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.6) !important;
    padding: 7px 18px !important;
    border-radius: 9999px !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1), inset 0 1px 1px rgba(255, 255, 255, 0.7) !important;
}

/* ============================================================
   EMERGENCY GLASS BANNER
   ============================================================ */
.ms-emergency-banner {
    background: linear-gradient(135deg, rgba(255, 241, 242, 0.92) 0%, rgba(255, 228, 230, 0.88) 100%) !important;
    backdrop-filter: blur(16px) !important;
    border: 1.5px solid rgba(254, 205, 211, 0.9) !important;
    border-left: 5px solid #e11d48 !important;
    border-radius: 16px !important;
    padding: 14px 20px !important;
    margin-bottom: 20px !important;
    font-size: 13px !important;
    color: #9f1239 !important;
    display: flex;
    align-items: center;
    gap: 14px;
    line-height: 1.5;
    box-shadow: 0 6px 20px rgba(225, 29, 72, 0.06), inset 0 1px 1px rgba(255, 255, 255, 0.9) !important;
}

/* ============================================================
   LIQUID GLASS CARDS
   ============================================================ */
.ms-card {
    background: rgba(255, 255, 255, 0.78) !important;
    backdrop-filter: blur(24px) saturate(200%) !important;
    -webkit-backdrop-filter: blur(24px) saturate(200%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.9) !important;
    border-radius: 20px !important;
    padding: 24px !important;
    box-shadow: 
        0 14px 38px -10px rgba(0, 80, 135, 0.08),
        0 4px 12px -2px rgba(13, 148, 136, 0.04),
        inset 0 1.5px 1.5px 0 rgba(255, 255, 255, 0.95) !important;
    margin-bottom: 20px !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.ms-card:hover {
    box-shadow: 
        0 20px 48px -10px rgba(0, 80, 135, 0.12),
        0 6px 16px -2px rgba(13, 148, 136, 0.06),
        inset 0 1.5px 1.5px 0 rgba(255, 255, 255, 0.98) !important;
    border-color: #ffffff !important;
}

.ms-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid rgba(226, 232, 240, 0.8);
    padding-bottom: 14px;
    margin-bottom: 16px;
}

.ms-card-title {
    font-size: 17.5px;
    font-weight: 700;
    color: #0f172a;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ============================================================
   FIX ALL FORM INPUTS, TEXTAREAS & SELECTBOXES
   ============================================================ */
[data-baseweb="input"],
[data-baseweb="input"] > div,
[data-baseweb="textarea"],
[data-baseweb="textarea"] > div,
.stTextInput > div > div,
.stTextArea > div > div,
.stNumberInput > div > div {
    background: #ffffff !important;
    border: 1.5px solid rgba(186, 230, 253, 0.95) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0, 80, 135, 0.04), inset 0 1px 2px rgba(0, 0, 0, 0.02) !important;
    transition: all 0.2s ease !important;
}

[data-baseweb="input"]:focus-within,
[data-baseweb="textarea"]:focus-within,
.stTextInput > div > div:focus-within,
.stTextArea > div > div:focus-within {
    border-color: #0284c7 !important;
    box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.18) !important;
    background: #ffffff !important;
}

input, textarea {
    color: #0f172a !important;
    font-weight: 500 !important;
}

/* SELECTBOXES & DROPDOWNS FIX */
[data-baseweb="select"],
[data-baseweb="select"] > div,
div[data-baseweb="select"] * {
    background-color: #ffffff !important;
    color: #0f172a !important;
}

[data-baseweb="select"] > div {
    border: 1.5px solid rgba(186, 230, 253, 0.95) !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0, 80, 135, 0.04) !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: #ffffff !important;
    border: 1.5px solid rgba(186, 230, 253, 0.9) !important;
    border-radius: 14px !important;
    box-shadow: 0 12px 32px rgba(0, 80, 135, 0.12) !important;
    color: #0f172a !important;
}

li[role="option"] {
    background-color: #ffffff !important;
    color: #0f172a !important;
    font-weight: 500 !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background-color: #e0f2fe !important;
    color: #005087 !important;
    font-weight: 700 !important;
}

/* ============================================================
   FIX FILE UPLOADER & UPLOADED FILE ITEM (IMAGES 1 & 2)
   ============================================================ */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(135deg, rgba(240, 249, 255, 0.92) 0%, rgba(204, 251, 241, 0.75) 100%) !important;
    backdrop-filter: blur(18px) saturate(180%) !important;
    border: 2px dashed rgba(2, 132, 199, 0.5) !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 24px rgba(0, 80, 135, 0.05), inset 0 1px 2px rgba(255, 255, 255, 0.9) !important;
    padding: 24px !important;
    transition: all 0.25s ease !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #0284c7 !important;
    background: linear-gradient(135deg, rgba(224, 242, 254, 0.98) 0%, rgba(153, 246, 228, 0.85) 100%) !important;
    box-shadow: 0 12px 30px rgba(2, 132, 199, 0.12) !important;
}

/* Fix Browse Files Button in File Uploader (Image 1) */
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #0284c7 0%, #0d9488 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25), inset 0 1px 1px rgba(255, 255, 255, 0.4) !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: linear-gradient(135deg, #0369a1 0%, #0f766e 100%) !important;
    color: #ffffff !important;
}

/* Fix Uploaded File Chip (Image 2) */
[data-testid="stFileUploaderFile"] {
    background: #ffffff !important;
    border: 1.5px solid rgba(186, 230, 253, 0.95) !important;
    border-radius: 14px !important;
    color: #0f172a !important;
    padding: 8px 14px !important;
    box-shadow: 0 4px 14px rgba(0, 80, 135, 0.06) !important;
}

[data-testid="stFileUploaderFile"] * {
    color: #0f172a !important;
    fill: #005087 !important;
}

/* Fix Expander Styling */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.82) !important;
    backdrop-filter: blur(18px) saturate(180%) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.95) !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 25px rgba(0, 80, 135, 0.05), inset 0 1px 1px #ffffff !important;
    overflow: hidden !important;
    margin-bottom: 14px !important;
}

[data-testid="stExpander"] details {
    border: none !important;
}

/* Fix Tabs Container & Active Tab */
[data-baseweb="tab-list"] {
    background: rgba(224, 242, 254, 0.75) !important;
    backdrop-filter: blur(16px) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.95) !important;
    border-radius: 16px !important;
    padding: 6px !important;
    gap: 6px !important;
    margin-bottom: 20px !important;
    box-shadow: inset 0 2px 4px rgba(0, 80, 135, 0.04) !important;
}

[data-baseweb="tab"] {
    border-radius: 12px !important;
    padding: 10px 20px !important;
    color: #475569 !important;
    font-weight: 600 !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

[data-baseweb="tab"][aria-selected="true"] {
    background: #ffffff !important;
    color: #005087 !important;
    font-weight: 800 !important;
    box-shadow: 0 4px 14px rgba(0, 80, 135, 0.12), inset 0 1px 1px #ffffff !important;
}

/* ============================================================
   ALL BUTTONS & FORM SUBMIT BUTTONS (IMAGES 4 & 5)
   ============================================================ */
.stButton > button,
[data-testid="stFormSubmitButton"] > button,
.stFormSubmitButton > button,
button[kind="primary"],
button[kind="secondary"],
button[kind="secondaryFormSubmit"] {
    background: linear-gradient(135deg, #0284c7 0%, #0d9488 100%) !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.45) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    padding: 12px 28px !important;
    font-size: 14.5px !important;
    letter-spacing: -0.01em !important;
    box-shadow: 
        0 6px 18px rgba(2, 132, 199, 0.28),
        inset 0 1px 1px rgba(255, 255, 255, 0.45) !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover,
.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #0369a1 0%, #0f766e 100%) !important;
    color: #ffffff !important;
    transform: translateY(-2px) !important;
    box-shadow: 
        0 10px 26px rgba(2, 132, 199, 0.38),
        inset 0 1px 1.5px rgba(255, 255, 255, 0.6) !important;
}

.stDownloadButton > button {
    background: #ffffff !important;
    color: #005087 !important;
    border-radius: 12px !important;
    border: 1.5px solid rgba(2, 132, 199, 0.35) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 700 !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 14px rgba(0, 80, 135, 0.08), inset 0 1px 1px #ffffff !important;
    transition: all 0.2s ease !important;
}

.stDownloadButton > button:hover {
    background: #005087 !important;
    color: #ffffff !important;
    border-color: #005087 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 22px rgba(0, 80, 135, 0.25) !important;
}

/* ============================================================
   LIQUID PILL BADGES
   ============================================================ */
.ms-pill-blue {
    background: linear-gradient(135deg, rgba(224, 242, 254, 0.95) 0%, rgba(186, 230, 253, 0.85) 100%);
    color: #0369a1;
    border: 1px solid rgba(14, 165, 233, 0.35);
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 12.5px;
    font-weight: 700;
    box-shadow: 0 2px 6px rgba(2, 132, 199, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.9);
}

.ms-pill-teal {
    background: linear-gradient(135deg, rgba(204, 251, 241, 0.95) 0%, rgba(153, 246, 228, 0.85) 100%);
    color: #0f766e;
    border: 1px solid rgba(20, 184, 166, 0.35);
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 12.5px;
    font-weight: 700;
    box-shadow: 0 2px 6px rgba(13, 148, 136, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.9);
}

.ms-pill-amber {
    background: linear-gradient(135deg, rgba(254, 243, 199, 0.95) 0%, rgba(253, 230, 138, 0.85) 100%);
    color: #92400e;
    border: 1px solid rgba(245, 158, 11, 0.35);
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 12.5px;
    font-weight: 700;
    box-shadow: 0 2px 6px rgba(217, 119, 6, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.9);
}

.ms-pill-red {
    background: linear-gradient(135deg, rgba(254, 226, 226, 0.95) 0%, rgba(254, 202, 202, 0.85) 100%);
    color: #991b1b;
    border: 1px solid rgba(239, 68, 68, 0.35);
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 12.5px;
    font-weight: 700;
    box-shadow: 0 2px 6px rgba(220, 38, 38, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.9);
}

.ms-pill-green {
    background: linear-gradient(135deg, rgba(220, 252, 231, 0.95) 0%, rgba(187, 247, 208, 0.85) 100%);
    color: #166534;
    border: 1px solid rgba(34, 197, 94, 0.35);
    padding: 5px 14px;
    border-radius: 9999px;
    font-size: 12.5px;
    font-weight: 700;
    box-shadow: 0 2px 6px rgba(22, 163, 74, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.9);
}

/* ============================================================
   LIQUID CHAT BUBBLES
   ============================================================ */
.ms-chat-bubble-user {
    background: linear-gradient(135deg, rgba(224, 242, 254, 0.94) 0%, rgba(186, 230, 253, 0.88) 100%) !important;
    backdrop-filter: blur(16px) !important;
    color: #034977 !important;
    border: 1.5px solid rgba(255, 255, 255, 0.95) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 16px 22px !important;
    margin: 10px 0 14px auto !important;
    max-width: 80% !important;
    font-size: 14.5px !important;
    line-height: 1.6 !important;
    box-shadow: 0 8px 24px rgba(2, 132, 199, 0.1), inset 0 1px 1px rgba(255, 255, 255, 0.95) !important;
}

.ms-chat-bubble-ai {
    background: rgba(255, 255, 255, 0.86) !important;
    backdrop-filter: blur(20px) saturate(190%) !important;
    color: #1e293b !important;
    border: 1.5px solid rgba(255, 255, 255, 0.95) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 18px 24px !important;
    margin: 10px auto 16px 0 !important;
    max-width: 86% !important;
    font-size: 14.5px !important;
    line-height: 1.65 !important;
    box-shadow: 0 10px 30px rgba(0, 80, 135, 0.06), inset 0 1.5px 1px rgba(255, 255, 255, 0.95) !important;
}

/* Chat Input Box */
[data-testid="stChatInput"] > div {
    background: #ffffff !important;
    border: 1.5px solid rgba(186, 230, 253, 0.95) !important;
    border-radius: 16px !important;
    box-shadow: 0 6px 20px rgba(0, 80, 135, 0.08) !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .ms-header-banner {
        flex-direction: column;
        align-items: flex-start;
        gap: 14px;
        padding: 20px 22px !important;
    }
    .ms-chat-bubble-user, .ms-chat-bubble-ai {
        max-width: 95% !important;
    }
}
</style>
"""


def apply_custom_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header(title: str, subtitle: str, badge: str = "MediScan Clinical"):
    html = (
        f'<div class="ms-header-banner">'
        f'<div>'
        f'<h1 class="ms-header-title">{title}</h1>'
        f'<div class="ms-header-subtitle">{subtitle}</div>'
        f'</div>'
        f'<div class="ms-badge">{badge}</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_emergency_banner():
    html = (
        '<div class="ms-emergency-banner">'
        '<span style="background: rgba(220, 38, 38, 0.12); color: #b91c1c; font-weight: 800; font-size: 11px; padding: 4px 10px; border-radius: 6px; letter-spacing: 0.05em; border: 1px solid rgba(220, 38, 38, 0.25);">CLINICAL NOTICE</span>'
        '<div>'
        '<strong>Safety Advisory:</strong> MediScan provides automated observational analysis for '
        '<strong>educational and clinical reference only</strong>. It is not an FDA-cleared diagnostic device. '
        'If experiencing severe chest pain, acute shortness of breath, sudden weakness, or trauma, '
        'contact emergency medical services (911 / 112) or visit an emergency room immediately.'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_status_chip(status: str) -> str:
    s = status.strip().capitalize()
    if s == "Normal":
        return f'<span class="ms-pill-green">✓ {s}</span>'
    elif s == "High":
        return f'<span class="ms-pill-red">▲ {s}</span>'
    elif s == "Low":
        return f'<span class="ms-pill-blue">▼ {s}</span>'
    return f'<span class="ms-pill-amber">• {s}</span>'


def render_range_meter(test_name: str, value: Optional[float], sex: str = "male") -> str:
    """Renders biomarker range meter as inline HTML without multiline indentation to prevent code block parsing."""
    if value is None:
        return '<div style="font-size: 12px; color: #64748b; font-style: italic; margin-top: 8px;">Biomarker not detected in document.</div>'
    from ocr_service import REFERENCE_RANGES
    ref = REFERENCE_RANGES.get(test_name, {"unit": "", "range": (0, 100)})
    low, high = ref.get(sex, ref.get("range", (0, 100)))
    span = max(1.0, high - low)
    scale_min = max(0, low - span * 0.8)
    scale_max = high + span * 0.8
    pct = ((value - scale_min) / (scale_max - scale_min)) * 100 if scale_max > scale_min else 50
    pct = max(4, min(96, pct))
    norm_start = ((low - scale_min) / (scale_max - scale_min)) * 100
    norm_end = ((high - scale_min) / (scale_max - scale_min)) * 100

    html = (
        f'<div style="margin-top: 10px; margin-bottom: 4px;">'
        f'<div style="display: flex; justify-content: space-between; font-size: 11.5px; color: #475569; font-weight: 600; margin-bottom: 6px;">'
        f'<span>Ref: {low} – {high} {ref["unit"]}</span>'
        f'<span>Detected: <b style="color: #005087;">{value}</b></span>'
        f'</div>'
        f'<div style="position: relative; width: 100%; height: 8px; background: rgba(226, 232, 240, 0.85); border-radius: 9999px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.06);">'
        f'<div style="position: absolute; left: {norm_start}%; width: {norm_end - norm_start}%; height: 100%; background: linear-gradient(90deg, #10b981 0%, #059669 100%); opacity: 0.9;"></div>'
        f'<div style="position: absolute; left: 0%; width: {norm_start}%; height: 100%; background: #94a3b8; opacity: 0.5;"></div>'
        f'<div style="position: absolute; left: {norm_end}%; width: {100 - norm_end}%; height: 100%; background: #f87171; opacity: 0.75;"></div>'
        f'</div>'
        f'<div style="position: relative; width: 100%; height: 14px; margin-top: -11px;">'
        f'<div style="position: absolute; left: calc({pct}% - 7px); top: 0px; width: 14px; height: 14px; background: #005087; border: 2.5px solid #ffffff; border-radius: 50%; box-shadow: 0 2px 6px rgba(0,80,135,0.4);"></div>'
        f'</div>'
        f'</div>'
    )
    return html


def render_risk_card(risk_level: str, reasons: List[str]) -> str:
    if risk_level == "Higher concern":
        rail, bg, text_color, badge = "#e11d48", "linear-gradient(135deg, rgba(255, 241, 242, 0.92) 0%, rgba(255, 228, 230, 0.85) 100%)", "#9f1239", "HIGH CLINICAL ATTENTION"
    elif risk_level == "Moderate concern":
        rail, bg, text_color, badge = "#d97706", "linear-gradient(135deg, rgba(254, 243, 199, 0.92) 0%, rgba(253, 230, 138, 0.85) 100%)", "#92400e", "ADVISORY MONITORING"
    else:
        rail, bg, text_color, badge = "#059669", "linear-gradient(135deg, rgba(240, 253, 244, 0.92) 0%, rgba(220, 252, 231, 0.85) 100%)", "#065f46", "EXPECTED BASELINE"

    reasons_html = "".join([f"<li style='margin-bottom: 7px;'>{r}</li>" for r in reasons]) if reasons else "<li>No prominent analytical flags identified across extracted biomarkers.</li>"

    html = (
        f'<div style="border-left: 5px solid {rail}; background: {bg}; padding: 20px 24px; border-radius: 16px; border: 1.5px solid rgba(255, 255, 255, 0.95); border-left-color: {rail}; margin: 18px 0; backdrop-filter: blur(16px); box-shadow: 0 8px 24px rgba(0,80,135,0.05), inset 0 1px 1px #ffffff;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">'
        f'<h4 style="margin: 0; color: {text_color}; font-size: 17.5px; font-weight: 800;">Screening Assessment: {risk_level}</h4>'
        f'<span style="font-size: 11px; font-weight: 700; color: {rail}; letter-spacing: 0.05em; background: rgba(255,255,255,0.95); padding: 4px 12px; border-radius: 9999px; border: 1px solid {rail}; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">{badge}</span>'
        f'</div>'
        f'<ul style="margin: 8px 0 0 18px; padding: 0; color: #1e293b; font-size: 14px; line-height: 1.6;">'
        f'{reasons_html}'
        f'</ul>'
        f'</div>'
    )
    return html


def render_structured_analysis(analysis_data: Dict[str, Any]):
    """Renders the comprehensive structured clinical vision analysis in liquid glass cards."""
    modality = analysis_data.get("modality", "Medical Imaging")
    anatomical = analysis_data.get("anatomical_region", "General")
    summary = analysis_data.get("summary", "No summary available.")
    quality = analysis_data.get("technical_quality", {})
    observations = analysis_data.get("observations", [])
    findings = analysis_data.get("possible_findings", [])
    limitations = analysis_data.get("limitations", [])
    terminology = analysis_data.get("medical_terminology", [])
    questions = analysis_data.get("questions_for_doctor", [])
    disclaimer = analysis_data.get("medical_disclaimer", "")

    # Modality & Region Liquid Glass Hero
    st.markdown(f"""
    <div class="ms-card" style="border-top: 4px solid #005087; background: linear-gradient(135deg, rgba(255, 255, 255, 0.88) 0%, rgba(240, 249, 255, 0.88) 100%) !important;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
            <div>
                <span class="ms-pill-blue">Modality</span>
                <h3 style="margin: 8px 0 4px 0; font-size: 22px; color: #005087;">{modality}</h3>
                <span style="color: #475569; font-size: 14.5px; font-weight: 500;">Anatomical Region: <b style="color: #0f172a;">{anatomical}</b></span>
            </div>
            <div>
                <span class="ms-pill-teal">Clarity: {quality.get('clarity', 'Adequate')}</span>
                <div style="font-size: 12.5px; color: #64748b; margin-top: 6px; max-width: 280px; line-height: 1.4;">{quality.get('notes', '')}</div>
            </div>
        </div>
        <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid rgba(226, 232, 240, 0.8); color: #1e293b; font-size: 15px; line-height: 1.65;">
            <b style="color: #005087;">Clinical Summary:</b> {summary}
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Visual Observations
        obs_html = "".join([f"<li style='margin-bottom: 9px;'>{o}</li>" for o in observations]) if observations else "<li>No specific visual landmarks highlighted.</li>"
        st.markdown(f"""
        <div class="ms-card">
            <div class="ms-card-header">
                <span class="ms-card-title">🔍 Visible Visual Observations</span>
                <span class="ms-pill-blue">{len(observations)} Highlighted</span>
            </div>
            <ul style="margin: 0; padding-left: 20px; font-size: 14px; color: #1e293b; line-height: 1.65;">
                {obs_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Technical Limitations
        lim_html = "".join([f"<li style='margin-bottom: 9px;'>{lim}</li>" for lim in limitations]) if limitations else "<li>Standard 2D projection imaging limits apply.</li>"
        st.markdown(f"""
        <div class="ms-card">
            <div class="ms-card-header">
                <span class="ms-card-title">⚠️ Limitations & Uncertainty</span>
                <span class="ms-pill-amber">Notice</span>
            </div>
            <ul style="margin: 0; padding-left: 20px; font-size: 14px; color: #475569; line-height: 1.65;">
                {lim_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Possible Findings (Differential)
        find_html = "".join([f"<li style='margin-bottom: 9px;'>{f}</li>" for f in findings]) if findings else "<li>No prominent focal pathology suspected on this view.</li>"
        st.markdown(f"""
        <div class="ms-card">
            <div class="ms-card-header">
                <span class="ms-card-title">🩺 Possible Clinical Considerations</span>
                <span class="ms-pill-teal">Differential</span>
            </div>
            <ul style="margin: 0; padding-left: 20px; font-size: 14px; color: #1e293b; line-height: 1.65;">
                {find_html}
            </ul>
            <div style="font-size: 12px; color: #64748b; font-style: italic; margin-top: 12px; padding: 8px 12px; background: rgba(240, 249, 255, 0.7); border-radius: 8px; border: 1px solid rgba(186, 230, 253, 0.6);">
                *Differential impressions must be verified by a licensed radiologist or attending clinician with full patient history.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Questions for Doctor
        q_html = "".join([f"<li style='margin-bottom: 9px;'><b>{q}</b></li>" for q in questions]) if questions else "<li>Discuss overall health status and follow-up scan schedules.</li>"
        st.markdown(f"""
        <div class="ms-card">
            <div class="ms-card-header">
                <span class="ms-card-title">💬 Questions to Ask Your Doctor</span>
                <span class="ms-pill-blue">Patient Prep</span>
            </div>
            <ol style="margin: 0; padding-left: 20px; font-size: 14px; color: #1e293b; line-height: 1.65;">
                {q_html}
            </ol>
        </div>
        """, unsafe_allow_html=True)

    # Health Guidance: What To Do, What To Avoid, What To Ignore
    health_guidance = analysis_data.get("health_guidance")
    if health_guidance:
        render_health_guidance_cards(health_guidance, title_prefix="AI Clinical Guidance")

    # Medical Terminology Explainer
    if terminology:
        st.markdown("""
        <div class="ms-card">
            <div class="ms-card-header">
                <span class="ms-card-title">📖 Medical Terminology Glossary</span>
                <span class="ms-pill-teal">Patient Plain Language</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px;">
        """, unsafe_allow_html=True)

        for item in terminology:
            term = item.get("term", "")
            definition = item.get("definition", "")
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.9); border: 1.5px solid rgba(186, 230, 253, 0.85); border-radius: 14px; padding: 14px 16px; box-shadow: 0 4px 12px rgba(0, 80, 135, 0.03), inset 0 1px 0 #ffffff;">
                <div style="font-weight: 800; color: #005087; font-size: 14px; margin-bottom: 5px;">{term}</div>
                <div style="color: #475569; font-size: 13px; line-height: 1.55;">{definition}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

    # Safety Notice Card
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(241, 245, 249, 0.92) 0%, rgba(226, 232, 240, 0.88) 100%); border: 1.5px solid rgba(203, 213, 225, 0.95); border-radius: 14px; padding: 16px 20px; margin-top: 16px; font-size: 13px; color: #334155; line-height: 1.55; backdrop-filter: blur(14px); box-shadow: 0 4px 14px rgba(0,0,0,0.03), inset 0 1px 0 #ffffff;">
        🛡️ <b>SAFETY & RESPONSIBILITY NOTICE:</b> {disclaimer}
    </div>
    """, unsafe_allow_html=True)


def render_health_guidance_cards(guidance: Dict[str, Any], title_prefix: str = "Actionable Health Guidance"):
    """
    Renders structured health guidance into 3 liquid glass cards:
    - Column 1: Proactive Steps (What To Do)
    - Column 2: Precautions & Hazards (What To Avoid)
    - Column 3: Myths & Panics (What To Ignore)
    """
    what_to_do = guidance.get("what_to_do", [])
    what_to_avoid = guidance.get("what_to_avoid", [])
    what_to_ignore = guidance.get("what_to_ignore", guidance.get("what_to_ignore_and_myths", []))

    st.markdown(f"""
    <div style="margin: 24px 0 14px 0;">
        <h3 style="margin: 0; font-size: 19px; color: #005087; font-weight: 800;">
            {title_prefix} — Evidence-Based Health Protocol
        </h3>
        <p style="margin: 4px 0 0 0; font-size: 13.5px; color: #64748b;">
            Synthesized directly from your clinical data: positive actions to adopt, hazards to steer clear of, and misconceptions to disregard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        do_items = "".join([f"<li style='margin-bottom: 10px;'>{item}</li>" for item in what_to_do]) if what_to_do else "<li>Maintain balanced nutrient intake and routine hydration.</li>"
        st.markdown(f"""
        <div class="ms-card" style="border-top: 4px solid #10b981; min-height: 290px; background: linear-gradient(180deg, rgba(236, 253, 245, 0.8) 0%, rgba(255, 255, 255, 0.85) 100%) !important;">
            <div class="ms-card-header">
                <span class="ms-card-title" style="color: #065f46;">Recommended Actions (What to Do)</span>
                <span style="font-size: 11px; font-weight: 700; color: #047857; background: #d1fae5; padding: 3px 10px; border-radius: 9999px;">Positive Steps</span>
            </div>
            <ul style="margin: 0; padding-left: 18px; font-size: 13.5px; color: #064e3b; line-height: 1.6;">
                {do_items}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        avoid_items = "".join([f"<li style='margin-bottom: 10px;'>{item}</li>" for item in what_to_avoid]) if what_to_avoid else "<li>Avoid unprescribed medications or extreme exertion.</li>"
        st.markdown(f"""
        <div class="ms-card" style="border-top: 4px solid #ef4444; min-height: 290px; background: linear-gradient(180deg, rgba(254, 242, 242, 0.8) 0%, rgba(255, 255, 255, 0.85) 100%) !important;">
            <div class="ms-card-header">
                <span class="ms-card-title" style="color: #991b1b;">Precautions & Hazards (What to Avoid)</span>
                <span style="font-size: 11px; font-weight: 700; color: #b91c1c; background: #fee2e2; padding: 3px 10px; border-radius: 9999px;">Critical Hazards</span>
            </div>
            <ul style="margin: 0; padding-left: 18px; font-size: 13.5px; color: #7f1d1d; line-height: 1.6;">
                {avoid_items}
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        ignore_items = "".join([f"<li style='margin-bottom: 10px;'>{item}</li>" for item in what_to_ignore]) if what_to_ignore else "<li>Ignore unverified social media medical panic.</li>"
        st.markdown(f"""
        <div class="ms-card" style="border-top: 4px solid #f59e0b; min-height: 290px; background: linear-gradient(180deg, rgba(254, 252, 232, 0.8) 0%, rgba(255, 255, 255, 0.85) 100%) !important;">
            <div class="ms-card-header">
                <span class="ms-card-title" style="color: #92400e;">Misconceptions to Disregard (What to Ignore)</span>
                <span style="font-size: 11px; font-weight: 700; color: #b45309; background: #fef3c7; padding: 3px 10px; border-radius: 9999px;">Debunked Myths</span>
            </div>
            <ul style="margin: 0; padding-left: 18px; font-size: 13.5px; color: #78350f; line-height: 1.6;">
                {ignore_items}
            </ul>
        </div>
        """, unsafe_allow_html=True)


def render_active_context_banner(scan_info: Dict[str, Any]):
    """
    Renders an active document context badge in the AI Assistant view,
    making it immediately apparent that the assistant has ingested the patient's report.
    """
    modality = scan_info.get("modality", "Clinical Document")
    filename = scan_info.get("filename", "Uploaded Report")
    analysis = scan_info.get("analysis", {})
    summary = analysis.get("summary") or analysis.get("executive_summary") or "Active clinical document loaded into assistant session."

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(240, 249, 255, 0.95) 0%, rgba(224, 242, 254, 0.9) 100%); border: 1.5px solid rgba(186, 230, 253, 0.95); border-left: 5px solid #005087; border-radius: 16px; padding: 16px 20px; margin-bottom: 20px; backdrop-filter: blur(20px); box-shadow: 0 8px 24px rgba(0, 80, 135, 0.06), inset 0 1px 1px #ffffff;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="width: 32px; height: 32px; border-radius: 8px; background: #0284c7; color: #ffffff; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 13px;">DOC</div>
                <div>
                    <div style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 800; color: #0284c7;">Connected Clinical Record</div>
                    <div style="font-size: 16px; font-weight: 800; color: #005087;">{filename} <span style="font-weight: 500; font-size: 13.5px; color: #475569;">({modality})</span></div>
                </div>
            </div>
            <span style="background: #e0f2fe; color: #0369a1; font-size: 11.5px; font-weight: 700; padding: 4px 12px; border-radius: 9999px; border: 1px solid #bae6fd;">
                Active Record
            </span>
        </div>
        <div style="margin-top: 10px; font-size: 13.5px; color: #1e293b; line-height: 1.55; border-top: 1px dashed rgba(186, 230, 253, 0.8); padding-top: 8px;">
            <b>Clinical Summary:</b> {summary}
        </div>
    </div>
    """, unsafe_allow_html=True)

