"""
Medi Scan AI — AI-Powered Medical Image Analysis & Health Information Platform
Combines Groq Vision-based medical image interpretation, conversational AI assistance,
persistent scan history, clinical lab report OCR screening, and real trained ML risk predictors.
"""

import os
import io
import time
from datetime import datetime
from PIL import Image
import pandas as pd
import streamlit as st

import importlib
import groq_service
importlib.reload(groq_service)
import history_service
importlib.reload(history_service)
import ml_service
importlib.reload(ml_service)
import ocr_service
importlib.reload(ocr_service)
import ui_components
importlib.reload(ui_components)
import visualizer
importlib.reload(visualizer)
import pdf_service
importlib.reload(pdf_service)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="MediScan Health • Clinical Diagnostic & Lab Analyzer",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

ui_components.apply_custom_css()

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "Dashboard"

if "current_scan" not in st.session_state:
    st.session_state["current_scan"] = None

if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "assistant", "content": "Hello! I am your **MediScan Clinical Assistant**. You can ask questions about medical imaging, laboratory biomarker values, or specific findings from any scan you analyze. How can I help you today?"}
    ]

if "ocr_values" not in st.session_state:
    st.session_state["ocr_values"] = {}

if "patient_sex" not in st.session_state:
    st.session_state["patient_sex"] = "female"

if "patient_age" not in st.session_state:
    st.session_state["patient_age"] = None

if "auto_run_ml" not in st.session_state:
    st.session_state["auto_run_ml"] = False

if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = groq_service.get_active_groq_key() or ""

if "vision_model" not in st.session_state:
    st.session_state["vision_model"] = os.getenv("GROQ_VISION_MODEL", groq_service.DEFAULT_VISION_MODEL)

if "chat_model" not in st.session_state:
    st.session_state["chat_model"] = os.getenv("GROQ_CHAT_MODEL", groq_service.DEFAULT_CHAT_MODEL)

# Check ML Models status
models, model_errors = ml_service.load_ml_models()
ml_readiness = ml_service.get_model_readiness()
all_ml_ready = all(ml_readiness.values())

# Check Groq Key status
has_groq_key = bool(groq_service.get_active_groq_key(st.session_state.get("groq_api_key")))

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding: 12px 14px; background: rgba(255, 255, 255, 0.85); border: 1.5px solid rgba(255, 255, 255, 0.95); border-radius: 16px; box-shadow: 0 4px 16px rgba(0, 80, 135, 0.05);">
        <div style="width: 38px; height: 38px; border-radius: 10px; background: linear-gradient(135deg, #005087 0%, #0d9488 100%); display: flex; align-items: center; justify-content: center; color: #ffffff; font-weight: 800; font-size: 20px; line-height: 1; box-shadow: 0 3px 8px rgba(0,80,135,0.2);">
            +
        </div>
        <div>
            <div style="font-size: 17px; font-weight: 800; color: #0b1f3a; letter-spacing: -0.02em; line-height: 1.2;">MediScan Health</div>
            <div style="font-size: 11px; color: #64748b; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;">Clinical Diagnostic Suite</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_map = {
        "Dashboard": "Dashboard",
        "Lab Report Analyzer": "Lab Report Analyzer",
        "Medical Imaging Scan": "New Scan",
        "Scan History & Records": "Scan History",
        "Clinical AI Assistant": "AI Assistant",
        "ML Risk Predictors": "ML Risk Predictors",
        "Settings & System": "Settings",
        "Clinical Guidelines": "Help & Safety"
    }

    nav_options = list(nav_map.keys())
    rev_nav_map = {v: k for k, v in nav_map.items()}
    current_label = rev_nav_map.get(st.session_state["active_nav"], "Dashboard")

    selected_nav = st.radio(
        "Navigation",
        options=nav_options,
        index=nav_options.index(current_label) if current_label in nav_options else 0,
        label_visibility="collapsed"
    )

    new_active = nav_map[selected_nav]
    if new_active != st.session_state["active_nav"]:
        st.session_state["active_nav"] = new_active
        st.rerun()

    st.divider()

    # System Status Indicators
    st.markdown("<div style='font-size: 12px; font-weight: 700; color: #475569; margin-bottom: 8px;'>SYSTEM STATUS</div>", unsafe_allow_html=True)
    if has_groq_key:
        st.markdown("<span class='ms-pill-green'>🟢 Groq AI Connected</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='ms-pill-amber'>🔑 API Key Required</span>", unsafe_allow_html=True)

    if all_ml_ready:
        st.markdown("<span class='ms-pill-teal'>🟢 ML Predictors (3/3 Ready)</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='ms-pill-amber'>⚠️ ML Predictors Incomplete</span>", unsafe_allow_html=True)

    st.divider()

    st.markdown("<div style='font-size: 12px; font-weight: 700; color: #475569; margin-bottom: 8px;'>PATIENT SETTINGS</div>", unsafe_allow_html=True)
    current_sex = st.session_state.get("patient_sex", "female").lower()
    sex_idx = 0 if current_sex == "female" else 1
    patient_sex = st.selectbox(
        "Biological sex for reference intervals:",
        ["female", "male"],
        index=sex_idx,
        help="Adapts reference intervals for hemoglobin and endocrine parameters."
    )
    st.session_state["patient_sex"] = patient_sex

    st.divider()
    st.caption("Medi Scan AI v3.0 • Educational & Clinical Reference Platform")

# Top Emergency Safety Advisory (Persistent)
ui_components.render_emergency_banner()


# ============================================================
# VIEW 1: DASHBOARD
# ============================================================
if st.session_state["active_nav"] == "Dashboard":
    ui_components.render_header(
        "Your AI-Powered Medical Imaging Assistant",
        "Transparent observational analysis, clinical report OCR, and machine learning risk estimation."
    )

    # Metric Cards
    total_scans = len(history_service.load_all_scans())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="ms-card" style="text-align: center; padding: 18px;">
            <div style="font-size: 28px; font-weight: 800; color: #005087;">{total_scans}</div>
            <div style="font-size: 13px; color: #64748b; font-weight: 600; margin-top: 4px;">Scans in History</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="ms-card" style="text-align: center; padding: 18px;">
            <div style="font-size: 28px; font-weight: 800; color: #0d9488;">Groq Vision</div>
            <div style="font-size: 13px; color: #64748b; font-weight: 600; margin-top: 4px;">Vision AI Engine</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="ms-card" style="text-align: center; padding: 18px;">
            <div style="font-size: 28px; font-weight: 800; color: #0284c7;">3 Models</div>
            <div style="font-size: 13px; color: #64748b; font-weight: 600; margin-top: 4px;">Trained ML Predictors</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="ms-card" style="text-align: center; padding: 18px;">
            <div style="font-size: 28px; font-weight: 800; color: #16a34a;">Active</div>
            <div style="font-size: 13px; color: #64748b; font-weight: 600; margin-top: 4px;">Clinical OCR & Triage</div>
        </div>
        """, unsafe_allow_html=True)

    # Quick Navigation Hero
    st.markdown("<h3 style='margin-top: 15px;'>Diagnostic Modules</h3>", unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown("""
        <div class="ms-card" style="height: 100%;">
            <div style="width: 38px; height: 38px; border-radius: 10px; background: rgba(0, 80, 135, 0.1); color: #005087; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 14px; margin-bottom: 12px;">RAD</div>
            <h4 style="margin: 0 0 6px 0;">Medical Imaging Analysis</h4>
            <p style="font-size: 13px; color: #64748b; line-height: 1.5;">Observational evaluation of radiographs, MRI, CT, ultrasound, or dermatologic scans with structured clinical findings.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Imaging Module →", use_container_width=True, key="dash_start_scan"):
            st.session_state["active_nav"] = "New Scan"
            st.rerun()

    with q2:
        st.markdown("""
        <div class="ms-card" style="height: 100%;">
            <div style="width: 38px; height: 38px; border-radius: 10px; background: rgba(13, 148, 136, 0.1); color: #0d9488; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 14px; margin-bottom: 12px;">LAB</div>
            <h4 style="margin: 0 0 6px 0;">Laboratory Report Analyzer</h4>
            <p style="font-size: 13px; color: #64748b; line-height: 1.5;">Screen complete blood counts, metabolic panels, and lipid profiles with demographic calibration and PDF export.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Lab Analyzer →", use_container_width=True, key="dash_lab_ocr"):
            st.session_state["active_nav"] = "Lab Report Analyzer"
            st.rerun()

    with q3:
        st.markdown("""
        <div class="ms-card" style="height: 100%;">
            <div style="width: 38px; height: 38px; border-radius: 10px; background: rgba(2, 132, 199, 0.1); color: #0284c7; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 14px; margin-bottom: 12px;">ASK</div>
            <h4 style="margin: 0 0 6px 0;">Clinical Assistant</h4>
            <p style="font-size: 13px; color: #64748b; line-height: 1.5;">Direct conversational consultation grounded in your uploaded imaging records and laboratory findings.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Assistant →", use_container_width=True, key="dash_assistant"):
            st.session_state["active_nav"] = "AI Assistant"
            st.rerun()

    # Recent Records Section
    st.markdown("<h3 style='margin-top: 30px;'>Recent Patient Records & Diagnostics</h3>", unsafe_allow_html=True)
    recent_scans = history_service.load_all_scans()[:3]
    if recent_scans:
        for scan in recent_scans:
            sc1, sc2, sc3 = st.columns([1, 4, 1.5])
            with sc1:
                if scan.get("thumbnail_b64"):
                    st.markdown(f'<img src="data:image/jpeg;base64,{scan["thumbnail_b64"]}" style="width: 100%; border-radius: 8px; border: 1px solid #cbd5e1; max-height: 80px; object-fit: cover;" />', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="background: linear-gradient(135deg, rgba(224, 242, 254, 0.85) 0%, rgba(186, 230, 253, 0.65) 100%); border: 1.5px solid rgba(255, 255, 255, 0.85); height: 75px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 800; color: #005087; box-shadow: 0 4px 14px rgba(0,80,135,0.06);">REPORT</div>', unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""
                <div style="font-weight: 700; color: #005087; font-size: 15px;">{scan.get('modality', 'Record')} • {scan.get('anatomical_region', 'Clinical')}</div>
                <div style="font-size: 13px; color: #475569; margin-top: 2px;">{scan.get('summary', '')[:140]}...</div>
                <div style="font-size: 11px; color: #94a3b8; margin-top: 4px;">{scan.get('timestamp_str')} • {scan.get('filename')}</div>
                """, unsafe_allow_html=True)
            with sc3:
                dest_nav = "Lab Report Analyzer" if scan.get("type") == "lab_report" else "New Scan"
                if st.button("Inspect Record", key=f"dash_view_{scan['id']}", use_container_width=True):
                    st.session_state["current_scan"] = scan
                    st.session_state["active_nav"] = dest_nav
                    st.rerun()
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="ms-card" style="text-align: center; padding: 30px;">
            <div style="font-size: 14px; font-weight: 700; color: #64748b; margin-bottom: 4px;">No diagnostic records archived yet</div>
            <p style="color: #94a3b8; margin: 0; font-size: 13px;">Upload an imaging scan or laboratory report to populate this clinical archive.</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# VIEW 2: NEW SCAN (AI Medical Image Analysis)
# ============================================================
elif st.session_state["active_nav"] == "New Scan":
    ui_components.render_header(
        "Medical Imaging Observational Analysis",
        "Upload supported radiographs, MRI, CT, ultrasound, or dermatologic images for evidence-aware observational analysis."
    )

    if not has_groq_key:
        st.warning("Groq API Key is not configured. Please enter your `GROQ_API_KEY` in the Settings tab before running image analysis.")

    col_upload, col_preview = st.columns([1.1, 0.9])

    with col_upload:
        st.markdown("""
        <div class="ms-card">
            <div class="ms-card-header">
                <span class="ms-card-title">Upload Medical Image Scan</span>
                <span class="ms-pill-blue">PNG, JPG, JPEG, WEBP (Max 20MB)</span>
            </div>
            <div style="font-size: 13px; color: #64748b; margin-bottom: 14px;">
                Drag and drop your scan file below. Ensure sensitive personal identifying labels are omitted or anonymized.
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_img = st.file_uploader(
            "Upload Image File",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed"
        )

        user_clinical_notes = st.text_area(
            "Optional Clinical Context / Symptoms:",
            placeholder="e.g. 45-year-old patient presenting with 2-week persistent cough, no fever, non-smoker. Suspected consolidation check.",
            height=85
        )

        selected_model = st.selectbox(
            "Vision Model:",
            groq_service.AVAILABLE_VISION_MODELS,
            index=0 if st.session_state["vision_model"] not in groq_service.AVAILABLE_VISION_MODELS else groq_service.AVAILABLE_VISION_MODELS.index(st.session_state["vision_model"])
        )

    with col_preview:
        st.markdown("""
        <div class="ms-card">
            <div class="ms-card-header">
                <span class="ms-card-title">Image Preview & Inspection</span>
            </div>
        """, unsafe_allow_html=True)

        if uploaded_img is not None:
            image_bytes = uploaded_img.getvalue()
            is_valid, msg, pil_img = groq_service.validate_image_file(image_bytes, uploaded_img.name)

            if is_valid and pil_img:
                st.image(pil_img, use_container_width=True, caption=f"{uploaded_img.name} ({pil_img.size[0]}×{pil_img.size[1]}px, {len(image_bytes)/1024:.1f} KB)")
                st.success(f"✓ Valid image verified ({pil_img.format or 'Image'}).")
            else:
                st.error(f"❌ {msg}")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 50px 20px; background: linear-gradient(135deg, rgba(240, 249, 255, 0.8) 0%, rgba(204, 251, 241, 0.5) 100%); border: 2px dashed rgba(2, 132, 199, 0.45); border-radius: 20px; color: #475569; backdrop-filter: blur(14px); box-shadow: inset 0 2px 6px rgba(2, 132, 199, 0.05);">
                <div style="font-size: 42px; margin-bottom: 12px; filter: drop-shadow(0 2px 6px rgba(0, 80, 135, 0.15));">🩺</div>
                <div style="font-size: 15px; font-weight: 700; color: #005087;">No Image Selected</div>
                <div style="font-size: 12.5px; margin-top: 4px; color: #64748b;">Select or drop a medical image on the left to inspect.</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Analyze Button
    if uploaded_img is not None:
        if st.button("🚀 Analyze Medical Image with Groq AI", type="primary", use_container_width=True):
            if not has_groq_key:
                st.error("Cannot proceed: Groq API Key is not configured. Please add it in Settings.")
            else:
                image_bytes = uploaded_img.getvalue()
                is_valid, val_msg, pil_img = groq_service.validate_image_file(image_bytes, uploaded_img.name)

                if not is_valid:
                    st.error(f"Validation failed: {val_msg}")
                else:
                    progress_placeholder = st.empty()
                    with progress_placeholder.container():
                        st.info("🔄 Step 1/4: Validating image format & security integrity...")
                        time.sleep(0.3)
                        st.info("🔄 Step 2/4: Preparing base64 payload & clinical system prompt...")
                        time.sleep(0.2)
                        st.info(f"🔄 Step 3/4: Transmitting to Groq Vision ({selected_model})...")

                    img_fmt = pil_img.format or "JPEG"
                    result = groq_service.analyze_medical_image(
                        image_bytes=image_bytes,
                        img_format=img_fmt,
                        user_notes=user_clinical_notes,
                        model=selected_model,
                        api_key=st.session_state.get("groq_api_key")
                    )

                    progress_placeholder.empty()

                    if result.get("success"):
                        st.success("✅ Analysis completed successfully.")
                        analysis_data = result["data"]

                        # Persist in History
                        saved_record = history_service.save_scan(
                            filename=uploaded_img.name,
                            file_size_kb=len(image_bytes) / 1024.0,
                            image_bytes=image_bytes,
                            analysis_data=analysis_data,
                            model_used=selected_model,
                            user_notes=user_clinical_notes
                        )
                        st.session_state["current_scan"] = saved_record
                    else:
                        st.error(f"❌ Analysis failed: {result.get('message', 'Unknown error')}")

    # Display Active Analysis Result
    if st.session_state.get("current_scan"):
        curr = st.session_state["current_scan"]
        st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown(f"### 📋 Structured Findings: {curr.get('filename')}")

        analysis_payload = curr.get("analysis", curr)
        ui_components.render_structured_analysis(analysis_payload)

        # Action Buttons
        b1, b2, b3 = st.columns(3)
        with b1:
            chat_btn = st.button("💬 Ask Follow-up Questions in AI Assistant", use_container_width=True)
            if chat_btn:
                st.session_state["active_nav"] = "AI Assistant"
                st.rerun()
        with b2:
            import json
            report_json = json.dumps(analysis_payload, indent=2)
            st.download_button(
                label="⬇️ Export Analysis JSON",
                data=report_json,
                file_name=f"MediScan_{curr.get('id', 'export')}.json",
                mime="application/json",
                use_container_width=True
            )
        with b3:
            if st.button("🗑️ Clear Current Scan View", use_container_width=True):
                st.session_state["current_scan"] = None
                st.rerun()


# ============================================================
# VIEW 3: SCAN HISTORY
# ============================================================
elif st.session_state["active_nav"] == "Scan History":
    ui_components.render_header(
        "Diagnostic History & Archives",
        "Search, inspect, and export previously analyzed medical scans and clinical laboratory panels."
    )

    scans = history_service.load_all_scans()

    # Search & Filter controls
    f1, f2, f3 = st.columns([2, 1.5, 1])
    with f1:
        search_query = st.text_input("Search archives by keyword:", placeholder="e.g. CBC, Chest, Hemoglobin, fracture...")
    with f2:
        modalities = ["All Modalities"] + sorted(list(set(s.get("modality", "Unknown") for s in scans))) if scans else ["All Modalities"]
        selected_modality = st.selectbox("Filter by modality:", modalities)
    with f3:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Clear History", type="secondary", use_container_width=True):
            history_service.clear_all_history()
            st.session_state["current_scan"] = None
            st.success("Archive database cleared.")
            st.rerun()

    filtered_scans = history_service.search_scans(search_query, selected_modality)

    if not filtered_scans:
        st.markdown("""
        <div class="ms-card" style="text-align: center; padding: 45px 24px;">
            <div style="font-size: 15px; font-weight: 700; color: #475569; margin-bottom: 4px;">No Matching Records Found</div>
            <p style="font-size: 13px; color: #94a3b8; margin: 0;">Upload laboratory reports or medical images to build a persistent archive.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"**Archive Contains {len(filtered_scans)} Diagnostic Record(s)**")
        for scan in filtered_scans:
            is_lab = scan.get("type") == "lab_report"
            record_title = f"{scan.get('filename')} — {scan.get('modality', 'Diagnostic Record')} ({scan.get('timestamp_str')})"

            with st.expander(record_title):
                c1, c2 = st.columns([1, 3])
                with c1:
                    if is_lab:
                        sex_val = scan.get("patient_sex", "female").title()
                        age_val = f"{scan.get('patient_age')} Yrs" if scan.get("patient_age") else "Not Specified"
                        triage_val = scan.get("triage_level", "Routine screening")
                        st.markdown(f"""
                        <div style="background: rgba(240, 249, 255, 0.9); border: 1px solid #bae6fd; border-radius: 10px; padding: 12px; margin-bottom: 8px;">
                            <div style="font-size: 11px; text-transform: uppercase; font-weight: 800; color: #0284c7; margin-bottom: 6px;">Patient Profile</div>
                            <div style="font-size: 13px; color: #0f172a; line-height: 1.5;">
                                <b>Sex:</b> {sex_val}<br>
                                <b>Age:</b> {age_val}<br>
                                <b>Triage:</b> {triage_val}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        if scan.get("thumbnail_b64"):
                            st.markdown(f'<img src="data:image/jpeg;base64,{scan["thumbnail_b64"]}" style="width: 100%; border-radius: 8px; border: 1px solid #cbd5e1; max-height: 180px; object-fit: contain; background: #000;" />', unsafe_allow_html=True)

                    st.caption(f"File: {scan.get('filename')} ({scan.get('file_size_kb', 0)} KB)")
                    st.caption(f"Engine: {scan.get('model_used')}")
                    if scan.get("user_notes"):
                        st.info(f"**Notes:** {scan.get('user_notes')}")

                with c2:
                    st.markdown(f"**Clinical Summary:** {scan.get('summary')}")
                    analysis = scan.get("analysis", {})

                    if is_lab:
                        detected_bio = analysis.get("detected_biomarkers", {})
                        if detected_bio:
                            bio_badges = " ".join([f"<span class='ms-pill-blue' style='font-size:11px; margin: 2px;'>{k.upper()}: <b>{v}</b></span>" for k, v in list(detected_bio.items())[:8]])
                            st.markdown(f"<div style='margin-top: 8px;'><b>Biomarkers Detected ({len(detected_bio)}):</b><br>{bio_badges}</div>", unsafe_allow_html=True)
                    else:
                        obs = analysis.get("observations", [])
                        finds = analysis.get("possible_findings", [])
                        if obs:
                            st.markdown("**Key Observations:**")
                            for o in obs[:3]:
                                st.markdown(f"- {o}")
                        if finds:
                            st.markdown("**Differential Considerations:**")
                            for f in finds[:3]:
                                st.markdown(f"- {f}")

                act1, act2, act3 = st.columns([1.2, 1, 1])
                with act1:
                    if is_lab:
                        # Direct PDF Download for stored laboratory reports
                        analysis_dict = scan.get("analysis", {})
                        pdf_data = pdf_service.generate_lab_pdf_report(
                            filename=scan.get("filename", "lab_report.pdf"),
                            patient_sex=scan.get("patient_sex", "female"),
                            patient_age=scan.get("patient_age"),
                            values=analysis_dict.get("detected_biomarkers", {}),
                            analyses=analysis_dict.get("analyses", {}),
                            triage_level=scan.get("triage_level", "Routine screening"),
                            triage_reasons=scan.get("triage_reasons", []),
                            health_guidance=analysis_dict.get("health_guidance", {}),
                            executive_summary=scan.get("summary")
                        )
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_data,
                            file_name=f"Report_{scan['filename'].rsplit('.', 1)[0]}.pdf",
                            mime="application/pdf",
                            key=f"dl_pdf_hist_{scan['id']}",
                            use_container_width=True
                        )
                    else:
                        if st.button("Inspect Scan", key=f"view_hist_{scan['id']}", use_container_width=True):
                            st.session_state["current_scan"] = scan
                            st.session_state["active_nav"] = "New Scan"
                            st.rerun()
                with act2:
                    if st.button("Open in Assistant", key=f"chat_hist_{scan['id']}", use_container_width=True):
                        st.session_state["current_scan"] = scan
                        st.session_state["active_nav"] = "AI Assistant"
                        st.rerun()
                with act3:
                    if st.button("Delete Record", key=f"del_hist_{scan['id']}", use_container_width=True):
                        history_service.delete_scan(scan["id"])
                        st.success(f"Deleted record {scan['id']}")
                        st.rerun()


# ============================================================
# VIEW 4: AI ASSISTANT
# ============================================================
elif st.session_state["active_nav"] == "AI Assistant":
    ui_components.render_header(
        "Clinical Diagnostic Assistant",
        "Interactive consultation grounded in active imaging findings, laboratory parameters, and medical evidence."
    )

    current_scan = st.session_state.get("current_scan")

    # Scan Context Header
    if current_scan:
        ui_components.render_active_context_banner(current_scan)
    else:
        st.info("No diagnostic scan or report is currently selected. The assistant is operating in general clinical reference mode. Upload a laboratory report or imaging scan to link patient findings directly.")

    # Suggested Inquiries for What To Do, What To Avoid, What To Ignore, and Project Summary
    st.markdown("<div style='font-size: 12px; font-weight: 800; color: #005087; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 8px;'>RECOMMENDED CLINICAL INQUIRIES</div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    suggested_clicked = None
    with s1:
        if st.button("What to avoid (Precautions)", use_container_width=True):
            suggested_clicked = "Based on my active report and biomarker results, what specific foods, substances, habits, and physical stressors should I strictly avoid or eliminate?"
    with s2:
        if st.button("What to do (Healthy habits)", use_container_width=True):
            suggested_clicked = "What evidence-based daily proactive steps, dietary additions, hydration, sleep habits, and exercises should I do to optimize my health?"
    with s3:
        if st.button("Misconceptions to ignore", use_container_width=True):
            suggested_clicked = "What common internet myths, unscientific detox fads, or premature panic points should I ignore regarding my current test results?"
    with s4:
        if st.button("Physician consultation summary", use_container_width=True):
            suggested_clicked = "Please generate a clear, structured clinical project summary of my entire report and a list of key questions I should discuss with my doctor."

    st.markdown("<hr style='margin: 18px 0;'>", unsafe_allow_html=True)

    # Chat message container
    for msg in st.session_state["chat_messages"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="ms-chat-bubble-user"><b>You:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="ms-chat-bubble-ai"><b>MediScan Clinical Assistant:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)

    # Chat Input Form
    user_input = st.chat_input("Type your question about imaging, medical terms, or health findings...")

    prompt_to_send = user_input or suggested_clicked

    if prompt_to_send:
        if not has_groq_key:
            st.error("Please configure your Groq API Key in the Settings tab to chat with the AI assistant.")
        else:
            # Append user message
            st.session_state["chat_messages"].append({"role": "user", "content": prompt_to_send})

            scan_ctx = current_scan.get("analysis", current_scan) if current_scan else None

            with st.spinner("Medi Scan AI is formulating an evidence-aware response..."):
                response = groq_service.chat_with_assistant(
                    messages=st.session_state["chat_messages"][-8:],  # context window
                    scan_context=scan_ctx,
                    model=st.session_state["chat_model"],
                    api_key=st.session_state["groq_api_key"]
                )

            if response.get("success"):
                st.session_state["chat_messages"].append({"role": "assistant", "content": response["content"]})
            else:
                st.session_state["chat_messages"].append({"role": "assistant", "content": f"⚠️ Error: {response.get('message', 'Failed to retrieve response.')}"})

            st.rerun()

    # Chat Controls
    ctrl1, ctrl2 = st.columns([1, 4])
    with ctrl1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["chat_messages"] = [
                {"role": "assistant", "content": "Conversation cleared. How can I help you today?"}
            ]
            st.rerun()


# ============================================================
# VIEW 5: LAB REPORT ANALYZER (OCR & Biomarker Screening)
# ============================================================
elif st.session_state["active_nav"] == "Lab Report Analyzer":
    ui_components.render_header(
        "AI Medical Laboratory Report Analyzer",
        "Automated OCR extraction, reference interval evaluation, interactive Plotly gauges, and plain-language clinical insights."
    )

    uploaded_lab = st.file_uploader(
        "📥 Upload Laboratory Source Document (PDF, PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg", "pdf"]
    )

    if uploaded_lab:
        with st.spinner("Processing document, optimizing contrast with OpenCV, & extracting laboratory text..."):
            if "pdf" in uploaded_lab.type:
                raw_text = ocr_service.extract_text_from_pdf(uploaded_lab)
            else:
                raw_text = ocr_service.extract_text_from_image(Image.open(uploaded_lab))

        # Auto-detect patient demographics (Sex and Age) from OCR text
        demographics = ocr_service.extract_patient_demographics(raw_text)
        if demographics.get("sex"):
            detected_sex = demographics["sex"]
            if detected_sex != st.session_state.get("patient_sex"):
                st.session_state["patient_sex"] = detected_sex
                patient_sex = detected_sex
        if demographics.get("age"):
            st.session_state["patient_age"] = demographics["age"]

        st.success("✅ Laboratory report processed successfully.")

        if demographics.get("sex") or demographics.get("age"):
            sex_str = (demographics.get("sex") or patient_sex).upper()
            age_str = f"{demographics['age']} Years" if demographics.get("age") else "Not specified"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(240, 253, 250, 0.95) 0%, rgba(204, 251, 241, 0.9) 100%); border: 1.5px solid rgba(153, 246, 228, 0.95); border-left: 5px solid #0d9488; border-radius: 14px; padding: 12px 18px; margin: 12px 0 16px 0; font-size: 14px; color: #134e4a;">
                🧬 <b>Auto-Identified Patient Profile from Report:</b> Biological Sex: <b>{sex_str}</b> | Age: <b>{age_str}</b> — <i>All reference intervals and ML screening models are automatically calibrated to {sex_str.lower()}.</i>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🔍 Inspect Extracted Raw OCR Text"):
            st.text_area("Extracted Text", raw_text, height=180)

        values = ocr_service.extract_lab_values(raw_text)
        st.session_state["ocr_values"] = values

        # Analyze each biomarker
        analyses = {}
        detected_items = []
        normal_count = 0
        abnormal_count = 0

        for name, val in values.items():
            res = ocr_service.analyze_biomarker(name, val, patient_sex)
            analyses[name] = res
            if val is not None:
                detected_items.append(res)
                if res["status"] == "Normal":
                    normal_count += 1
                else:
                    abnormal_count += 1

        # Summary Metrics
        st.markdown("<h3 style='margin-top: 20px;'>📊 Laboratory Panel Overview</h3>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Biomarkers Detected", f"{len(detected_items)} / {len(values)}")
        m2.metric("Within Reference", normal_count)
        m3.metric("Flagged / Abnormal", abnormal_count)
        m4.metric("Biological Sex Baseline", patient_sex.title())

        # Multi-Biomarker Deviation Chart
        if detected_items:
            deviation_chart = visualizer.create_multi_biomarker_deviation_chart(detected_items)
            if deviation_chart:
                st.plotly_chart(deviation_chart, use_container_width=True)

            # Extracted Biomarker Gauges & Cards
            st.markdown("<h3 style='margin-top: 25px;'>🧪 Detected Biomarker Detail & Gauges</h3>", unsafe_allow_html=True)

            # Render in responsive 2-column layout
            for i in range(0, len(detected_items), 2):
                col_a, col_b = st.columns(2)
                with col_a:
                    item_a = detected_items[i]
                    key_a = [k for k, v in ocr_service.REFERENCE_RANGES.items() if v["name"] == item_a["name"]][0]
                    chip_a = ui_components.render_status_chip(item_a["status"])

                    st.markdown(f"""
                    <div class="ms-card" style="margin-bottom: 12px; padding-bottom: 14px;">
                        <div class="ms-card-header">
                            <div>
                                <span class="ms-pill-blue" style="font-size: 11px;">{item_a.get('category', 'Lab')}</span>
                                <span style="font-weight: 800; color: #005087; font-size: 16px; margin-left: 6px;">{item_a['name']}</span>
                            </div>
                            {chip_a}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    gauge_a = visualizer.create_biomarker_gauge(
                        test_key=key_a,
                        value=item_a["value"],
                        reference_low=item_a["reference_low"],
                        reference_high=item_a["reference_high"],
                        unit=item_a["unit"]
                    )
                    st.plotly_chart(gauge_a, use_container_width=True, key=f"gauge_{key_a}")

                    exp_a = item_a.get("explanation", {})
                    if exp_a:
                        with st.expander(f"📖 Plain-Language Guide: {item_a['name']}"):
                            st.markdown(f"**What it is:** {exp_a.get('what_is', 'Biomarker evaluated during standard clinical blood testing.')}")
                            if item_a["status"] == "High":
                                st.warning(f"**Elevated Value Meaning:** {exp_a.get('high_meaning', 'Value exceeds the typical baseline range.')}")
                            elif item_a["status"] == "Low":
                                st.info(f"**Low Value Meaning:** {exp_a.get('low_meaning', 'Value is below typical baseline range.')}")
                            else:
                                st.success("✓ **Normal Range:** Value aligns with expected healthy reference intervals.")

                if i + 1 < len(detected_items):
                    with col_b:
                        item_b = detected_items[i+1]
                        key_b = [k for k, v in ocr_service.REFERENCE_RANGES.items() if v["name"] == item_b["name"]][0]
                        chip_b = ui_components.render_status_chip(item_b["status"])

                        st.markdown(f"""
                        <div class="ms-card" style="margin-bottom: 12px; padding-bottom: 14px;">
                            <div class="ms-card-header">
                                <div>
                                    <span class="ms-pill-blue" style="font-size: 11px;">{item_b.get('category', 'Lab')}</span>
                                    <span style="font-weight: 800; color: #005087; font-size: 16px; margin-left: 6px;">{item_b['name']}</span>
                                </div>
                                {chip_b}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        gauge_b = visualizer.create_biomarker_gauge(
                            test_key=key_b,
                            value=item_b["value"],
                            reference_low=item_b["reference_low"],
                            reference_high=item_b["reference_high"],
                            unit=item_b["unit"]
                        )
                        st.plotly_chart(gauge_b, use_container_width=True, key=f"gauge_{key_b}")

                        exp_b = item_b.get("explanation", {})
                        if exp_b:
                            with st.expander(f"📖 Plain-Language Guide: {item_b['name']}"):
                                st.markdown(f"**What it is:** {exp_b.get('what_is', 'Biomarker evaluated during standard clinical blood testing.')}")
                                if item_b["status"] == "High":
                                    st.warning(f"**Elevated Value Meaning:** {exp_b.get('high_meaning', 'Value exceeds the typical baseline range.')}")
                                elif item_b["status"] == "Low":
                                    st.info(f"**Low Value Meaning:** {exp_b.get('low_meaning', 'Value is below typical baseline range.')}")
                                else:
                                    st.success("✓ **Normal Range:** Value aligns with expected healthy reference intervals.")
        else:
            st.info("ℹ️ No numerical biomarkers could be automatically matched from the report. Inspect the raw OCR text above or verify image clarity.")

        # Full Table
        st.markdown("<h3 style='margin-top: 25px;'>📋 Complete Reference Interval Table</h3>", unsafe_allow_html=True)
        table_rows = []
        for name, res in analyses.items():
            if res.get("value") is not None:
                table_rows.append({
                    "Category": res.get("category", "General"),
                    "Biomarker": res["name"],
                    "Observed Value": f"{res['value']} {res['unit']}",
                    "Normal Reference": f"{res['reference_low']} – {res['reference_high']} {res['unit']}",
                    "Clinical Status": res["status"]
                })
        if table_rows:
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        # Algorithmic multi-organ screening triage
        st.markdown("<h3 style='margin-top: 25px;'>⚠️ Algorithmic Multi-Organ Screening Triage</h3>", unsafe_allow_html=True)
        risk_level, risk_reasons = ocr_service.calculate_rule_based_risk(values, patient_sex)
        st.markdown(ui_components.render_risk_card(risk_level, risk_reasons), unsafe_allow_html=True)

        # Automated Health Guidance: What To Do, What To Avoid, What To Ignore
        health_suggestions = ocr_service.generate_health_suggestions(values, patient_sex)
        ui_components.render_health_guidance_cards(health_suggestions, title_prefix="Algorithmic Auto-Suggestions")

        # Groq AI Executive Project Summary & Deep Analysis
        ai_summary_key = f"ai_lab_summary_{uploaded_lab.name}"
        if has_groq_key:
            st.markdown("<hr style='margin: 25px 0;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='margin-bottom: 6px;'>Clinical Synthesis & Deep Analysis</h3>", unsafe_allow_html=True)
            st.caption("Links this laboratory panel to the clinical engine for executive synthesis and personalized advice.")

            if ai_summary_key not in st.session_state:
                if st.button("Generate Executive Clinical Synthesis & Guidance", type="primary", use_container_width=True):
                    with st.spinner("Analyzing panel data and formulating clinical synthesis..."):
                        ai_res = groq_service.generate_lab_executive_summary(
                            raw_ocr_text=raw_text,
                            detected_biomarkers={k: v for k, v in values.items() if v is not None},
                            triage_level=risk_level,
                            clinical_flags=risk_reasons,
                            patient_sex=patient_sex,
                            model=st.session_state["chat_model"],
                            api_key=st.session_state["groq_api_key"]
                        )
                        if ai_res.get("success"):
                            st.session_state[ai_summary_key] = ai_res["data"]
                            st.rerun()
                        else:
                            st.error(f"Failed to generate clinical summary: {ai_res.get('message')}")

            if ai_summary_key in st.session_state:
                ai_data = st.session_state[ai_summary_key]
                exec_summary_text = ai_data.get("executive_summary", "")

                st.markdown(f"""
                <div class="ms-card" style="border-top: 4px solid #005087; background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(240, 249, 255, 0.92) 100%) !important;">
                    <div class="ms-card-header">
                        <span class="ms-card-title">Clinical Executive Summary</span>
                        <span class="ms-pill-blue">Active Synthesis</span>
                    </div>
                    <div style="font-size: 15px; color: #1e293b; line-height: 1.65; margin: 10px 0;">
                        {exec_summary_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                ai_guidance = {
                    "what_to_do": ai_data.get("what_to_do", []),
                    "what_to_avoid": ai_data.get("what_to_avoid", []),
                    "what_to_ignore": ai_data.get("what_to_ignore", [])
                }
                ui_components.render_health_guidance_cards(ai_guidance, title_prefix="Clinical Recommendations")

        # Automatically synchronize into current_scan so AI Assistant is directly grounded
        active_summary = (
            st.session_state.get(ai_summary_key, {}).get("executive_summary")
            or f"Laboratory panel for {uploaded_lab.name}: {len(detected_items)} biomarkers evaluated. Risk assessment: {risk_level}."
        )
        active_guidance = st.session_state.get(ai_summary_key) or health_suggestions

        # Persist to history database
        file_size_kb = (uploaded_lab.size / 1024.0) if hasattr(uploaded_lab, "size") and uploaded_lab.size else 0.0
        lab_record = history_service.save_lab_report(
            filename=uploaded_lab.name,
            file_size_kb=file_size_kb,
            patient_sex=patient_sex,
            patient_age=demographics.get("age"),
            values=values,
            analyses=analyses,
            triage_level=risk_level,
            triage_reasons=risk_reasons,
            health_guidance=active_guidance,
            executive_summary=active_summary,
            raw_text=raw_text
        )

        st.session_state["current_scan"] = lab_record

        # Action Bar: Actions for PDF export, ML predictors & AI assistant
        st.markdown("<hr style='margin: 25px 0;'>", unsafe_allow_html=True)
        act1, act2, act3 = st.columns(3)
        with act1:
            pdf_bytes = pdf_service.generate_lab_pdf_report(
                filename=uploaded_lab.name,
                patient_sex=patient_sex,
                patient_age=demographics.get("age"),
                values=values,
                analyses=analyses,
                triage_level=risk_level,
                triage_reasons=risk_reasons,
                health_guidance=active_guidance,
                executive_summary=active_summary
            )
            base_name = uploaded_lab.name.rsplit(".", 1)[0]
            clean_date = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                label="Download Clinical Report (PDF)",
                data=pdf_bytes,
                file_name=f"MediScan_Report_{base_name}_{clean_date}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
        with act2:
            if st.button("Evaluate with ML Risk Predictors →", use_container_width=True):
                st.session_state["active_nav"] = "ML Risk Predictors"
                st.rerun()
        with act3:
            if st.button("Consult Clinical Assistant →", use_container_width=True):
                # Seed greeting message tailored to report if not already present
                greeting_text = (
                    f"Hello! I have reviewed your laboratory report for **{uploaded_lab.name}**.\n\n"
                    f"**Clinical Summary:** {active_summary}\n\n"
                    f"Feel free to ask about specific biomarkers, recommended lifestyle actions, precautions, or questions to ask your physician."
                )
                if not any(uploaded_lab.name in m.get("content", "") for m in st.session_state["chat_messages"]):
                    st.session_state["chat_messages"].append({"role": "assistant", "content": greeting_text})

                st.session_state["active_nav"] = "AI Assistant"
                st.rerun()
    else:
        st.markdown("""
        <div class="ms-card" style="text-align: center; padding: 45px 24px;">
            <div style="width: 44px; height: 44px; border-radius: 12px; background: rgba(13, 148, 136, 0.1); color: #0d9488; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 15px; margin: 0 auto 14px auto;">LAB</div>
            <h3 style="margin-bottom: 8px; color: #005087;">Upload Laboratory Report</h3>
            <p style="color: #64748b; max-width: 580px; margin: 0 auto 20px auto; line-height: 1.6;">
                Upload a complete blood count (CBC), lipid profile, metabolic panel, or endocrine report (PDF or image).
                MediScan applies OpenCV image preprocessing, extracts clinical biomarkers, compares demographic reference intervals, and formats a downloadable clinical PDF report.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# VIEW 6: ML RISK PREDICTORS (Trained XGBoost & LightGBM)
# ============================================================
elif st.session_state["active_nav"] == "ML Risk Predictors":
    ui_components.render_header(
        "Trained Clinical ML Risk Predictors",
        "Machine learning models trained on benchmark public cohorts for statistical risk estimation."
    )

    ocr_vals = st.session_state.get("ocr_values", {})
    detected_ml_params = {k: v for k, v in ocr_vals.items() if v is not None}

    if detected_ml_params:
        params_badges = " ".join([f"<span class='ms-pill-blue' style='font-size:12px; margin: 3px 4px;'>{k.upper()}: <b>{v}</b></span>" for k, v in detected_ml_params.items()])
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(240, 249, 255, 0.95) 0%, rgba(224, 242, 254, 0.92) 100%); border: 1.5px solid rgba(186, 230, 253, 0.95); border-left: 5px solid #005087; border-radius: 16px; padding: 16px 20px; margin-bottom: 20px; box-shadow: 0 4px 16px rgba(0, 80, 135, 0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                <div style="font-weight: 800; color: #005087; font-size: 15px;">
                    Auto-Populated Parameters from Laboratory Report
                </div>
                <span style="background: #e0f2fe; color: #0284c7; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 9999px;">
                    {len(detected_ml_params)} Parameters Ingested
                </span>
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 4px; margin: 10px 0 6px 0;">
                <span class='ms-pill-teal' style='font-size:12px; margin: 3px 4px;'>SEX: <b>{patient_sex.upper()}</b></span>
                {params_badges}
            </div>
            <div style="font-size: 13px; color: #475569;">
                All matching parameters below have been pre-filled directly from your report. You can review, fine-tune, or click Predict on any model.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.get("auto_run_ml"):
        st.success("Laboratory data successfully loaded and pre-filled into all ML models below.")
        st.session_state["auto_run_ml"] = False

    tab_dia, tab_ane, tab_thy = st.tabs(["Diabetes Risk (XGBoost)", "Anemia Evaluation (LightGBM)", "Thyroid Disorder (XGBoost 9-Class)"])

    # 1. Diabetes
    with tab_dia:
        st.markdown("#### Diabetes Risk Classifier (8 Features — Pima Indians Benchmark)")
        if not ml_readiness.get("diabetes"):
            st.error("Diabetes model or scaler is missing from `models/`.")
        else:
            with st.form("form_diabetes"):
                col1, col2 = st.columns(2)
                with col1:
                    is_male = (patient_sex.lower() == "male")
                    preg_default = 0 if is_male else 1
                    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=30, value=preg_default, step=1, disabled=is_male,
                                                  help="Disabled for biological male baseline.")
                    glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=400.0,
                                               value=float(ocr_vals.get("glucose") or 120.0), step=1.0)
                    blood_pressure = st.number_input("Blood Pressure / Diastolic (mm Hg)", min_value=0.0, max_value=250.0,
                                                     value=float(ocr_vals.get("blood_pressure") or 70.0), step=1.0)
                    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=150.0, value=20.0, step=1.0)
                with col2:
                    insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=1000.0,
                                              value=float(ocr_vals.get("insulin") or 80.0), step=1.0)
                    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=100.0,
                                          value=float(ocr_vals.get("bmi") or 25.0), step=0.1)
                    pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.47, step=0.01)
                    age_val = int(st.session_state.get("patient_age") or 33)
                    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=age_val, step=1)

                submitted_dia = st.form_submit_button("🔍 Predict Diabetes Risk", use_container_width=True)

            if submitted_dia:
                res = ml_service.predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age)
                if res.get("success"):
                    color = "#b91c1c" if res["is_elevated_risk"] else "#15803d"
                    st.markdown(f"<div style='font-size: 18px; font-weight: 800; color: {color}; margin: 12px 0;'>Result: {res['label']}</div>", unsafe_allow_html=True)
                    if res.get("confidence") is not None:
                        st.progress(res["confidence"])
                        st.caption(f"Estimated class confidence: {res['confidence']*100:.1f}%")
                    
                    if res.get("feature_breakdown"):
                        top_features = [f"{item['feature']} ({item['value']})" for item in res["feature_breakdown"][:6]]
                        top_contribs = [item["contribution"] for item in res["feature_breakdown"][:6]]
                        fig_dia = visualizer.create_feature_contribution_chart("Diabetes Model", top_features, top_contribs)
                        st.plotly_chart(fig_dia, use_container_width=True)
                else:
                    st.error(f"Prediction failed: {res.get('error')}")

    # 2. Anemia
    with tab_ane:
        st.markdown("#### Anemia Risk Classifier (5 Features — LightGBM CBC Benchmark)")
        if not ml_readiness.get("anemia"):
            st.error("Anemia model or scaler is missing from `models/`.")
        else:
            with st.form("form_anemia"):
                col1, col2 = st.columns(2)
                with col1:
                    gender_idx = 0 if patient_sex.lower() == "female" else 1
                    gender = st.selectbox("Gender", ["Female", "Male"], index=gender_idx)
                    hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=30.0,
                                                  value=float(ocr_vals.get("hemoglobin") or 12.0), step=0.1)
                    mch = st.number_input("MCH (pg)", min_value=0.0, max_value=60.0,
                                          value=float(ocr_vals.get("mch") or 27.0), step=0.1)
                with col2:
                    mchc = st.number_input("MCHC (g/dL)", min_value=0.0, max_value=60.0,
                                           value=float(ocr_vals.get("mchc") or 32.0), step=0.1)
                    mcv = st.number_input("MCV (fL)", min_value=0.0, max_value=150.0,
                                          value=float(ocr_vals.get("mcv") or 85.0), step=0.1)

                submitted_ane = st.form_submit_button("🔍 Predict Anemia Risk", use_container_width=True)

            if submitted_ane:
                res = ml_service.predict_anemia(gender, hemoglobin, mch, mchc, mcv)
                if res.get("success"):
                    color = "#b91c1c" if res["is_elevated_risk"] else "#15803d"
                    st.markdown(f"<div style='font-size: 18px; font-weight: 800; color: {color}; margin: 12px 0;'>Result: {res['label']}</div>", unsafe_allow_html=True)
                    if res.get("confidence") is not None:
                        st.progress(res["confidence"])
                        st.caption(f"Estimated class confidence: {res['confidence']*100:.1f}%")

                    if res.get("feature_breakdown"):
                        top_features = [f"{item['feature']} ({item['value']})" for item in res["feature_breakdown"]]
                        top_contribs = [item["contribution"] for item in res["feature_breakdown"]]
                        fig_ane = visualizer.create_feature_contribution_chart("Anemia LightGBM", top_features, top_contribs)
                        st.plotly_chart(fig_ane, use_container_width=True)
                else:
                    st.error(f"Prediction failed: {res.get('error')}")

    # 3. Thyroid
    with tab_thy:
        st.markdown("#### Thyroid Condition Classifier (21 Features — XGBoost 9-class Model)")
        if not ml_readiness.get("thyroid"):
            st.error("Thyroid model, scaler, or target encoder is missing from `models/`.")
        else:
            with st.form("form_thyroid"):
                st.markdown("##### 1. Demographics & Acute Status")
                c1, c2, c3 = st.columns(3)
                with c1:
                    t_age_val = int(st.session_state.get("patient_age") or 35)
                    t_age = st.number_input("Age", min_value=1, max_value=120, value=t_age_val, step=1)
                with c2:
                    thy_gender_idx = 0 if patient_sex.lower() == "female" else 1
                    t_sex = st.selectbox("Sex", ["Female", "Male"], index=thy_gender_idx, key="thy_sex_input")
                with c3:
                    sick = st.checkbox("Currently Sick")

                st.markdown("##### 2. Medications & Medical History")
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    on_thyroxine = st.checkbox("On Thyroxine")
                with m2:
                    query_on_thyroxine = st.checkbox("Query on Thyroxine")
                with m3:
                    on_antithyroid_med = st.checkbox("On Antithyroid Med")
                with m4:
                    is_male_thy = (patient_sex.lower() == "male")
                    pregnant = st.checkbox("Pregnant", disabled=is_male_thy)

                h1, h2, h3, h4 = st.columns(4)
                with h1:
                    thyroid_surgery = st.checkbox("Thyroid Surgery")
                with h2:
                    i131_treatment = st.checkbox("I-131 Treatment")
                with h3:
                    query_hypo = st.checkbox("Query Hypothyroid")
                with h4:
                    query_hyper = st.checkbox("Query Hyperthyroid")

                c_other1, c_other2, c_other3, c_other4 = st.columns(4)
                with c_other1:
                    lithium = st.checkbox("Lithium")
                with c_other2:
                    goitre = st.checkbox("Goitre")
                with c_other3:
                    tumor = st.checkbox("Tumor")
                with c_other4:
                    hypopituitary = st.checkbox("Hypopituitary")

                psych = st.checkbox("Psychological condition")

                st.markdown("##### 3. Thyroid Laboratory Values")
                v1, v2, v3, v4, v5 = st.columns(5)
                with v1:
                    tsh = st.number_input("TSH (mIU/L)", min_value=0.0, value=float(ocr_vals.get("tsh") or 2.0), step=0.01)
                with v2:
                    t3 = st.number_input("T3 (nmol/L)", min_value=0.0, value=float(ocr_vals.get("t3") or 1.8), step=0.01)
                with v3:
                    tt4 = st.number_input("TT4 (nmol/L)", min_value=0.0, value=float(ocr_vals.get("tt4") or 100.0), step=0.1)
                with v4:
                    t4u = st.number_input("T4U", min_value=0.0, value=float(ocr_vals.get("t4u") or 1.0), step=0.01)
                with v5:
                    fti = st.number_input("FTI", min_value=0.0, value=float(ocr_vals.get("fti") or 100.0), step=0.1)

                submitted_thy = st.form_submit_button("🔍 Predict Thyroid Condition", use_container_width=True)

            if submitted_thy:
                thy_features = {
                    "age": t_age, "sex": t_sex, "sick": sick,
                    "on_thyroxine": on_thyroxine, "query_on_thyroxine": query_on_thyroxine,
                    "on_antithyroid_medication": on_antithyroid_med, "pregnant": pregnant,
                    "thyroid_surgery": thyroid_surgery, "i131_treatment": i131_treatment,
                    "query_hypothyroid": query_hypo, "query_hyperthyroid": query_hyper,
                    "lithium": lithium, "goitre": goitre, "tumor": tumor,
                    "hypopituitary": hypopituitary, "psych": psych,
                    "tsh": tsh, "t3": t3, "tt4": tt4, "t4u": t4u, "fti": fti
                }
                res = ml_service.predict_thyroid(thy_features)
                if res.get("success"):
                    color = "#15803d" if res["is_negative"] else "#b91c1c"
                    st.markdown(f"<div style='font-size: 18px; font-weight: 800; color: {color}; margin: 12px 0;'>Predicted Class: {res['label']}</div>", unsafe_allow_html=True)
                    if res.get("confidence") is not None:
                        st.progress(res["confidence"])
                        st.caption(f"Estimated class confidence: {res['confidence']*100:.1f}%")

                    if res.get("feature_breakdown"):
                        top_features = [f"{item['feature']} ({item['value']})" for item in res["feature_breakdown"][:8]]
                        top_contribs = [item["contribution"] for item in res["feature_breakdown"][:8]]
                        fig_thy = visualizer.create_feature_contribution_chart("Thyroid 9-Class XGBoost", top_features, top_contribs)
                        st.plotly_chart(fig_thy, use_container_width=True)
                else:
                    st.error(f"Prediction failed: {res.get('error')}")


# ============================================================
# VIEW 7: SETTINGS & API CONFIGURATION
# ============================================================
elif st.session_state["active_nav"] == "Settings":
    ui_components.render_header(
        "Platform Settings & Groq API Configuration",
        "Manage credentials, select active AI vision/chat models, and configure local data retention."
    )

    col_cfg, col_stat = st.columns([1.2, 0.8])

    with col_cfg:
        st.markdown("""
        <div class="ms-card">
            <div class="ms-card-header">
                <span class="ms-card-title">🔑 Groq API Credentials</span>
            </div>
            <div style="font-size: 13px; color: #64748b; margin-bottom: 12px;">
                Enter your Groq API Key below. The key is managed securely server-side and never exposed in client bundles or public repositories.
            </div>
        </div>
        """, unsafe_allow_html=True)

        current_key = st.session_state.get("groq_api_key", "")
        entered_key = st.text_input(
            "Groq API Key (GROQ_API_KEY):",
            value=current_key,
            type="password",
            placeholder="gsk_..."
        )

        st.markdown("#### Model Selection")
        selected_vision = st.selectbox(
            "Primary Vision Model (for Image Scans):",
            groq_service.AVAILABLE_VISION_MODELS,
            index=groq_service.AVAILABLE_VISION_MODELS.index(st.session_state["vision_model"]) if st.session_state["vision_model"] in groq_service.AVAILABLE_VISION_MODELS else 0
        )

        selected_chat = st.selectbox(
            "Primary Chat Model (for AI Assistant):",
            groq_service.AVAILABLE_CHAT_MODELS,
            index=groq_service.AVAILABLE_CHAT_MODELS.index(st.session_state["chat_model"]) if st.session_state["chat_model"] in groq_service.AVAILABLE_CHAT_MODELS else 0
        )

        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            st.session_state["groq_api_key"] = entered_key.strip()
            st.session_state["vision_model"] = selected_vision
            st.session_state["chat_model"] = selected_chat
            os.environ["GROQ_API_KEY"] = entered_key.strip()
            os.environ["GROQ_VISION_MODEL"] = selected_vision
            os.environ["GROQ_CHAT_MODEL"] = selected_chat

            # Also persist into local .env if possible
            try:
                env_lines = [
                    f"GROQ_API_KEY={entered_key.strip()}",
                    f"GROQ_VISION_MODEL={selected_vision}",
                    f"GROQ_CHAT_MODEL={selected_chat}"
                ]
                with open(".env", "w", encoding="utf-8") as f:
                    f.write("\n".join(env_lines) + "\n")
            except Exception:
                pass

            st.success("Settings saved successfully!")
            st.rerun()

    with col_stat:
        st.markdown("""
        <div class="ms-card">
            <div class="ms-card-header">
                <span class="ms-card-title">📡 Connectivity & Tests</span>
            </div>
        """, unsafe_allow_html=True)

        if st.button("🧪 Test Groq API Connection", use_container_width=True):
            test_key = groq_service.get_active_groq_key(st.session_state.get("groq_api_key"))
            if not test_key:
                st.error("No API key available to test.")
            else:
                try:
                    from groq import Groq
                    test_client = Groq(api_key=test_key)
                    resp = test_client.chat.completions.create(
                        model=st.session_state["chat_model"],
                        messages=[{"role": "user", "content": "Respond with: PING_OK"}],
                        max_tokens=10
                    )
                    st.success(f"🟢 Connection Successful! Model returned: {resp.choices[0].message.content.strip()}")
                except Exception as exc:
                    st.error(f"🔴 Connection Failed: {str(exc)}")

        st.divider()
        st.markdown("#### Privacy & Local Storage")
        st.caption(f"History File: `{history_service.HISTORY_FILE}`")
        if st.button("Clear Cached Scans", use_container_width=True):
            history_service.clear_all_history()
            st.success("Cached scan records cleared.")

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# VIEW 8: HELP & SAFETY ADVISORY
# ============================================================
elif st.session_state["active_nav"] == "Help & Safety":
    ui_components.render_header(
        "Help, Clinical Guidelines & Safety Architecture",
        "Understanding Medi Scan AI capabilities, limitations, and ethical healthcare AI safeguards."
    )

    st.markdown("""
    <div class="ms-card">
        <h3 style="color: #005087; margin-top: 0;">🛡️ Core Safety Principles</h3>
        <p style="font-size: 14px; color: #334155; line-height: 1.6;">
            Medi Scan AI is engineered as an <b>informational screening and educational assistant</b> designed to facilitate patient-physician dialogue. It is strictly subject to the following clinical and ethical constraints:
        </p>
        <ul style="font-size: 14px; color: #334155; line-height: 1.7;">
            <li><b>Non-Diagnostic Scope:</b> AI outputs represent observational descriptions, NOT verified medical diagnoses. Definitive diagnosis requires physical examination, patient history, and clinician evaluation.</li>
            <li><b>No Medical Prescription:</b> The platform never prescribes medication or dictates dosages.</li>
            <li><b>Emergency Deferral:</b> Emergency symptoms (chest pain, acute dyspnea, focal neurological deficits) must immediately be routed to local emergency clinical services.</li>
            <li><b>Transparent Uncertainty:</b> The system highlights projection limitations, missing lateral views, and technical artifacts rather than pretending to achieve complete diagnostic certainty.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="ms-card">
            <h4 style="margin-top: 0;">📸 Supported Imaging Modalities</h4>
            <ul style="font-size: 13.5px; color: #475569; line-height: 1.6;">
                <li><b>Radiographs (X-rays):</b> Chest, musculoskeletal (extremities, spine, pelvis).</li>
                <li><b>Magnetic Resonance Imaging (MRI):</b> Brain, spine, knee, joint scans (T1, T2, FLAIR).</li>
                <li><b>Computed Tomography (CT):</b> Thorax, abdomen, head cross-sections.</li>
                <li><b>Ultrasound:</b> Abdominal, thyroid, vascular.</li>
                <li><b>Dermoscopy:</b> High-resolution skin lesion photographs.</li>
                <li><b>Laboratory Documents:</b> CBC and metabolic panel PDF/image reports.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="ms-card">
            <h4 style="margin-top: 0;">❓ Frequently Asked Questions</h4>
            <div style="font-size: 13.5px; color: #475569; line-height: 1.6;">
                <b>Q: Is my medical image sent to public web forums?</b><br>
                A: No. Uploaded images are transmitted solely to the Groq API endpoint for analysis and stored strictly on your local machine.<br><br>
                <b>Q: Can I delete my scan history?</b><br>
                A: Yes. You can remove individual scans or clear all records with a single click in the Scan History or Settings tab.
            </div>
        </div>
        """, unsafe_allow_html=True)
