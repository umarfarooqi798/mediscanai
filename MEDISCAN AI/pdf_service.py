"""
MediScan Health — Clinical PDF Report Generator
Generates clean, professional, vectorized clinical PDF reports for laboratory panels
using ReportLab with standard hospital formatting, tables, and typography.
"""

import io
from datetime import datetime
from typing import Dict, Any, List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_lab_pdf_report(
    filename: str,
    patient_sex: str,
    patient_age: Optional[int],
    values: Dict[str, Optional[float]],
    analyses: Dict[str, Dict[str, Any]],
    triage_level: str,
    triage_reasons: List[str],
    health_guidance: Dict[str, Any],
    executive_summary: Optional[str] = None
) -> bytes:
    """
    Synthesizes a high-definition clinical laboratory PDF report.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Clean Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f2942")
    )

    subtitle_style = ParagraphStyle(
        'DocSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#64748b")
    )

    section_style = ParagraphStyle(
        'SectionHead',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#005087"),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1e293b")
    )

    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    summary_box_style = ParagraphStyle(
        'SummaryBox',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#0f172a")
    )

    table_header_style = ParagraphStyle(
        'TableHead',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.white
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell_style,
        fontName='Helvetica-Bold'
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#64748b")
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>MEDISCAN HEALTH</b><br/><font size='8' color='#64748b'>CLINICAL LABORATORY DIAGNOSTIC SUITE</font>", title_style),
            Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%B %d, %Y')}<br/><b>Time:</b> {datetime.now().strftime('%H:%M:%S')}", subtitle_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[360, 180])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#005087"), spaceAfter=10))

    # 2. Patient Demographics & Document Profile
    age_display = f"{patient_age} Years" if patient_age else "Not Specified"
    demo_data = [
        [
            Paragraph("<b>Patient Biological Sex:</b>", table_cell_bold),
            Paragraph(patient_sex.title(), table_cell_style),
            Paragraph("<b>Estimated Patient Age:</b>", table_cell_bold),
            Paragraph(age_display, table_cell_style)
        ],
        [
            Paragraph("<b>Source Document:</b>", table_cell_bold),
            Paragraph(filename or "Uploaded Laboratory Report", table_cell_style),
            Paragraph("<b>Overall Triage Status:</b>", table_cell_bold),
            Paragraph(f"<b>{triage_level.upper()}</b>", table_cell_style)
        ]
    ]
    demo_table = Table(demo_data, colWidths=[120, 150, 120, 150])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(demo_table)
    story.append(Spacer(1, 10))

    # 3. Clinical Executive Summary Box
    if executive_summary:
        story.append(Paragraph("Clinical Overview & Executive Summary", section_style))
        summary_content = [
            [Paragraph(f"<b>Clinical Summary:</b> {executive_summary}", summary_box_style)]
        ]
        sum_table = Table(summary_content, colWidths=[540])
        sum_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f0f7ff")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#bae6fd")),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(sum_table)
        story.append(Spacer(1, 10))

    # 4. Detected Biomarkers Table
    story.append(Paragraph("Evaluated Biomarkers & Reference Interval Analysis", section_style))

    table_rows = [
        [
            Paragraph("Biomarker Name", table_header_style),
            Paragraph("Category", table_header_style),
            Paragraph("Observed Value", table_header_style),
            Paragraph("Reference Range", table_header_style),
            Paragraph("Clinical Status", table_header_style)
        ]
    ]

    row_count = 0
    for name, res in analyses.items():
        val = res.get("value")
        if val is not None:
            row_count += 1
            status = res.get("status", "Normal")
            status_color = "#15803d" if status == "Normal" else "#b91c1c"
            status_cell = f"<font color='{status_color}'><b>{status.upper()}</b></font>"

            table_rows.append([
                Paragraph(res.get("name", name.title()), table_cell_bold),
                Paragraph(res.get("category", "General"), table_cell_style),
                Paragraph(f"{val} {res.get('unit', '')}", table_cell_style),
                Paragraph(f"{res.get('reference_low', '')} - {res.get('reference_high', '')} {res.get('unit', '')}", table_cell_style),
                Paragraph(status_cell, table_cell_style)
            ])

    if row_count == 0:
        table_rows.append([
            Paragraph("No numerical biomarkers detected in uploaded document.", table_cell_style),
            Paragraph("-", table_cell_style),
            Paragraph("-", table_cell_style),
            Paragraph("-", table_cell_style),
            Paragraph("-", table_cell_style)
        ])

    bio_table = Table(table_rows, colWidths=[150, 80, 105, 120, 85])
    bio_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#005087")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(bio_table)
    story.append(Spacer(1, 12))

    # 5. Screening Observations & Triage Reasons
    if triage_reasons:
        story.append(Paragraph("Clinical Observations & Flagged Findings", section_style))
        for reason in triage_reasons:
            story.append(Paragraph(f"• {reason}", body_style))
            story.append(Spacer(1, 3))
        story.append(Spacer(1, 8))

    # 6. Actionable Health Protocol (What to Do, Avoid, Ignore)
    what_to_do = health_guidance.get("what_to_do", [])
    what_to_avoid = health_guidance.get("what_to_avoid", [])
    what_to_ignore = health_guidance.get("what_to_ignore", health_guidance.get("what_to_ignore_and_myths", []))

    if what_to_do or what_to_avoid or what_to_ignore:
        story.append(Paragraph("Evidence-Based Health Protocol", section_style))

        # Do's
        if what_to_do:
            do_text = "<br/>".join([f"• {item}" for item in what_to_do[:4]])
            do_content = [
                [Paragraph("<b>RECOMMENDED PROACTIVE ACTIONS (WHAT TO DO):</b><br/>" + do_text, summary_box_style)]
            ]
            do_table = Table(do_content, colWidths=[540])
            do_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#ecfdf5")),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#a7f3d0")),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(do_table)
            story.append(Spacer(1, 6))

        # Avoid's
        if what_to_avoid:
            avoid_text = "<br/>".join([f"• {item}" for item in what_to_avoid[:4]])
            avoid_content = [
                [Paragraph("<b>CRITICAL PRECAUTIONS (WHAT TO AVOID):</b><br/>" + avoid_text, summary_box_style)]
            ]
            avoid_table = Table(avoid_content, colWidths=[540])
            avoid_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#fecaca")),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(avoid_table)
            story.append(Spacer(1, 6))

        # Ignore's
        if what_to_ignore:
            ignore_text = "<br/>".join([f"• {item}" for item in what_to_ignore[:3]])
            ignore_content = [
                [Paragraph("<b>DEBUNKED MISCONCEPTIONS (WHAT TO IGNORE):</b><br/>" + ignore_text, summary_box_style)]
            ]
            ignore_table = Table(ignore_content, colWidths=[540])
            ignore_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#fffbeb")),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#fde68a")),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(ignore_table)
            story.append(Spacer(1, 10))

    # 7. Disclaimer
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=8))
    story.append(Paragraph(
        "<b>CLINICAL SAFETY NOTICE:</b> This laboratory report analysis was generated for clinical screening, "
        "informational, and educational reference only. It does not replace diagnostic clinical review by a licensed physician "
        "or pathologist. Always discuss laboratory findings directly with your healthcare provider for formal diagnosis and treatment plans.",
        disclaimer_style
    ))

    doc.build(story)
    return buffer.getvalue()
