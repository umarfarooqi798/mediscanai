"""
Mediscan AI — Interactive Laboratory Visualization Engine
Builds Plotly charts, biomarker gauge meters, reference comparison charts,
and model explainability contribution plots.
"""

import plotly.graph_objects as go
from typing import Dict, Any, Optional, List, Tuple

BIOMARKER_METADATA = {
    "hemoglobin": {"name": "Hemoglobin", "unit": "g/dL", "min": 6.0, "max": 22.0},
    "glucose": {"name": "Blood Glucose", "unit": "mg/dL", "min": 40.0, "max": 300.0},
    "hba1c": {"name": "HbA1c", "unit": "%", "min": 3.0, "max": 14.0},
    "cholesterol": {"name": "Total Cholesterol", "unit": "mg/dL", "min": 80.0, "max": 350.0},
    "hdl": {"name": "HDL Cholesterol", "unit": "mg/dL", "min": 20.0, "max": 100.0},
    "ldl": {"name": "LDL Cholesterol", "unit": "mg/dL", "min": 40.0, "max": 250.0},
    "triglycerides": {"name": "Triglycerides", "unit": "mg/dL", "min": 40.0, "max": 500.0},
    "wbc": {"name": "White Blood Cells", "unit": "x10³/µL", "min": 1.0, "max": 25.0},
    "platelets": {"name": "Platelets", "unit": "x10³/µL", "min": 50.0, "max": 600.0},
    "rbc": {"name": "Red Blood Cells", "unit": "x10⁶/µL", "min": 2.0, "max": 7.5},
    "hematocrit": {"name": "Hematocrit (HCT)", "unit": "%", "min": 20.0, "max": 65.0},
    "mcv": {"name": "MCV", "unit": "fL", "min": 50.0, "max": 130.0},
    "mch": {"name": "MCH", "unit": "pg", "min": 15.0, "max": 45.0},
    "mchc": {"name": "MCHC", "unit": "g/dL", "min": 20.0, "max": 42.0},
    "creatinine": {"name": "Serum Creatinine", "unit": "mg/dL", "min": 0.2, "max": 6.0},
    "urea_bun": {"name": "BUN / Urea", "unit": "mg/dL", "min": 3.0, "max": 80.0},
    "tsh": {"name": "TSH", "unit": "mIU/L", "min": 0.05, "max": 15.0},
    "calcium": {"name": "Calcium", "unit": "mg/dL", "min": 5.0, "max": 15.0},
    "potassium": {"name": "Potassium", "unit": "mmol/L", "min": 2.5, "max": 7.5},
    "sodium": {"name": "Sodium", "unit": "mmol/L", "min": 115.0, "max": 165.0}
}


def create_biomarker_gauge(
    test_key: str,
    value: float,
    reference_low: float,
    reference_high: float,
    unit: str,
    title: Optional[str] = None
) -> go.Figure:
    """
    Creates an intuitive interactive gauge with color-coded Low, Normal, and High reference ranges.
    """
    meta = BIOMARKER_METADATA.get(test_key, {"name": test_key.title(), "unit": unit, "min": 0, "max": 100})
    display_title = title or meta["name"]
    
    # Calculate sensible axis bounds
    axis_min = min(meta.get("min", reference_low * 0.5), reference_low * 0.7, value * 0.8)
    axis_max = max(meta.get("max", reference_high * 1.5), reference_high * 1.3, value * 1.15)
    
    # Determine value status color
    if value < reference_low:
        bar_color = "#0284c7"  # Low blue
    elif value > reference_high:
        bar_color = "#e11d48"  # High red
    else:
        bar_color = "#059669"  # Normal emerald

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": f" {unit}", "font": {"family": "Plus Jakarta Sans, sans-serif", "size": 22, "color": "#0f172a"}},
        title={"text": f"<b>{display_title}</b>", "font": {"family": "Plus Jakarta Sans, sans-serif", "size": 15, "color": "#005087"}},
        gauge={
            "axis": {
                "range": [axis_min, axis_max],
                "tickwidth": 1,
                "tickcolor": "#94a3b8",
                "tickfont": {"size": 10, "color": "#64748b"}
            },
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "rgba(240, 249, 255, 0.6)",
            "borderwidth": 1,
            "bordercolor": "rgba(186, 230, 253, 0.8)",
            "steps": [
                {"range": [axis_min, reference_low], "color": "rgba(224, 242, 254, 0.85)"},  # Low zone
                {"range": [reference_low, reference_high], "color": "rgba(209, 250, 229, 0.85)"},  # Normal zone
                {"range": [reference_high, axis_max], "color": "rgba(254, 226, 226, 0.85)"}   # High zone
            ],
            "threshold": {
                "line": {"color": "#005087", "width": 4},
                "thickness": 0.75,
                "value": value
            }
        }
    ))

    fig.update_layout(
        height=170,
        margin=dict(l=22, r=22, t=38, b=12),
        paper_bgcolor="rgba(255, 255, 255, 0)",
        plot_bgcolor="rgba(255, 255, 255, 0)",
        font=dict(family="Plus Jakarta Sans, sans-serif")
    )
    return fig


def create_multi_biomarker_deviation_chart(
    biomarkers: List[Dict[str, Any]]
) -> Optional[go.Figure]:
    """
    Renders a grouped diagnostic deviation bar chart comparing all detected biomarkers
    as a percentage deviation from the normal reference interval midpoint.
    0% = exact normal midpoint. Negative = below reference. Positive = above reference.
    """
    if not biomarkers:
        return None

    labels = []
    deviations = []
    colors = []
    hover_texts = []

    for item in biomarkers:
        val = item.get("value")
        ref_low = item.get("reference_low")
        ref_high = item.get("reference_high")
        unit = item.get("unit", "")
        name = item.get("name", "")

        if val is None or ref_low is None or ref_high is None:
            continue

        midpoint = (ref_low + ref_high) / 2.0
        span = ref_high - ref_low
        if span == 0:
            span = 1.0

        # Percent deviation from midpoint normalized to half-span
        # -100% is ref_low, +100% is ref_high
        norm_dev = ((val - midpoint) / (span / 2.0)) * 100.0
        
        labels.append(name)
        deviations.append(round(norm_dev, 1))

        if norm_dev < -100.0:
            colors.append("#0284c7")  # Below normal
            status = "Low"
        elif norm_dev > 100.0:
            colors.append("#e11d48")  # Above normal
            status = "Elevated"
        else:
            colors.append("#10b981")  # Within normal
            status = "Normal"

        hover_texts.append(
            f"<b>{name}</b><br>Value: {val} {unit}<br>Reference: {ref_low} – {ref_high} {unit}<br>Status: {status}"
        )

    if not labels:
        return None

    fig = go.Figure()

    # Normal baseline band
    fig.add_hrect(
        y0=-100, y1=100,
        fillcolor="rgba(16, 185, 129, 0.12)",
        line_width=0,
        annotation_text="Normal Reference Baseline (-100% to +100%)",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#059669")
    )

    fig.add_trace(go.Bar(
        x=labels,
        y=deviations,
        marker=dict(
            color=colors,
            line=dict(color="rgba(255, 255, 255, 0.9)", width=1.5)
        ),
        text=[f"{d:+.0f}%" for d in deviations],
        textposition="outside",
        hovertext=hover_texts,
        hoverinfo="text"
    ))

    fig.update_layout(
        title={
            "text": "<b>Multi-Biomarker Reference Interval Deviation Summary</b>",
            "font": {"family": "Plus Jakarta Sans, sans-serif", "size": 16, "color": "#005087"}
        },
        yaxis=dict(
            title="Deviation from Reference Midpoint (%)",
            zeroline=True,
            zerolinecolor="#94a3b8",
            zerolinewidth=1.5,
            gridcolor="rgba(226, 232, 240, 0.6)"
        ),
        xaxis=dict(
            tickangle=-25,
            tickfont=dict(size=12, color="#1e293b")
        ),
        paper_bgcolor="rgba(255, 255, 255, 0.7)",
        plot_bgcolor="rgba(240, 249, 255, 0.5)",
        height=320,
        margin=dict(l=40, r=30, t=50, b=70)
    )

    return fig


def create_feature_contribution_chart(
    title: str,
    features: List[str],
    contributions: List[float]
) -> go.Figure:
    """
    Renders an explainability horizontal bar chart showing top contributing clinical features.
    """
    colors = ["#e11d48" if c > 0 else "#059669" for c in contributions]

    fig = go.Figure(go.Bar(
        x=contributions,
        y=features,
        orientation='h',
        marker=dict(color=colors, line=dict(color="rgba(255, 255, 255, 0.8)", width=1)),
        text=[f"{c:+.1f}%" for c in contributions],
        textposition="auto"
    ))

    fig.update_layout(
        title={"text": f"<b>{title} — Key Factor Explainability</b>", "font": {"size": 14, "color": "#005087"}},
        xaxis=dict(title="Relative Risk Factor Weight (%)", gridcolor="rgba(226, 232, 240, 0.6)"),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(255, 255, 255, 0)",
        plot_bgcolor="rgba(240, 249, 255, 0.4)",
        height=240,
        margin=dict(l=10, r=20, t=40, b=30)
    )
    return fig
