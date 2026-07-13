import io
import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Helper to map categories to colors
AQI_COLORS = {
    "Good": colors.HexColor("#00b050"),
    "Satisfactory": colors.HexColor("#92d050"),
    "Moderate": colors.HexColor("#ffff00"),
    "Poor": colors.HexColor("#ff9900"),
    "Very Poor": colors.HexColor("#ff0000"),
    "Severe": colors.HexColor("#7030a0")
}

AQI_HEX = {
    "Good": "#00b050",
    "Satisfactory": "#92d050",
    "Moderate": "#e5c100", # slightly darker yellow for readability
    "Poor": "#ff9900",
    "Very Poor": "#ff0000",
    "Severe": "#7030a0"
}

def get_health_advisory(category):
    """
    Returns warning messages and actionable recommendations for each AQI Category.
    """
    advisories = {
        "Good": {
            "warning": "No health impacts are expected. Air quality is ideal for all outdoor activities.",
            "recommendations": [
                "Enjoy standard outdoor activities.",
                "Open windows to ventilate indoor spaces.",
                "Ideal air quality for sports, running, and outdoor play."
            ]
        },
        "Satisfactory": {
            "warning": "Air quality is acceptable. However, highly sensitive individuals may experience minor breathing discomfort.",
            "recommendations": [
                "Standard outdoor activities are safe.",
                "Sensitive individuals should monitor for cough or shortness of breath.",
                "Ventilation is safe for homes and offices."
            ]
        },
        "Moderate": {
            "warning": "May cause breathing discomfort for children, elderly, and individuals with lung/heart diseases.",
            "recommendations": [
                "Limit prolonged or heavy outdoor exertion.",
                "Take more breaks during outdoor activities.",
                "Sensitive groups should consider indoor alternatives."
            ]
        },
        "Poor": {
            "warning": "May cause breathing discomfort to most people on prolonged exposure. Heart/lung patients may experience aggravated symptoms.",
            "recommendations": [
                "Reduce prolonged or heavy outdoor activities.",
                "Wear N95 masks when stepping outdoors.",
                "Sensitive groups should stay indoors.",
                "Keep doors and windows closed to reduce indoor pollution infiltration."
            ]
        },
        "Very Poor": {
            "warning": "May cause respiratory illness to the people on prolonged exposure. Significant health impacts for people with heart/lung disease.",
            "recommendations": [
                "Avoid prolonged outdoor exertion; limit outdoor activities strictly.",
                "Wear an N95 mask mandatory for all outdoor commutes.",
                "Sensitive individuals must remain indoors.",
                "Run air purifiers indoors if available."
            ]
        },
        "Severe": {
            "warning": "Affects healthy people and seriously impacts those with existing diseases. Emergency atmospheric conditions.",
            "recommendations": [
                "Avoid all outdoor physical activity. Stay indoors.",
                "Keep windows closed completely and run air purifiers.",
                "Wear double masks (N95) if outdoor transit is absolutely unavoidable.",
                "Monitor oxygen levels and contact medical support if breathing is labored."
            ]
        }
    }
    return advisories.get(category, advisories["Moderate"])

def generate_pdf_report(prediction_data, target_date):
    """
    Generates a professional, print-ready PDF report of the AQI prediction.
    Returns a BytesIO stream containing the PDF data.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom typography styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=colors.HexColor('#0F172A'), # slate-900
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        textColor=colors.HexColor('#475569'), # slate-600
        spaceAfter=15
    )
    
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=15,
        spaceAfter=10,
        borderPadding=2
    )

    body_style = ParagraphStyle(
        'BodyText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#334155'),
        leading=14
    )
    
    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=colors.HexColor('#0284C7'), # light blue
        spaceAfter=10
    )

    # 1. Header Section
    story.append(Paragraph("AQIFusionNet Forecast Report", title_style))
    story.append(Paragraph("CNN-LSTM-GRU + XGBoost Ensemble short-term forecasting for Delhi NCR", subtitle_style))
    
    # Decorative line
    divider = Table([[""]], colWidths=[532], rowHeights=[2])
    divider.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#3B82F6')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(divider)
    story.append(Spacer(1, 15))
    
    # 2. Key Metadata Table
    date_str = target_date.strftime("%B %d, %Y")
    days_fwd = prediction_data['days_forward']
    aqi_val = prediction_data['aqi']
    category = prediction_data['category']
    category_color = AQI_COLORS.get(category, colors.black)
    
    meta_data = [
        [Paragraph("<b>Target Forecast Date:</b>", body_style), Paragraph(date_str, body_style)],
        [Paragraph("<b>Days Forward (Post-Training):</b>", body_style), Paragraph(f"{days_fwd} days", body_style)],
        [Paragraph("<b>Forecast Generated On:</b>", body_style), Paragraph(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), body_style)]
    ]
    meta_table = Table(meta_data, colWidths=[200, 332])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))
    
    # 3. Main AQI Result Box (Highlight Card)
    aqi_para = Paragraph(f"<font size='36'><b>{aqi_val}</b></font><br/><font size='12'>Predicted Future AQI</font>", ParagraphStyle('AqiNum', parent=body_style, alignment=1, textColor=colors.HexColor('#0F172A')))
    cat_para = Paragraph(f"<font size='18'><b>{category.upper()}</b></font><br/><font size='10'>Air Quality Category</font>", ParagraphStyle('AqiCat', parent=body_style, alignment=1, textColor=colors.white))
    
    aqi_card = Table([[aqi_para, cat_para]], colWidths=[266, 266], rowHeights=[70])
    aqi_card.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor('#F8FAFC')),
        ('BACKGROUND', (1,0), (1,0), category_color),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0')),
    ]))
    story.append(aqi_card)
    story.append(Spacer(1, 20))
    
    # 4. Pollutants breakdown table
    story.append(Paragraph("Estimated Pollutant Breakdown", section_title_style))
    
    pollutants_list = [
        ("PM2.5", prediction_data.get("PM2.5", 0.0), "ug/m3", "Fine particulate matter associated with respiratory irritation."),
        ("PM10", prediction_data.get("PM10", 0.0), "ug/m3", "Coarse particulate matter depositing deep in lungs."),
        ("NO", prediction_data.get("NO", 0.0), "ug/m3", "Nitric Oxide, precursor to ozone and smog formations."),
        ("NO2", prediction_data.get("NO2", 0.0), "ug/m3", "Nitrogen Dioxide, key indicator of vehicular traffic emissions."),
        ("NOx", prediction_data.get("NOx", 0.0), "ug/m3", "Nitrogen Oxides, general indicator of heavy fuel combustion."),
        ("NH3", prediction_data.get("NH3", 0.0), "ug/m3", "Ammonia, highly linked to agricultural and waste output."),
        ("CO", prediction_data.get("CO", 0.0), "mg/m3", "Carbon Monoxide, toxic gas reducing oxygen carriage capacity."),
        ("SO2", prediction_data.get("SO2", 0.0), "ug/m3", "Sulfur Dioxide, indicator of industrial refinery emissions."),
        ("O3", prediction_data.get("O3", 0.0), "ug/m3", "Ozone, highly reactive ground-level secondary pollutant.")
    ]
    
    # Build Table Header
    pollutant_table_data = [[
        Paragraph("<b>Pollutant</b>", ParagraphStyle('Th', parent=body_style, textColor=colors.white)),
        Paragraph("<b>Value</b>", ParagraphStyle('Th', parent=body_style, textColor=colors.white)),
        Paragraph("<b>Unit</b>", ParagraphStyle('Th', parent=body_style, textColor=colors.white)),
        Paragraph("<b>Environmental Context</b>", ParagraphStyle('Th', parent=body_style, textColor=colors.white))
    ]]
    
    for name, val, unit, desc in pollutants_list:
        pollutant_table_data.append([
            Paragraph(f"<b>{name}</b>", body_style),
            Paragraph(f"{val:.2f}", body_style),
            Paragraph(unit, body_style),
            Paragraph(desc, ParagraphStyle('Desc', parent=body_style, fontSize=9, textColor=colors.HexColor('#64748B')))
        ])
        
    poll_table = Table(pollutant_table_data, colWidths=[70, 70, 60, 332])
    poll_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')])
    ]))
    story.append(poll_table)
    story.append(Spacer(1, 20))
    
    # 5. Health Advisory Section
    story.append(Paragraph("Health Advisory & Guidelines", section_title_style))
    advisory = get_health_advisory(category)
    warning_text = f"<b>AQI Category Rating Warning:</b> {advisory['warning']}"
    story.append(Paragraph(warning_text, body_style))
    story.append(Spacer(1, 8))
    
    recommendations_html = "".join([f"<li>{rec}</li>" for rec in advisory['recommendations']])
    story.append(Paragraph(f"<ul>{recommendations_html}</ul>", ParagraphStyle('Bullet', parent=body_style, leftIndent=15, spaceBefore=4)))
    story.append(Spacer(1, 20))
    
    # 6. Model Ensemble Summary
    story.append(Paragraph("About AQIFusionNet Model Architecture", section_title_style))
    model_desc = """
    <b>AQIFusionNet</b> is a state-of-the-art hybrid deep learning and machine learning ensemble framework specifically optimized for short-term Air Quality Index forecasting.
    It combines multi-stage layers:
    <ul>
        <li><b>CNN (1D Convolutional Neural Network):</b> Extracts spatial and cross-variable correlations from raw input features.</li>
        <li><b>LSTM (Long Short-Term Memory):</b> Captures long-term temporal dependencies and historical lags.</li>
        <li><b>GRU (Gated Recurrent Unit):</b> Optimizes spatial-temporal representations with fewer parameters to capture short-term transitions.</li>
        <li><b>XGBoost (Extreme Gradient Boosting):</b> Acts as a robust tabular refiner, taking outputs from CNN-LSTM-GRU layers to output a highly accurate AQI prediction.</li>
    </ul>
    """
    story.append(Paragraph(model_desc, ParagraphStyle('ModelText', parent=body_style, leading=14)))
    story.append(Spacer(1, 25))
    
    # Footer metadata
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        textColor=colors.HexColor('#94A3B8'),
        alignment=1
    )
    story.append(Paragraph("AQIFusionNet • Hybrid CNN-LSTM-GRU + XGBoost Ensemble • Delhi Air Quality Project", footer_style))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def generate_csv_report(prediction_data, target_date):
    """
    Generates a CSV string representing the prediction results.
    """
    data = {
        "Metric/Pollutant": [
            "Forecast Date",
            "Days Forward Beyond Training",
            "Predicted AQI",
            "AQI Category",
            "PM2.5 (ug/m3)",
            "PM10 (ug/m3)",
            "NO (ug/m3)",
            "NO2 (ug/m3)",
            "NOx (ug/m3)",
            "NH3 (ug/m3)",
            "CO (mg/m3)",
            "SO2 (ug/m3)",
            "O3 (ug/m3)"
        ],
        "Value": [
            target_date.strftime("%Y-%m-%d"),
            prediction_data["days_forward"],
            prediction_data["aqi"],
            prediction_data["category"],
            prediction_data["PM2.5"],
            prediction_data["PM10"],
            prediction_data["NO"],
            prediction_data["NO2"],
            prediction_data["NOx"],
            prediction_data["NH3"],
            prediction_data["CO"],
            prediction_data["SO2"],
            prediction_data["O3"]
        ]
    }
    df = pd.DataFrame(data)
    return df.to_csv(index=False)
