import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_mock_pdf():
    pdf_path = "/Users/amananand/Downloads/financial-analyst/data/raw/kpmg_mock_annual_report_fy2023.pdf"
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
                            rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    story = []
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#00338D'),
        spaceAfter=15
    )
    
    section_style = ParagraphStyle(
        'DocSection',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#00A3A6'),
        spaceBefore=15,
        spaceAfter=10
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=8
    )
    
    # ── Page 1: Title & Overview ──
    story.append(Paragraph("KPMG Mock Financial Annual Report", title_style))
    story.append(Paragraph("Financial Year ended March 31, 2023 (FY2023)", body_style))
    story.append(Spacer(1, 20))
    
    story.append(Paragraph("1. Executive & Business Overview", section_style))
    story.append(Paragraph(
        "FinDoc Intelligence mock enterprise operates in global digital services, providing "
        "cloud-native integration pipelines and advanced retrieval models. In FY2023, the firm witnessed "
        "robust customer adoption driven by the acceleration of enterprise AI intelligence frameworks.",
        body_style
    ))
    story.append(Paragraph(
        "Our operations are split into three geographic divisions: North America, Europe, and Asia-Pacific. "
        "North America remains our primary market segment, contributing to over 50% of the overall revenue.",
        body_style
    ))
    story.append(Spacer(1, 30))
    
    # ── Page 2: Financial Performance ──
    story.append(Paragraph("2. Financial Statements & Performance", section_style))
    story.append(Paragraph(
        "In FY2023, the consolidated total revenue from operations stood at ₹2,40,000 Crore, compared to "
        "₹2,10,000 Crore in the previous fiscal year, demonstrating a growth of 14.28%.",
        body_style
    ))
    
    # Income Statement Mock Table
    data = [
        ['Metric Detail', 'FY2023 (₹ in Crore)', 'FY2022 (₹ in Crore)'],
        ['Consolidated Revenue from Operations', '2,40,000', '2,10,000'],
        ['Operating Expenses', '1,80,000', '1,62,000'],
        ['Operating Profit (EBITDA)', '60,000', '48,000'],
        ['Net Profit for the Year', '45,000', '36,000'],
        ['Earnings per Share (EPS) in ₹', '150.00', '120.00']
    ]
    
    table = Table(data, colWidths=[240, 130, 130])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#00338D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 8),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#1E293B')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('BOTTOMPADDING', (0,1), (-1,-1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    
    # ── Page 3: Key Risk Factors ──
    story.append(Paragraph("3. Risk Management & Key Risk Factors", section_style))
    story.append(Paragraph(
        "Our operations are subject to multiple industry risk factors. Key risk categories include:",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Geopolitical tensions:</b> Tensions in Eastern Europe and APAC markets could affect global hardware supply chains and inflate energy prices.",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Regulatory Changes:</b> Stringent data residency rules in Europe might increase compliance cost of Cloud operations.",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Currency Fluctuations:</b> Rapid shifts in currency valuation between USD and INR present cash conversion hazards.",
        body_style
    ))
    
    doc.build(story)
    print(f"Mock PDF successfully generated at: {pdf_path}")

if __name__ == "__main__":
    generate_mock_pdf()
