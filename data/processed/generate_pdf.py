import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
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
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#00338D'),
        spaceAfter=15
    )
    
    section_style = ParagraphStyle(
        'DocSection',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#00A3A6'),
        spaceBefore=12,
        spaceAfter=8
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=6
    )
    
    footnote_style = ParagraphStyle(
        'DocFootnote',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor('#475569'),
        spaceAfter=4
    )

    # ── Page 1: Cover & Executive Overview ──
    story.append(Paragraph("FinDoc Intelligence Ltd.", title_style))
    story.append(Paragraph("Consolidated Annual Report & Financial Statements — FY2023", body_style))
    story.append(Paragraph("For the Financial Year ended March 31, 2023", body_style))
    story.append(Spacer(1, 15))
    
    story.append(Paragraph("1. Executive & Business Overview", section_style))
    story.append(Paragraph(
        "FinDoc Intelligence Ltd. operates as a global leader in enterprise digital services, providing "
        "cloud-native integration pipelines, cybersecurity consulting, and advanced AI retrieval models. "
        "During FY2023, the firm witnessed robust customer adoption driven by the acceleration of enterprise "
        "AI intelligence frameworks and compliance transformation projects.",
        body_style
    ))
    story.append(Paragraph(
        "Our operations are split into three geographic divisions: North America, Europe, and Asia-Pacific. "
        "North America remains our primary market segment, contributing to over 50% of the overall revenue. "
        "The company's total employee strength stood at 45,200 as of March 31, 2023, with an annual turnover "
        "rate of 12.5% indicating strong talent retention compared to industry benchmarks.",
        body_style
    ))
    story.append(Paragraph(
        "For the medium term, management has targeted a Return on Equity (ROE) of 20.0% to 22.0% as part of "
        "our capital allocation strategy, focused on margin improvement in cloud services and high-margin consulting.",
        body_style
    ))
    story.append(PageBreak())
    
    # ── Page 2: Consolidated Income Statement ──
    story.append(Paragraph("2. Financial Performance — Income Statement", section_style))
    story.append(Paragraph(
        "In FY2023, the consolidated total revenue from operations stood at ₹2,40,000 Crore, compared to "
        "₹2,10,000 Crore in the previous fiscal year, demonstrating a growth of 14.28%. "
        "Operating Expenses were ₹1,80,000 Crore (compared to ₹1,62,000 Crore in FY2022). "
        "Our EBITDA grew to ₹74,000 Crore (up from ₹48,000 Crore in FY2022), resulting in an EBITDA margin of 30.83%.",
        body_style
    ))
    
    # Income Statement Table
    is_data = [
        ['Income Statement Item', 'FY2023 (₹ in Crore)', 'FY2022 (₹ in Crore)'],
        ['Consolidated Revenue from Operations', '2,40,000', '2,10,000'],
        ['Operating Expenses (excluding D&A)', '1,66,000', '1,62,000'],
        ['Operating Profit (EBITDA)', '74,000', '48,000'],
        ['Depreciation & Amortization (D&A)', '10,000', '8,000'],
        ['Earnings Before Interest & Tax (EBIT)', '64,000', '40,000'],
        ['Interest Expense', '4,000', '5,000'],
        ['Profit Before Tax (PBT)', '60,000', '35,000'],
        ['Tax Expense (at 25.0% rate)', '15,000', '8,750'],
        ['Net Profit for the Year (PAT)', '45,000', '26,250'],
        ['Earnings per Share (EPS) in ₹', '150.00', '87.50']
    ]
    
    t_is = Table(is_data, colWidths=[240, 130, 130])
    t_is.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#00338D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#1E293B')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('BOTTOMPADDING', (0,1), (-1,-1), 4),
    ]))
    story.append(t_is)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Note: Net profit margin for FY2023 was 18.75%, up from 12.5% in FY2022.", footnote_style))
    story.append(PageBreak())
    
    # ── Page 3: Balance Sheet ──
    story.append(Paragraph("3. Financial Position — Balance Sheet", section_style))
    story.append(Paragraph(
        "As of March 31, 2023, the total assets of the company reached ₹1,95,000 Crore, supported by strong reserves "
        "accumulation and working capital. The company maintains an optimal debt-to-equity ratio.",
        body_style
    ))
    
    # Balance Sheet Table
    bs_data = [
        ['Balance Sheet Line Item', 'As of March 31, 2023 (₹ in Cr)', 'As of March 31, 2022 (₹ in Cr)'],
        ['Non-Current Assets (inc. Goodwill & R&D)', '1,20,000', '1,05,000'],
        ['Current Assets (Trade receivables, Inventory)', '75,000', '65,000'],
        ['-- Of which: Cash and Cash Equivalents', '12,500', '7,500'],
        ['Total Assets', '1,95,000', '1,70,000'],
        ['Share Capital (Paid-up)', '3,000', '3,000'],
        ['Other Equity (Reserves & Surplus)', '1,07,000', '92,000'],
        ['Total Shareholders\' Equity', '1,10,000', '95,000'],
        ['Non-Current Liabilities', '45,000', '40,000'],
        ['-- Of which: Long-Term Borrowings (Debt)', '18,000', '20,000'],
        ['Current Liabilities (Payables, Short-term debt)', '40,000', '35,000'],
        ['Total Equity and Liabilities', '1,95,000', '1,70,000']
    ]
    
    t_bs = Table(bs_data, colWidths=[240, 130, 130])
    t_bs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#00338D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#1E293B')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('BOTTOMPADDING', (0,1), (-1,-1), 4),
    ]))
    story.append(t_bs)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Note: Management conducted annual impairment test for Goodwill and found no impairment loss. "
                           "Total research and development (R&D) capitalized stood at ₹8,500 Crore.", footnote_style))
    story.append(PageBreak())
    
    # ── Page 4: Cash Flow Statement ──
    story.append(Paragraph("4. Cash Flow Statement", section_style))
    story.append(Paragraph(
        "Our operations generated significant cash flow, allowing us to fund capital expenditure internally "
        "and reward shareholders through dividends.",
        body_style
    ))
    
    # Cash Flow Table
    cf_data = [
        ['Cash Flow Activities', 'FY2023 (₹ in Crore)', 'FY2022 (₹ in Crore)'],
        ['Net Cash Flow from Operating Activities', '55,000', '32,000'],
        ['Net Cash used in Investing Activities', '-18,000', '-22,000'],
        ['-- Of which: Capital Expenditure (Capex)', '-15,000', '-18,000'],
        ['Net Cash used in Financing Activities', '-32,000', '-8,000'],
        ['-- Of which: Repayment of Borrowings', '-12,000', '-5,000'],
        ['-- Of which: Dividends Paid', '-10,000', '-3,000'],
        ['Net Increase in Cash and Cash Equivalents', '5,000', '2,000'],
        ['Cash and Cash Equivalents at Beginning of Year', '7,500', '5,500'],
        ['Cash and Cash Equivalents at End of Year', '12,500', '7,500']
    ]
    
    t_cf = Table(cf_data, colWidths=[240, 130, 130])
    t_cf.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#00338D')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F8FAFC')),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#1E293B')),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 8.5),
        ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('BOTTOMPADDING', (0,1), (-1,-1), 4),
    ]))
    story.append(t_cf)
    story.append(Spacer(1, 10))
    story.append(Paragraph("Note: Dividend declared per share was ₹10.00 per equity share, leading to a dividend payout ratio of 22.22% "
                           "against the FY2023 net profit.", footnote_style))
    story.append(PageBreak())
    
    # ── Page 5: Notes & Footnotes (Audit Target) ──
    story.append(Paragraph("5. Notes & Footnotes to the Accounts", section_style))
    story.append(Paragraph(
        "This section details specific transactions and balances required for statutory compliance and audit transparency.",
        body_style
    ))
    
    story.append(Paragraph("Note 5.1: Statutory Auditor Fees & Non-Audit Services", section_style))
    story.append(Paragraph(
        "During the year, the statutory auditors, B.K. Ram & Co. LLP, were paid a total statutory audit fee of "
        "₹4.5 Crore. Additionally, fees paid to B.K. Ram & Co. LLP for non-audit tax advisory services stood at "
        "₹1.2 Crore, which represents 26.6% of the statutory audit fee.",
        body_style
    ))
    
    story.append(Paragraph("Note 5.2: Related Party Transactions", section_style))
    story.append(Paragraph(
        "The company entered into transactions with related parties in the normal course of business. Outstanding advances "
        "provided to the wholly-owned subsidiary, FinDoc Cloud Inc. (USA), for working capital support stood at "
        "₹5,000 Crore as of March 31, 2023. These transactions are executed at arm's length. The audit committee reviewed "
        "these transactions and concluded they were at arm's length and approved by the board.",
        body_style
    ))
    
    story.append(Paragraph("Note 5.3: Lease Liabilities (Ind AS 116)", section_style))
    story.append(Paragraph(
        "Under Ind AS 116, the company has recognized Right-of-Use (ROU) assets and lease liabilities. Lease liabilities "
        "recognized as of March 31, 2023, stood at ₹3,200 Crore (compared to ₹2,800 Crore in FY2022). Interest expense on "
        "lease liabilities was ₹280 Crore.",
        body_style
    ))
    
    story.append(Paragraph("Note 5.4: Contingent Liabilities", section_style))
    story.append(Paragraph(
        "Contingent liabilities represent obligations that arise from past events whose existence will be confirmed "
        "only by future events. Tax disputes pending with appellate authorities total ₹1,500 Crore. Management, based on "
        "legal advice, has assessed that an outflow of resources is not probable, and hence no provision has been made "
        "in the financial statements.",
        body_style
    ))
    story.append(PageBreak())
    
    # ── Page 6: Segment Reporting & Corporate Governance ──
    story.append(Paragraph("6. Segment Reporting & Corporate Governance", section_style))
    
    # Segment Table
    seg_data = [
        ['Business Segment', 'FY2023 Revenue (₹ in Cr)', 'FY2023 Segment Profit (₹ in Cr)'],
        ['Cloud Services', '1,10,000', '32,000'],
        ['Cyber Security', '80,000', '26,000'],
        ['Consulting', '50,000', '16,000'],
        ['Total', '2,40,000', '74,000']
    ]
    t_seg = Table(seg_data, colWidths=[200, 150, 150])
    t_seg.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#00A3A6')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('ALIGN', (1,0), (-1,-1), 'RIGHT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_seg)
    story.append(Spacer(1, 10))
    
    story.append(Paragraph("Geographic Revenue Breakdown:", section_style))
    story.append(Paragraph(
        "• North America: 52% (₹1,24,800 Crore) - Primary growth market.<br/>"
        "• Europe: 28% (₹67,200 Crore) - Key Cloud and Consulting hub.<br/>"
        "• Asia-Pacific: 20% (₹48,000 Crore) - Emerging cyber operations region.",
        body_style
    ))
    
    story.append(Paragraph("Corporate Governance & Audit Committee", section_style))
    story.append(Paragraph(
        "The Audit Committee of the Board oversees financial reporting, internal controls, and auditor performance. "
        "The committee is composed of 4 independent directors, chaired by Mr. S. Srinivasan. "
        "The committee met 6 times during FY2023. There were no defaults on loan repayments or interest payments "
        "reported during the financial year.",
        body_style
    ))
    
    doc.build(story)
    print(f"Mock PDF successfully generated at: {pdf_path}")

if __name__ == "__main__":
    generate_mock_pdf()
