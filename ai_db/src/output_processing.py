from fpdf import FPDF
import matplotlib.pyplot as plt

def generate_risk_chart(risk_score):
    plt.bar(['Risk'], [risk_score])
    plt.savefig('reports/risk_chart.png')
    plt.close()

def generate_pdf_report(case_text, similar_cases, risk_score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"사건 개요: {case_text}")
    pdf.multi_cell(0, 10, f"리스크 점수: {risk_score:.2f}")
    for c in similar_cases:
        pdf.multi_cell(0, 10, f"유사 판례: {c['title']} (distance: {c['distance']:.2f})")
    pdf.output("reports/report.pdf")
