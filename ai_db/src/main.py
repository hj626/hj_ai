from input_processing import CaseInput
from core_ai import search_similar_cases, calculate_risk_score
from output_processing import generate_risk_chart, generate_pdf_report
from sentence_transformers import SentenceTransformer

# 테스트용 입력
user_input = CaseInput(
    case_type='민사',
    case_text='임대인이 임차인의 계약 불이행으로 손해배상을 청구한 사건',
    weight_facts=1.0,
    weight_issues=1.0,
    weight_outcome=1.0
)

# 임베딩 모델 로딩 (실제 학습된 모델 Notebook에서)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
case_vector = model.encode([user_input.case_text])[0]

# 유사 판례 검색
similar_cases = search_similar_cases(case_vector, top_k=5)

# 리스크 점수 계산
weights = {
    'facts': user_input.weight_facts,
    'issues': user_input.weight_issues,
    'outcome': user_input.weight_outcome
}
risk_score = calculate_risk_score(similar_cases, weights)

# 결과 처리
generate_risk_chart(risk_score)
generate_pdf_report(user_input.case_text, similar_cases, risk_score)

print("테스트 완료: reports/report.pdf 및 risk_chart.png 생성됨")
