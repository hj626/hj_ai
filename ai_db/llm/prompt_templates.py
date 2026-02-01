# src/llm/prompt_templates.py

def build_summary_prompt(user_case, cases, overall_risk_level):
    case_blocks = []

    for c in cases:
        case_blocks.append(
            f"""
사건명: {c.get('사건명')}
판결유형: {c.get('판결유형')}
유사성 평가: {c.get('similarity_band')}
판단 근거 요약: {c.get('xai_reason')}
""".strip()
        )

    cases_text = "\n\n".join(case_blocks)

    return f"""
당신은 법률 분석 시스템의 내부 결과를
일반 사용자에게 전달하기 위해
'설명문 형태의 요약 텍스트'를 생성하는 역할입니다.

아래 정보는 이미 분석된 결과이며,
새로운 사실이나 판단을 추가해서는 안 됩니다.

출력 형식 규칙:
- 제목, 소제목, 번호, 마크다운 기호(##, ###, -, *)를 사용하지 말 것
- 보고서 형식이 아닌 설명문 형태로 작성할 것
- 문단은 2~3개 이내로 제한할 것

사용자 사건 요약:
{user_case}

유사 판례 분석 요약:
{cases_text}

종합 법적 리스크 수준:
{overall_risk_level}

위 정보를 바탕으로 다음 내용을 하나의 설명문으로 작성하세요.
- 이 사건에서 핵심적으로 문제되는 쟁점
- 유사 판례들이 공통적으로 보여주는 판단 경향
- 사용자 사건에서 불리하게 작용할 수 있는 요소
- 종합적인 리스크 수준의 의미

작성 조건:
- 6~8문장 이내
- 객관적인 설명만 사용할 것
- 법률 비전문가도 이해할 수 있는 한국어
- 추측, 조언, 단정적 표현 금지
- 마지막 문장은 반드시 다음 문구로 끝낼 것:
  "본 내용은 법률 자문이 아니며 참고용 분석입니다."
""".strip()

