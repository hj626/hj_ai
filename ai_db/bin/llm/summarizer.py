# src/llm/summarizer.py

from .gemini_client import call_gemini
from .prompt_templates import build_summary_prompt


def generate_case_summary(
    user_case: str,
    results_df,
    overall_risk_level: str,
) -> str:
    """
    유사 판례 분석 결과를 바탕으로
    사용자용 종합 설명 요약 생성
    """

    top_cases = (
        results_df
        .head(5)
        .to_dict(orient="records")
    )

    prompt = build_summary_prompt(
        user_case=user_case,
        cases=top_cases,
        overall_risk_level=overall_risk_level
    )

    return call_gemini(prompt)
