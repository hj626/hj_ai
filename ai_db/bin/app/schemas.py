# app/schemas.py
from pydantic import BaseModel
from typing import List, Optional


# -----------------------------
# 요청 모델
# -----------------------------
class CaseRequest(BaseModel):
    case_type: Optional[str] = None  # ✅ Optional로 변경 (UI에서 안 보내도 됨)
    case_text: str                   # 사건 본문 텍스트 (필수)


# -----------------------------
# /analyze 관련 모델
# -----------------------------
class SimilarCase(BaseModel):
    case_id: Optional[str]           # ✅ None 가능 (사건번호 누락 케이스)
    case_name: str
    court: str
    case_number: str

    decision_type: str               # 판결 / 결정
    decision_result: str             # 상고기각 / 파기환송 / 파기자판 / 판단불명

    similarity: float                # 0~1
    case_type_label: str             # ✅ 추가: 판례의 사건종류명 (민사/형사/가사 등)
    xai_reason: str                  # 유사도 근거 설명


class CaseResponse(BaseModel):
    overall_risk_level: str          # 낮음 / 중간 / 높음
    summary: str                     # LLM 요약
    similar_cases: List[SimilarCase]
    
    # ✅ 추가: 자동 분류 정보
    inferred_case_type: str          # AI가 추정한 유형 (형사/가사/노동/전체)
    case_type_label: str             # UI 표시용 ("형사 사건", "노동 분쟁" 등)
    case_type_confidence: float      # 신뢰도 0.0 ~ 1.0
    case_type_description: str       # 사용자에게 보여줄 설명


# -----------------------------
# /case/{case_id}/summary 관련 모델
# -----------------------------
class CaseSummaryResponse(BaseModel):
    case_id: str
    summary: str                     # 판시사항 / 주문 중심 요약


# -----------------------------
# /case/{case_id}/full 관련 모델
# -----------------------------
class CaseFullTextResponse(BaseModel):
    case_id: str
    case_name: str
    full_text: str                   # 판례 전체 전문
    summary: Optional[str] = ""      # ✅ 추가: 요약도 함께 반환