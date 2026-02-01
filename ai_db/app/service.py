# app/service.py
"""
메인 서비스 로직
- case_type이 있으면 그대로 사용 (하위 호환)
- case_type이 없으면 자동 분류
"""
import pandas as pd
import faiss
import re
from sentence_transformers import SentenceTransformer
from app.llm.summarizer import generate_case_summary
from app.schemas import CaseSummaryResponse, CaseFullTextResponse
from app.classifier import infer_case_type, get_case_type_label, get_case_type_description
from app.search_engine import get_search_subset, search_with_fallback

# ------------------------
# 0️⃣ 데이터 로드
# ------------------------
print("\n" + "=" * 80)
print("🚀 서비스 초기화 중...")
print("=" * 80)

df_analysis = pd.read_parquet(
    r"C:\LawAI\notebooks\korean_precedents_embedded.parquet",
    engine="pyarrow"
)

df_full = pd.read_csv(
    r"C:\LawAI\notebooks\korean_precedents_clean.csv"
)

print(f"✅ Parquet 로드: {len(df_analysis)} rows")
print(f"✅ CSV 로드: {len(df_full)} rows")

# ✅ 사건번호 → 인덱스 매핑
case_id_to_idx = {}
for idx, row in df_full.iterrows():
    case_num = row.get("사건번호")
    if pd.notna(case_num):
        normalized = str(case_num).strip()
        if normalized:
            case_id_to_idx[normalized] = idx

print(f"✅ case_id_to_idx 크기: {len(case_id_to_idx)}")

# 사건종류명 분포 출력
print("\n📊 사건종류명 분포:")
print(df_analysis["사건종류명"].value_counts().head(10))
print("=" * 80 + "\n")

faiss_index = faiss.read_index(
    r"C:\LawAI\notebooks\case_index.faiss"
)

model = SentenceTransformer(
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
)

# ------------------------
# 판결 결과 추출
# ------------------------
def extract_decision_result(case_text: str) -> str:
    if not case_text:
        return "판단불명"

    order_match = re.search(
        r"【주\s*문】(.+?)(【이\s*유】|$)",
        case_text,
        re.DOTALL
    )
    target = order_match.group(1) if order_match else case_text

    patterns = {
        "파기환송": r"(파기|파훼).*(환송|차려)",
        "상고기각": r"상고.*기각",
        "인용": r"청구.*인용|원고.*승소",
        "기각": r"청구.*기각",
    }

    for label, pattern in patterns.items():
        if re.search(pattern, target):
            return label
    return "판단불명"

DECISION_RISK_MAP = {
    "상고기각": 0.85,
    "기각": 0.8,
    "파기환송": 0.5,
    "인용": 0.2,
    "판단불명": 0.5
}

def similarity_band(sim: float) -> str:
    if sim >= 0.85:
        return "매우 높은 유사도"
    elif sim >= 0.65:
        return "상당한 유사도"
    elif sim >= 0.4:
        return "일부 쟁점 유사"
    else:
        return "참고 수준"

# ------------------------
# 1️⃣ /analyze
# ------------------------
def analyze_case(request):
    import time
    start = time.time()
    
    print("\n" + "=" * 80)
    print("🚀 analyze_case START")
    print("=" * 80)
    print(f"입력 텍스트 길이: {len(request.case_text)} chars")

    if not request.case_text or not request.case_text.strip():
        raise ValueError("case_text is empty")

    # ✅ case_type 처리: 있으면 사용, 없으면 자동 추정
    if request.case_type:
        # 기존 방식 (하위 호환)
        inferred_type = request.case_type
        confidence = 1.0  # 사용자가 직접 선택했으므로 100%
        print(f"📌 사용자 지정 case_type: {inferred_type}")
    else:
        # 새로운 방식 (자동 분류)
        inferred_type, confidence = infer_case_type(request.case_text)
        print(f"🔍 자동 분류: {inferred_type} (신뢰도: {confidence:.2f})")
    
    type_label = get_case_type_label(inferred_type)
    type_desc = get_case_type_description(inferred_type, confidence)

    # ✅ 쿼리 임베딩
    query_vec = model.encode([request.case_text]).astype("float32")

    # ✅ Subset 검색 + Fallback
    results = search_with_fallback(
        query_vec=query_vec,
        faiss_index=faiss_index,
        df_full=df_analysis,
        case_type=inferred_type,
        top_k=10,
        fallback_threshold=3
    )

    print(f"\n📊 최종 검색 결과: {len(results)} 건")

    # ✅ 후처리
    results["similarity_band"] = results["similarity"].apply(similarity_band)
    results["decision_result"] = results["case_text"].apply(extract_decision_result)
    results["risk_score"] = results["decision_result"].map(DECISION_RISK_MAP).fillna(0.5)

    avg_risk = results["risk_score"].mean() if len(results) > 0 else 0.5
    overall_risk = (
        "높음" if avg_risk >= 0.7 else "중간" if avg_risk >= 0.4 else "낮음"
    )

    top_cases = results.head(5)

    # ✅ 요약 생성
    try:
        summary = generate_case_summary(
            user_case=request.case_text,
            results_df=top_cases,
            overall_risk_level=overall_risk
        )
    except Exception as e:
        print(f"⚠️ 요약 생성 오류: {e}")
        summary = "요약 생성 중 오류가 발생했습니다."

    # ✅ 응답 생성
    similar_cases_list = []
    
    for idx, (i, r) in enumerate(top_cases.iterrows()):
        case_num_raw = r.get("사건번호")
        
        case_id = None
        if pd.notna(case_num_raw):
            normalized = str(case_num_raw).strip()
            if normalized in case_id_to_idx:
                case_id = normalized
        
        similar_cases_list.append({
            "case_id": case_id,
            "case_name": str(r.get("사건명", "")),
            "court": str(r.get("법원명", "")),
            "case_number": str(case_num_raw) if pd.notna(case_num_raw) else "",
            "decision_type": str(r.get("판결유형", "판결")),
            "decision_result": str(r.get("decision_result", "판단불명")),
            "similarity": float(r.get("similarity", 0)),
            "case_type_label": str(r.get("사건종류명", "")),  # ✅ 추가
            "xai_reason": (
                f"{r['similarity_band']}에 해당하며 판단 결과는 '{r['decision_result']}'입니다."
            ),
        })

    print(f"\n✅ analyze_case END: {time.time() - start:.2f}s")
    print("=" * 80 + "\n")

    return {
        "overall_risk_level": overall_risk,
        "summary": summary,
        "similar_cases": similar_cases_list,
        # ✅ 자동 분류 정보
        "inferred_case_type": inferred_type,
        "case_type_label": type_label,
        "case_type_confidence": confidence,
        "case_type_description": type_desc,
    }

# ------------------------
# 2️⃣ /case/{case_id}/summary
# ------------------------
def get_case_summary(case_id: str) -> CaseSummaryResponse:
    """사건 요약 조회"""
    case_id_norm = case_id.strip()
    
    if case_id_norm not in case_id_to_idx:
        raise ValueError(f"Case not found: {case_id}")
    
    idx = case_id_to_idx[case_id_norm]
    row = df_full.iloc[idx:idx+1]

    try:
        summary = generate_case_summary(
            user_case="",
            results_df=row,
            overall_risk_level=""
        )
    except Exception as e:
        print(f"⚠️ 요약 생성 오류: {e}")
        summary = "요약 생성 불가"

    return CaseSummaryResponse(case_id=case_id, summary=summary)

# ------------------------
# 3️⃣ /case/{case_id}/full
# ------------------------
def get_case_full_text(case_id: str) -> CaseFullTextResponse:
    """판례 전문 조회"""
    print(f"📂 get_case_full_text: '{case_id}'")
    
    case_id_norm = case_id.strip()
    
    if case_id_norm not in case_id_to_idx:
        print(f"❌ Case not found: {case_id}")
        raise ValueError(f"Case not found: {case_id}")
    
    idx = case_id_to_idx[case_id_norm]
    r = df_full.iloc[idx]
    
    full_text = r.get("case_text", "")
    
    if not full_text or pd.isna(full_text):
        print(f"⚠️ full_text 비어있음")
        full_text = "판례 전문을 찾을 수 없습니다."
    else:
        print(f"✅ full_text 로드 성공: {len(full_text)} chars")
    
    # 요약
    try:
        summary = generate_case_summary(
            user_case="",
            results_df=df_full.iloc[idx:idx+1],
            overall_risk_level=""
        )
    except Exception as e:
        print(f"⚠️ 요약 생성 오류: {e}")
        summary = ""

    return CaseFullTextResponse(
        case_id=case_id,
        case_name=str(r.get("사건명", "")),
        full_text=full_text,
        summary=summary
    )