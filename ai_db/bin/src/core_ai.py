import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from llm.summarizer import generate_case_summary


# =========================
# 1. 리소스 로딩
# =========================

CASE_DF = pd.read_parquet("korean_precedents_clean.parquet")
FAISS_INDEX = faiss.read_index("case_index.faiss")

EMBEDDING_MODEL = SentenceTransformer(
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
)

TOP_K = 5


# =========================
# 2. Similarity / XAI
# =========================

def normalize_faiss_distance(distances: np.ndarray) -> np.ndarray:
    """
    FAISS L2 distance → [0,1] similarity
    """
    d_min, d_max = distances.min(), distances.max()
    sim = (d_max - distances) / (d_max - d_min + 1e-8)
    return sim.clip(0, 1)


def similarity_band(sim: float) -> str:
    if sim >= 0.85:
        return "입력 사건과 사실관계 및 법적 쟁점이 거의 동일한 핵심 판례"
    elif sim >= 0.60:
        return "사실관계와 주요 쟁점이 상당 부분 유사한 판례"
    elif sim >= 0.35:
        return "일부 쟁점에서 참고할 수 있는 유사 판례"
    else:
        return "쟁점 구조만 부분적으로 유사한 보조 참고 판례"


def explain_xai(row) -> str:
    return (
        f"이 판례는 입력 사건과 비교했을 때 "
        f"{row['similarity_band']}로 분류됩니다. "
        f"해당 사건에서는 '{row['판결유형']}' 판단이 내려졌으며, "
        f"유사한 사실관계에서의 법원 판단 경향을 참고할 수 있습니다."
    )


# =========================
# 3. FAISS 검색
# =========================

def search_similar_cases(user_case_text: str) -> pd.DataFrame:
    """
    사용자 사건 → 유사 판례 검색 + XAI 생성
    """
    query_vec = EMBEDDING_MODEL.encode(
        [user_case_text],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = FAISS_INDEX.search(query_vec, TOP_K)

    results = CASE_DF.iloc[indices[0]].copy()
    results["similarity"] = normalize_faiss_distance(distances[0])

    results["similarity_band"] = results["similarity"].apply(similarity_band)
    results["xai_reason"] = results.apply(explain_xai, axis=1)

    return results[[
        "사건명",
        "판결유형",
        "similarity",
        "similarity_band",
        "xai_reason"
    ]]


# =========================
# 4. 종합 리스크 규칙
# =========================

def determine_overall_risk(similarities: pd.Series) -> str:
    """
    유사도 분포 기반 종합 리스크 판단 (Rule-based)
    """
    high = (similarities >= 0.85).sum()
    mid = (similarities >= 0.60).sum()

    if high >= 2:
        return "높음"
    elif high == 1 or mid >= 2:
        return "중간"
    else:
        return "낮음"


# =========================
# 5. 엔트리 포인트
# =========================

def run_case_analysis(user_case_text: str) -> dict:
    """
    전체 분석 파이프라인 엔트리 포인트
    """
    cases_df = search_similar_cases(user_case_text)

    overall_risk_level = determine_overall_risk(
        cases_df["similarity"]
    )

    summary = generate_case_summary(
        user_case=user_case_text,
        results_df=cases_df,
        overall_risk_level=overall_risk_level
    )

    return {
        "summary": summary,
        "overall_risk_level": overall_risk_level,
        "cases": cases_df.to_dict(orient="records"),
        "disclaimer": "본 결과는 법률 자문이 아니며 참고용 분석입니다."
    }
