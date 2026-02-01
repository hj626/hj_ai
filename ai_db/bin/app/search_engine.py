# app/search_engine.py
"""
사건 유형별 Subset 검색 엔진
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def get_search_subset(case_type: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    사건 유형에 따라 검색 대상 subset을 반환
    
    Args:
        case_type: "형사", "가사", "노동", "전체" 중 하나
        df: 전체 판례 DataFrame
        
    Returns:
        필터링된 DataFrame
    """
    if case_type == "형사":
        # 사건종류명이 "형사"인 것만
        subset = df[df["사건종류명"] == "형사"]
        print(f"✅ 형사 subset: {len(subset)} rows")
        return subset
    
    if case_type == "가사":
        # 사건종류명이 "가사"인 것만
        subset = df[df["사건종류명"] == "가사"]
        print(f"✅ 가사 subset: {len(subset)} rows")
        return subset
    
    if case_type == "노동":
        # 노동은 민사/일반행정에 섞여있으므로 키워드 필터 필수
        labor_keywords = r"근로자|임금|해고|퇴직금|부당해고|근로계약|노동위원회|산재|근로기준법"
        
        subset = df[
            (df["사건종류명"].isin(["민사", "일반행정"])) &
            (df["case_text"].str.contains(
                labor_keywords,
                regex=True,
                na=False,
                case=False
            ))
        ]
        print(f"✅ 노동 subset: {len(subset)} rows")
        return subset
    
    # "전체" 또는 기타
    print(f"✅ 전체 검색: {len(df)} rows")
    return df


def search_with_fallback(
    query_vec: np.ndarray,
    faiss_index,
    df_full: pd.DataFrame,
    case_type: str,
    top_k: int = 10,
    fallback_threshold: int = 3
) -> pd.DataFrame:
    """
    Subset 검색 + Fallback 로직
    
    Args:
        query_vec: 쿼리 임베딩 벡터
        faiss_index: FAISS 인덱스
        df_full: 전체 판례 DataFrame
        case_type: 추정된 사건 유형
        top_k: 최종 반환할 결과 수
        fallback_threshold: 이 개수 미만이면 전체 검색으로 확장
        
    Returns:
        검색 결과 DataFrame
    """
    # 1️⃣ Subset 결정
    subset_df = get_search_subset(case_type, df_full)
    
    # 2️⃣ FAISS 검색 (여유있게)
    D, I = faiss_index.search(query_vec, top_k * 5)
    
    # 3️⃣ Subset mask 적용
    candidates = df_full.iloc[I[0]].copy()
    filtered = candidates[candidates.index.isin(subset_df.index)]
    
    print(f"📊 Subset 검색 결과: {len(filtered)} 건")
    
    # 4️⃣ Fallback: 결과가 너무 적으면 전체 검색
    if len(filtered) < fallback_threshold and case_type != "전체":
        print(f"⚠️ 결과 부족 ({len(filtered)} < {fallback_threshold}) → 전체 검색으로 확장")
        
        # 전체 다시 검색
        D_full, I_full = faiss_index.search(query_vec, top_k * 3)
        filtered = df_full.iloc[I_full[0]].copy()
        
        # Distance 정규화
        d_min, d_max = D_full[0].min(), D_full[0].max()
        filtered["similarity"] = ((d_max - D_full[0]) / (d_max - d_min + 1e-8)).clip(0, 1)
        
        return filtered.head(top_k)
    
    # 5️⃣ 정상 반환 (similarity 계산)
    if len(filtered) > 0:
        d_min, d_max = D[0].min(), D[0].max()
        
        # 원본 인덱스 기준으로 distance 매핑
        distance_map = dict(zip(I[0], D[0]))
        filtered["_distance"] = filtered.index.map(distance_map)
        filtered["similarity"] = (
            (d_max - filtered["_distance"]) / (d_max - d_min + 1e-8)
        ).clip(0, 1)
        filtered = filtered.drop(columns=["_distance"])
    
    return filtered.head(top_k)


def format_search_results(results_df: pd.DataFrame, case_type: str, confidence: float) -> List[Dict[str, Any]]:
    """
    검색 결과를 API 응답 형식으로 변환
    """
    formatted = []
    
    for idx, row in results_df.iterrows():
        formatted.append({
            "case_id": str(row.get("사건번호", "")),
            "case_name": str(row.get("사건명", "")),
            "court": str(row.get("법원명", "")),
            "case_number": str(row.get("사건번호", "")),
            "case_type": str(row.get("사건종류명", "")),
            "decision_type": str(row.get("판결유형", "판결")),
            "similarity": float(row.get("similarity", 0)),
            "xai_reason": (
                f"{int(row.get('similarity', 0) * 100)}% 유사도 · "
                f"{row.get('사건종류명', '기타')} 사건"
            )
        })
    
    return formatted