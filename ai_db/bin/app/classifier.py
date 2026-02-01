# app/classifier.py
"""
사건 유형 자동 분류 모듈
Rule-based 방식으로 텍스트에서 사건 유형을 추정
"""
import re
from typing import Tuple

def infer_case_type(text: str) -> Tuple[str, float]:
    """
    텍스트에서 사건 유형을 자동으로 추정
    
    Args:
        text: 사용자가 입력한 사연
        
    Returns:
        (case_type, confidence): 추정된 유형과 신뢰도 (0.0~1.0)
    """
    if not text or not text.strip():
        return "전체", 0.0
    
    text = text.lower()
    
    # 형사 패턴 (높은 특이도)
    criminal_patterns = [
        r"폭행|상해|절도|사기|횡령|배임",
        r"고소|고발|처벌|기소|구속",
        r"징역|벌금|집행유예|실형",
        r"형사|검찰|경찰서|수사",
        r"피의자|피고인|범죄"
    ]
    criminal_score = sum(
        1 for p in criminal_patterns 
        if re.search(p, text)
    )
    
    # 가사 패턴
    family_patterns = [
        r"이혼|별거|혼인",
        r"양육권|친권|면접교섭",
        r"위자료|재산분할|혼인재산",
        r"배우자|남편|아내|자녀",
        r"가정법원|가사"
    ]
    family_score = sum(
        1 for p in family_patterns 
        if re.search(p, text)
    )
    
    # 노동 패턴
    labor_patterns = [
        r"해고|부당해고|징계해고",
        r"임금|급여|월급|퇴직금",
        r"근로계약|고용계약|근로자",
        r"노동위원회|근로기준법",
        r"부당노동행위|산재|산업재해",
        r"연장근로|야간근무|휴게시간"
    ]
    labor_score = sum(
        1 for p in labor_patterns 
        if re.search(p, text)
    )
    
    # 점수 기반 판단 (우선순위: 형사 > 가사 > 노동)
    max_score = max(criminal_score, family_score, labor_score)
    
    if max_score == 0:
        return "전체", 0.3  # 애매한 경우
    
    if criminal_score == max_score and criminal_score >= 2:
        confidence = min(0.9, 0.6 + criminal_score * 0.1)
        return "형사", confidence
    
    if family_score == max_score and family_score >= 2:
        confidence = min(0.9, 0.6 + family_score * 0.1)
        return "가사", confidence
    
    if labor_score == max_score and labor_score >= 2:
        confidence = min(0.9, 0.6 + labor_score * 0.1)
        return "노동", confidence
    
    # 점수가 낮으면 전체 검색
    return "전체", 0.4


def get_case_type_label(case_type: str) -> str:
    """UI 표시용 레이블"""
    labels = {
        "형사": "형사 사건",
        "가사": "가사 사건",
        "노동": "노동 분쟁",
        "전체": "일반 민사"
    }
    return labels.get(case_type, "일반 사건")


def get_case_type_description(case_type: str, confidence: float) -> str:
    """사용자에게 보여줄 설명"""
    if confidence >= 0.8:
        reliability = "높은 확신으로"
    elif confidence >= 0.6:
        reliability = "상당한 확신으로"
    else:
        reliability = "추정상"
    
    descriptions = {
        "형사": f"{reliability} 형사 사건으로 판단하여 형사 판례를 검색합니다.",
        "가사": f"{reliability} 가사 사건으로 판단하여 가사 판례를 검색합니다.",
        "노동": f"{reliability} 노동 분쟁으로 판단하여 노동 관련 판례를 검색합니다.",
        "전체": "구체적인 사건 유형이 명확하지 않아 전체 판례를 검색합니다."
    }
    
    return descriptions.get(case_type, "판례를 검색합니다.")