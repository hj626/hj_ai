from pydantic import BaseModel, validator
import re

class CaseInput(BaseModel):
    case_type: str
    case_text: str
    weight_facts: float = 1.0
    weight_issues: float = 1.0
    weight_outcome: float = 1.0

    @validator('case_type')
    def validate_case_type(cls, v):
        if v not in ['민사', '형사', '노동', '가사']:
            raise ValueError('유효하지 않은 사건 유형')
        return v

    @validator('case_text')
    def clean_text(cls, v):
        # HTML/스크립트 제거
        return re.sub(r'<.*?>', '', v)
