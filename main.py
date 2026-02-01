# 통합main / 레이지로딩 기능

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai_hj.llm import main as hj_main
from ai_db.app import main as db_main


app = FastAPI(title="Legal_AI API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8484", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 각 모듈 필요할 때 import
hj_module = None
db_module = None


def get_hj_module():
    """승소율/형량 분석 모듈 - 첫 호출시에만 import"""
    global hj_module
    if hj_module is None:
        print("sj_LLM 모듈 로딩 중.")
        hj_module = hj_main
        print("sj_LLM 모듈 로딩 완료")
    return hj_module

def get_db_module():
    """판례 검색 모듈 - 첫 호출시에만 import"""
    global db_module
    if db_module is None:
        print("🔄 판례 검색 모듈 로딩 중.")
        db_module = db_main
        print("✅ 판례 검색 모듈 로딩 완료!")
    return db_module


# Request 스키마
class AnalyzeRequest(BaseModel):
    case_text: str

class CaseRequest(BaseModel):
    case_text: str


# 승소율 탭 - 클릭시 llm/main.py 로딩
@app.post("/analyze/win-rate")
async def analyze_win_rate(request: AnalyzeRequest):
    """승소율 탭 클릭 → 여기서 처음 llm/main.py import"""
    try:
        llm = get_hj_module()  # 여기서 처음 import!
        return await llm.analyze_win_rate(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 형량 탭 - 클릭시 llm/main.py 로딩 (이미 로딩됐으면 재사용)
@app.post("/analyze/sentence")
async def analyze_sentence(request: AnalyzeRequest):
    """형량 탭 클릭 → llm/main.py 재사용"""
    try:
        llm = get_hj_module()  # 이미 import 됐으면 재사용
        return await llm.analyze_sentence(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 판례 검색 탭 - 클릭시 app/main.py 로딩
@app.post("/analyze")
async def analyze_case(request: CaseRequest):
    """판례 검색 탭 클릭 → 여기서 처음 app/main.py import"""
    try:
        case = get_db_module()  # 여기서 처음 import!
        return await case.analyze(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/case/{case_id}/summary")
async def case_summary(case_id: str):
    try:
        case = get_db_module()
        return await case.case_summary(case_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/case/{case_id}/full")
async def case_full(case_id: str):
    try:
        case = get_db_module()
        return await case.case_full(case_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("="*50)
    print("⚖️  법률 AI 통합 서버 시작")
    print("📍 http://0.0.0.0:8000")
    print("💡 Lazy Loading: 탭 클릭시 모듈 로딩")
    print("="*50)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)