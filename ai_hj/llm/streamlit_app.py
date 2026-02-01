# streamlit_app.py
import streamlit as st
import json
from jem_api import LegalAnalyzer
import time


# 페이지 설정
st.set_page_config(
    page_title="⚖️ AI 법률 상담 시스템",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .feedback-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None



# streamlit_app.py 수정 제안

# 1. 모델 로드 함수를 따로 만듭니다.
@st.cache_resource  # 👈 이 코드가 있으면 처음 한 번만 로드하고 계속 재사용합니다!
# def get_analyzer(model_path, api_key):
def get_analyzer():
    return LegalAnalyzer(
        model_path="../lerning/saved_mode3",
        gemini_api_key="GEMINI_API_KEY"
    )

# 2. 사이드바 설정 부분 아래에 바로 추가
# if 'analyzer' not in st.session_state:
if 'analyzer' not in st.session_state or st.session_state.analyzer is None:
    try:
        with st.sidebar:
        # 시작하자마자 자동으로 모델 로드 시작
            with st.spinner("모델 로딩중"):
                st.session_state.analyzer = get_analyzer()
        st.sidebar.success("모델 준비 완료")
    except Exception as e:
        st.sidebar.error(f"❌ 자동 로드 실패: {e}")





# 사이드바 - 설정
with st.sidebar:
    st.header("🔧 설정")
    
    model_path = st.text_input(
        "모델 경로",
        value="../lerning/saved_mode3",
        help="학습된 BERT 모델이 저장된 경로"
    )
          
    # # 모델 로드 버튼
    # if st.button("🚀 모델 로드", use_container_width=True):
    #     try:
    #         with st.spinner("모델 로딩 중..."):
    #             st.session_state.analyzer = LegalAnalyzer(
    #                 model_path=model_path,
    #                 gemini_api_key="GEMINI_API_KEY"
    #             )
    #         st.success("✅ 모델 로드 완료!")
    #     except Exception as e:
    #         st.error(f"❌ 오류 발생: {str(e)}")
    
    # 모델 상태 표시
    if st.session_state.analyzer:
        st.info("✅ 모델 준비 완료")
    else:
        st.warning("⚠️ 모델을 먼저 로드하세요")
    
    st.markdown("---")
    
    # 예시 사연 버튼들
    st.subheader("📝 예시 사연")
    
    if st.button("예시 1: 부당해고", use_container_width=True):
        st.session_state.example_story = """저는 회사에서 부당해고를 당했습니다. 
5년간 성실히 근무했으나 경영상의 이유로 갑자기 해고 통보를 받았습니다.
퇴직금 500만원도 받지 못했고, 해고 예고 수당도 없었습니다.
회사는 구두로만 통보했고 서면 통지는 받지 못했습니다."""
    
    if st.button("예시 2: 교통사고", use_container_width=True):
        st.session_state.example_story = """신호대기 중 뒤에서 추돌당했습니다.
상대방이 100% 과실인데도 보험처리를 거부하고 있습니다.
병원 치료비 200만원과 차량 수리비 300만원이 발생했습니다.
진단서는 전치 2주입니다."""
    
    if st.button("예시 3: 임대차 분쟁", use_container_width=True):
        st.session_state.example_story = """임대인이 보증금 1000만원을 돌려주지 않습니다.
계약서에 명시된 날짜가 지났는데도 연락이 두절되었습니다.
집도 이미 비웠고 원상복구도 완료했습니다."""

# 메인 영역
st.markdown('<p class="main-header">⚖️ AI 법률 상담 시스템</p>', unsafe_allow_html=True)

# 탭 생성
tab1, tab2, tab3 = st.tabs(["📝 사연 입력", "📊 분석 결과", "ℹ️ 사용 가이드"])

with tab1:
    st.subheader("📝 법률 상담이 필요한 사연을 입력하세요")
    
    # 예시 사연이 선택되었으면 자동으로 채우기
    default_text = st.session_state.get('example_story', '')
    
    user_story = st.text_area(
        "사연 입력",
        value=default_text,
        height=200,
        placeholder="예: 저는 회사에서 부당해고를 당했습니다..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        analyze_button = st.button("🔍 분석 시작", use_container_width=True, type="primary")
    
    with col2:
        clear_button = st.button("🗑️ 초기화", use_container_width=True)
    
    if clear_button:
        st.session_state.analysis_result = None
        if 'example_story' in st.session_state:
            del st.session_state.example_story
        st.rerun()
    
    if analyze_button:
        if not st.session_state.analyzer:
            st.error("⚠️ 먼저 모델을 로드해주세요! (좌측 사이드바)")
        elif not user_story.strip():
            st.warning("⚠️ 사연을 입력해주세요!")
        else:
            try:
                # 분석 진행
                with st.spinner("🔍 BERT 모델 분석 중..."):
                    progress_bar = st.progress(0)
                    time.sleep(0.5)
                    progress_bar.progress(30)
                    
                    bert_results = st.session_state.analyzer.predict_bert(user_story)
                    progress_bar.progress(60)
                
                with st.spinner("💬 Gemini AI 피드백 생성 중..."):
                    time.sleep(0.5)
                    progress_bar.progress(80)
                    
                    result = st.session_state.analyzer.analyze(user_story)
                    progress_bar.progress(100)
                
                st.session_state.analysis_result = result
                st.success("✅ 분석 완료!")
                time.sleep(0.5)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ 분석 중 오류 발생: {str(e)}")

with tab2:
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        st.subheader("📊 AI 분석 결과")
        
        # 메트릭 카드들
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📊 승소율",
                value=f"{result['win_rate']:.1f}%",
                delta=None
            )
        
        with col2:
            if result['sentence'] > 0.1:
                st.metric(
                    label="⚖️ 예상 형량",
                    value=f"{result['sentence']:.1f}년",
                    delta=None
                )
            else:
                st.metric(
                    label="⚖️ 예상 형량",
                    value="해당 없음",
                    delta=None
                )
        
        with col3:
            if result['fine'] > 10000:
                st.metric(
                    label="💰 예상 벌금",
                    value=f"{result['fine']:,.0f}원",
                    delta=None
                )
            else:
                st.metric(
                    label="💰 예상 벌금",
                    value="해당 없음",
                    delta=None
                )
        
        with col4:
            risk_color = "🟢" if result['risk'] < 30 else "🟡" if result['risk'] < 70 else "🔴"
            st.metric(
                label=f"{risk_color} 위험도",
                value=f"{result['risk']:.1f}/100",
                delta=None
            )
        
        st.markdown("---")
        
        # 위험도 분석
        st.subheader("⚠️ 위험도 분석")
        if result['risk'] >= 70:
            st.error("🔴 **높은 위험**: 즉시 전문 변호사 상담을 받으시기 바랍니다.")
        elif result['risk'] >= 40:
            st.warning("🟡 **중간 위험**: 법률 전문가와 상담하시는 것이 좋습니다.")
        else:
            st.success("🟢 **낮은 위험**: 기본적인 법률 절차로 대응 가능할 것으로 보입니다.")
        
        st.markdown("---")
        
        # Gemini 피드백
        st.subheader("💡 AI 전문가 피드백")
        st.markdown(
            f'<div class="feedback-box">{result["feedback"]}</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # JSON 다운로드
        col1, col2 = st.columns(2)
        
        with col1:
            json_str = json.dumps(result, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 JSON 다운로드",
                data=json_str,
                file_name="analysis_result.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # 텍스트 리포트 생성
            report = f"""
# AI 법률 분석 리포트

## 📋 사연
{result['original_story']}

## 📊 분석 결과
- 승소율: {result['win_rate']:.1f}%
- 예상 형량: {result['sentence']:.1f}년
- 예상 벌금: {result['fine']:,.0f}원
- 위험도: {result['risk']:.1f}/100

## 💡 전문가 피드백
{result['feedback']}
"""
            st.download_button(
                label="📄 텍스트 리포트 다운로드",
                data=report,
                file_name="legal_report.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    else:
        st.info("👈 먼저 '사연 입력' 탭에서 분석을 진행해주세요!")

with tab3:
    st.subheader("ℹ️ 사용 가이드")
    
    st.markdown("""
    ### 🚀 시작하기
    
    1. **모델 로드**
       - 좌측 사이드바에서 '🚀 모델 로드' 버튼 클릭
       - 모델 경로와 API 키가 자동으로 설정됩니다
    
    2. **사연 입력**
       - '📝 사연 입력' 탭에서 법률 상담이 필요한 내용 작성
       - 또는 사이드바의 예시 사연 버튼 클릭
    
    3. **분석 시작**
       - '🔍 분석 시작' 버튼 클릭
       - BERT 모델과 Gemini AI가 분석을 수행합니다
    
    4. **결과 확인**
       - '📊 분석 결과' 탭에서 상세 결과 확인
       - JSON 또는 텍스트 형식으로 다운로드 가능
    
    ---
    
    ### 📊 분석 항목
    
    - **승소율**: AI가 예측한 승소 가능성 (0-100%)
    - **예상 형량**: 형사 사건의 경우 예상되는 형량
    - **예상 벌금**: 벌금형이 예상되는 경우 금액
    - **위험도**: 사건의 심각도 및 법적 위험 수준 (0-100)
    
    ---
    
    ### ⚠️ 주의사항
    
    - 이 시스템은 참고용 AI 분석 결과입니다
    - 실제 법률 자문을 대체할 수 없습니다
    - 중요한 사안은 반드시 전문 변호사와 상담하세요
    
    ---
    
    ### 🔧 기술 스택
    
    - **BERT**: 한국어 법률 문서 학습 모델 (klue/bert-base)
    - **Gemini AI**: Google의 최신 생성 AI
    - **Streamlit**: 빠른 웹 앱 프로토타이핑
    """)
    
    st.markdown("---")
    st.info("💡 **Tip**: 더 정확한 분석을 위해 구체적이고 상세한 사연을 작성해주세요!")

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        ⚖️ AI 법률 상담 시스템 v1.0 | Powered by BERT & Gemini AI
    </div>
    """,
    unsafe_allow_html=True
)