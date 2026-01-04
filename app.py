"""
FRUITY Dashboard - 메인 엔트리포인트
====================================
현대백화점 청과 수요 예측 기반 발주 지원 시스템

실행 방법:
    cd dashboard
    streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

# 현재 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from auth import get_authenticator, is_admin, init_session_state

# 페이지 설정
st.set_page_config(
    page_title="FRUITY - 청과 수요예측",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)


def login_page():
    """로그인 페이지"""
    st.title("🍎 FRUITY Dashboard")
    st.markdown("##### 현대백화점 청과 수요예측 시스템")
    st.markdown("---")

    authenticator = get_authenticator()

    try:
        authenticator.login(location='main')
    except Exception as e:
        st.error(f"로그인 오류: {e}")

    if st.session_state.get('authentication_status') is False:
        st.error("사용자명 또는 비밀번호가 올바르지 않습니다.")
    elif st.session_state.get('authentication_status') is None:
        st.info("로그인해주세요.")


def order_page():
    """발주의뢰 등록 페이지"""
    from components.header import render_header
    from components.order_table import render_order_table
    from components.chatbot_cold_start import cold_start_dialog
    from data.local_loader import get_predictions_df, check_data_exists
    from auth import get_user_stores

    username = st.session_state.get('username')
    user_stores = get_user_stores(username)

    # CSS 스타일
    st.markdown("""
    <style>
        .main { font-family: 'Malgun Gothic', sans-serif; }
        h2 { color: #1f4e79; border-bottom: 2px solid #1f4e79; padding-bottom: 10px; }
        [data-testid="metric-container"] {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 10px;
            border-radius: 5px;
        }
        .stButton > button { border-radius: 5px; }
        .stNumberInput > div > div > input { text-align: center; }
    </style>
    """, unsafe_allow_html=True)

    # 헤더 렌더링 (날짜/점포 선택)
    filters = render_header(allowed_stores=user_stores)

    # 예측 데이터 로드
    store = filters['store']
    date_t = filters['base_date'].strftime('%Y-%m-%d')
    horizon = filters['horizon']

    # 데이터 존재 확인
    if not check_data_exists(store):
        st.warning(f"점포 {store}의 예측 데이터가 없습니다. 예측을 먼저 실행해주세요.")
        st.info("예측 실행: `python scheduler.py --mode predicting`")
        return

    # 로컬 CSV에서 로드
    df = get_predictions_df(store=store, date_t=date_t, horizon=horizon)

    if df.empty:
        st.warning(f"선택한 조건의 예측 데이터가 없습니다. (점포: {store}, 날짜: {date_t}, horizon: t+{horizon})")
        return

    # 테이블 렌더링
    prediction_date_str = filters['base_date'].strftime('%Y-%m-%d')
    updated_df = render_order_table(df, filters['horizon'], prediction_date_str, store=store)

    # 저장 버튼 (하단)
    st.markdown("---")
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col2:
        if st.button("🆕 신규 SKU 예측", use_container_width=True):
            cold_start_dialog()

    with col3:
        if st.button("📥 임시저장", use_container_width=True):
            st.success("임시저장 완료!")

    with col4:
        if st.button("✅ 발주확정", type="primary", use_container_width=True):
            zero_items = updated_df[updated_df['의뢰수량'] == 0]
            if len(zero_items) > 0:
                st.warning(f"의뢰수량이 0인 항목이 {len(zero_items)}건 있습니다.")
            else:
                st.success("발주가 확정되었습니다!")
                st.balloons()


def admin_page():
    """모델 관리 페이지 (admin만)"""
    from components.admin_model_view import render_admin_model_view
    render_admin_model_view()


def main():
    """메인"""
    init_session_state()

    # 로그인 안 된 경우
    if not st.session_state.get('authentication_status'):
        login_page()
        return

    # 로그인 된 경우
    username = st.session_state.get('username')
    name = st.session_state.get('name')
    admin = is_admin(username)

    # 사이드바
    with st.sidebar:
        st.write(f"**{name}** 님")
        authenticator = get_authenticator()
        authenticator.logout("로그아웃", "sidebar")

        # admin만 페이지 선택 가능
        if admin:
            st.markdown("---")
            page = st.radio(
                "메뉴",
                options=["⚙️ 모델 관리", "📋 발주의뢰"],
            )
        else:
            page = "📋 발주의뢰"

    # 페이지 렌더링
    if page == "⚙️ 모델 관리":
        admin_page()
    else:
        order_page()


if __name__ == "__main__":
    main()