"""
Order Table Component
=====================
발주의뢰 테이블 (수요 예측 + LLM 챗봇)
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import uuid
from pathlib import Path

# 상위 디렉토리 import
sys.path.insert(0, str(Path(__file__).parent.parent))

from components.chatbot import get_chatbot
from data.supabase_client import save_chat_history
import config


def render_order_table(df: pd.DataFrame, horizon: int, prediction_date: str = None) -> pd.DataFrame:
    """
    발주의뢰 테이블 렌더링

    Parameters
    ----------
    df : pd.DataFrame
        예측 데이터가 포함된 발주 목록
    horizon : int
        예측 horizon (1~4)
    prediction_date : str
        예측 기준일 (YYYY-MM-DD 형식)

    Returns
    -------
    pd.DataFrame
        수정된 발주 목록 (의뢰수량 포함)
    """
    st.markdown(f"### 발주의뢰 목록 ({len(df)}건) - t+{horizon} 예측")

    # 테이블 스타일 CSS
    st.markdown("""
    <style>
        /* Streamlit 기본 여백 줄이기 */
        .stVerticalBlock {
            gap: 0.3rem !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0 !important;
        }
        /* 텍스트 크기 */
        .stMarkdown p, .stText, div[data-testid="stText"] {
            font-size: 16px !important;
        }
        div[data-testid="column"] > div > div > div > div {
            font-size: 16px !important;
        }
        .stTextInput input {
            font-size: 16px !important;
        }
        /* number_input에서 +/- 버튼 숨기기 및 너비 조절 */
        .stNumberInput button {
            display: none !important;
        }
        .stNumberInput {
            width: 80px !important;
        }
        .stNumberInput > div {
            width: 80px !important;
        }
        .stNumberInput input {
            font-size: 16px !important;
            text-align: center !important;
            width: 80px !important;
            padding: 4px 8px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # session_state 초기화
    if 'expanded_rows' not in st.session_state:
        st.session_state.expanded_rows = set()
    if 'chat_rows' not in st.session_state:
        st.session_state.chat_rows = set()
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = {}
    if 'initial_report_sent' not in st.session_state:
        st.session_state.initial_report_sent = set()
    if 'order_quantities' not in st.session_state:
        st.session_state.order_quantities = {row['단품코드']: 0 for _, row in df.iterrows()}

    # 테이블 헤더
    # 순번, 단품코드, 단품명, 단위, 의뢰수량, 예측값(p50), 하한값(p10), 상한값(p90), 전일판매량, 주평균, 주요영향변수, 챗봇, 비고
    header_cols = st.columns([0.4, 0.9, 1.8, 0.5, 0.8, 0.8, 0.8, 0.8, 0.7, 0.7, 1.5, 0.8, 1.2])
    headers = [
        '순번', '단품코드', '단품명', '단위', '의뢰\n수량',
        '예측값\n(p50)', '하한값\n(p10)', '상한값\n(p90)',
        '전일\n판매량', '주평균\n판매량', '주요 영향 변수', '챗봇', '비고'
    ]

    for col, header in zip(header_cols, headers):
        col.markdown(f"**{header}**", unsafe_allow_html=True)

    st.markdown("---")

    # 각 행 렌더링
    updated_df = df.copy()

    for idx, row in df.iterrows():
        sku_code = row['단품코드']

        # 메인 행 (13개 컬럼)
        cols = st.columns([0.4, 0.9, 1.8, 0.5, 0.8, 0.8, 0.8, 0.8, 0.7, 0.7, 1.5, 0.8, 1.2])

        # 순번
        cols[0].write(row['순번'])

        # 단품코드
        cols[1].write(sku_code)

        # 단품명
        cols[2].write(row['단품명'])

        # 단위
        cols[3].write(row['단위'])

        # 의뢰수량 (editable)
        order_qty = cols[4].number_input(
            label=f"qty_{sku_code}",
            label_visibility="collapsed",
            min_value=0,
            max_value=9999,
            value=st.session_state.order_quantities.get(sku_code, 0),
            step=1,
            key=f"order_qty_{sku_code}"
        )
        st.session_state.order_quantities[sku_code] = order_qty
        updated_df.at[idx, '의뢰수량'] = order_qty

        # 예측값(p50) (강조)
        cols[5].markdown(f"**:blue[{row['예측값(p50)']}]**")

        # 하한값(p10)
        cols[6].write(row['하한값(p10)'])

        # 상한값(p90)
        cols[7].write(row['상한값(p90)'])

        # 전일 판매량 (lag_1) - _row_data에서 추출
        row_data = row.get('_row_data', {})
        lag_1 = row_data.get('lag_1', '-')
        cols[8].write(lag_1)

        # 주평균 판매량 (rolling_mean_6) - _row_data에서 추출
        rolling_mean_6 = row_data.get('rolling_mean_6', '-')
        if isinstance(rolling_mean_6, float):
            rolling_mean_6 = round(rolling_mean_6, 1)
        cols[9].write(rolling_mean_6)

        # 주요 영향 변수 (Top 3) - 줄바꿈 처리
        top_features = row['주요 영향 변수']
        if isinstance(top_features, str) and ', ' in top_features:
            top_features = top_features.replace(', ', ',\n')
        cols[10].write(top_features)

        # 챗봇 토글
        chat_btn = cols[11].button("🤖 AI", key=f"chat_{sku_code}", help="AI 챗봇 열기")
        if chat_btn:
            if sku_code in st.session_state.chat_rows:
                st.session_state.chat_rows.remove(sku_code)
            else:
                st.session_state.chat_rows.add(sku_code)
                # 채팅 기록 초기화
                if sku_code not in st.session_state.chat_messages:
                    st.session_state.chat_messages[sku_code] = []

        # 비고
        note = cols[12].text_input(
            label=f"note_{sku_code}",
            label_visibility="collapsed",
            value=row['비고'],
            key=f"note_{sku_code}",
            placeholder="메모..."
        )
        updated_df.at[idx, '비고'] = note

        # 챗봇 확장 영역
        if sku_code in st.session_state.chat_rows:
            render_chat_interface(sku_code, row['단품명'], row_data, horizon, prediction_date)

        st.markdown("---")

    # 하단 집계
    render_footer(updated_df)

    return updated_df


def render_chat_interface(sku_code: str, sku_name: str, row_data: dict, horizon: int = 1, prediction_date: str = None):
    """
    SKU별 챗봇 인터페이스 렌더링

    Parameters
    ----------
    sku_code : str
        단품코드
    sku_name : str
        단품명
    row_data : dict
        해당 행의 원본 데이터 (210_results 테이블 데이터)
    horizon : int
        예측 horizon (1~4)
    prediction_date : str
        예측 기준일 (YYYY-MM-DD 형식)
    """
    # 세션 ID 초기화
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

    # 챗봇 인스턴스
    chatbot = get_chatbot()

    with st.container():
        st.markdown(f"""
        <div style="background-color: #e8f4ea; padding: 15px; border-radius: 10px; margin: 10px 0;">
        """, unsafe_allow_html=True)

        st.markdown(f"#### AI {sku_name} 어시스턴트")
        st.caption("수요 예측에 대해 질문하거나, 발주량 조정 시나리오를 물어보세요.")

        # 초기 리포트 생성 (챗봇 처음 열 때만)
        report_key = f"{sku_code}_{horizon}"
        if report_key not in st.session_state.initial_report_sent:
            st.session_state.initial_report_sent.add(report_key)

            # 초기 리포트 생성
            initial_report = chatbot.generate_initial_report(row_data, sku_name, horizon)

            if sku_code not in st.session_state.chat_messages:
                st.session_state.chat_messages[sku_code] = []

            st.session_state.chat_messages[sku_code].append({
                'role': 'assistant',
                'content': initial_report
            })

        # 채팅 기록 표시
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_messages.get(sku_code, []):
                if msg['role'] == 'user':
                    st.markdown(f"**나**: {msg['content']}")
                else:
                    st.markdown(f"**AI**: {msg['content']}")

        # 입력 영역
        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.text_input(
                label=f"chat_input_{sku_code}",
                label_visibility="collapsed",
                placeholder="질문을 입력하세요...",
                key=f"chat_input_{sku_code}"
            )

        with col2:
            send_btn = st.button("전송", key=f"send_{sku_code}")

        if send_btn and user_input:
            # 사용자 메시지 추가
            if sku_code not in st.session_state.chat_messages:
                st.session_state.chat_messages[sku_code] = []

            st.session_state.chat_messages[sku_code].append({
                'role': 'user',
                'content': user_input
            })

            # AI 응답
            chat_history = st.session_state.chat_messages.get(sku_code, [])
            ai_response = chatbot.get_response(
                user_message=user_input,
                context=row_data,
                chat_history=chat_history[:-1]  # 현재 메시지 제외
            )

            st.session_state.chat_messages[sku_code].append({
                'role': 'assistant',
                'content': ai_response
            })

            # Supabase에 대화 저장
            if config.USE_SUPABASE:
                try:
                    save_chat_history(
                        store_cd='210',
                        sku_code=sku_code,
                        prediction_date=prediction_date or datetime.now().strftime('%Y-%m-%d'),
                        horizon=f't+{horizon}',
                        user_message=user_input,
                        assistant_message=ai_response,
                        session_id=st.session_state.session_id
                    )
                except Exception as e:
                    pass

            # 리렌더링
            st.rerun()

        # 예시 질문 버튼
        st.markdown("**빠른 질문:**")
        example_cols = st.columns(3)

        examples = chatbot.get_quick_suggestions()

        for i, (col, example) in enumerate(zip(example_cols, examples)):
            if col.button(example, key=f"example_{sku_code}_{i}"):
                if sku_code not in st.session_state.chat_messages:
                    st.session_state.chat_messages[sku_code] = []

                st.session_state.chat_messages[sku_code].append({
                    'role': 'user',
                    'content': example
                })

                chat_history = st.session_state.chat_messages.get(sku_code, [])
                ai_response = chatbot.get_response(
                    user_message=example,
                    context=row_data,
                    chat_history=chat_history[:-1]
                )

                st.session_state.chat_messages[sku_code].append({
                    'role': 'assistant',
                    'content': ai_response
                })

                # Supabase에 대화 저장
                if config.USE_SUPABASE:
                    try:
                        save_chat_history(
                            store_cd='210',
                            sku_code=sku_code,
                            prediction_date=prediction_date or datetime.now().strftime('%Y-%m-%d'),
                            horizon=f't+{horizon}',
                            user_message=example,
                            assistant_message=ai_response,
                            session_id=st.session_state.session_id
                        )
                    except Exception as e:
                        pass

                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def render_footer(df: pd.DataFrame):
    """
    하단 집계 영역 렌더링

    Parameters
    ----------
    df : pd.DataFrame
        발주 목록
    """
    st.markdown("### 집계")

    col1, col2, col3, col4 = st.columns(4)

    total_items = len(df)
    total_order_qty = df['의뢰수량'].sum()
    total_pred_qty = df['예측값(p50)'].sum()
    order_vs_pred = (total_order_qty / total_pred_qty * 100) if total_pred_qty > 0 else 0

    col1.metric("의뢰건수", f"{total_items}건")
    col2.metric("의뢰수량 합계", f"{total_order_qty:,}개")
    col3.metric("예측수량 합계", f"{total_pred_qty:,}개")
    col4.metric("발주율", f"{order_vs_pred:.1f}%",
                delta=f"{total_order_qty - total_pred_qty:+,}개" if total_order_qty != total_pred_qty else None)
