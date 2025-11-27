"""
Header Component
================
발주의뢰 대시보드 헤더 (필터 영역)
"""

import streamlit as st
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 상위 디렉토리 import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import STORES, DEPARTMENTS, PARTS, PC_CODES, CORNERS, ORDER_TYPES


def render_header() -> dict:
    """
    헤더 영역 렌더링

    Returns
    -------
    dict
        선택된 필터 값들
        {
            'store': str,
            'base_date': datetime,
            'order_date': datetime,
            'horizon': int,
            ...
        }
    """
    st.markdown("## 발주의뢰 등록 (신선식품/원테이블)")
    st.markdown("---")

    # 첫 번째 행: 점포, 팀, 파트, PC, 코너
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        store = st.selectbox(
            "점",
            options=list(STORES.keys()),
            format_func=lambda x: STORES[x],
            index=0
        )

    with col2:
        dept = st.selectbox(
            "팀",
            options=list(DEPARTMENTS.keys()),
            format_func=lambda x: DEPARTMENTS[x],
            index=0,
            disabled=True
        )

    with col3:
        part = st.selectbox(
            "파트",
            options=list(PARTS.keys()),
            format_func=lambda x: PARTS[x],
            index=0,
            disabled=True
        )

    with col4:
        pc = st.selectbox(
            "PC",
            options=list(PC_CODES.keys()),
            format_func=lambda x: PC_CODES[x],
            index=0,
            disabled=True
        )

    with col5:
        corner = st.selectbox(
            "코너",
            options=list(CORNERS.keys()),
            format_func=lambda x: CORNERS[x],
            index=0,
            disabled=True
        )

    # 두 번째 행: 오늘 날짜, 발주일, 발주유형, Horizon 표시
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        base_date = st.date_input(
            "오늘 날짜 (t)",
            value=datetime.now().date(),
            help="기준 날짜 (t 시점)"
        )

    with col2:
        # 기본값: 내일 (t+1)
        default_order_date = datetime.now().date() + timedelta(days=1)
        order_date = st.date_input(
            "발주일",
            value=default_order_date,
            min_value=base_date,
            max_value=base_date + timedelta(days=4),
            help="발주 대상 날짜"
        )

    with col3:
        order_type = st.selectbox(
            "발주유형",
            options=list(ORDER_TYPES.keys()),
            format_func=lambda x: ORDER_TYPES[x],
            index=0,
            disabled=True
        )

    with col4:
        # Horizon 자동 계산
        horizon = (order_date - base_date).days
        if horizon < 1:
            horizon = 1
        elif horizon > 4:
            horizon = 4

        st.metric(
            label="예측 Horizon",
            value=f"t+{horizon}",
            delta=f"{order_date.strftime('%Y-%m-%d')} 예측"
        )

    st.markdown("---")

    return {
        'store': store,
        'department': dept,
        'part': part,
        'pc': pc,
        'corner': corner,
        'base_date': datetime.combine(base_date, datetime.min.time()),
        'order_date': datetime.combine(order_date, datetime.min.time()),
        'order_type': order_type,
        'horizon': horizon
    }
