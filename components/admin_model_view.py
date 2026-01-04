"""
Admin Model View Component
===========================
모델 관리 페이지 메인 뷰
- 점포/월 선택
- Line Plot + 모델 정보 (2열 레이아웃)
- 성능 지표
- 파이프라인 상태 (토글)
- 스케줄 캘린더 (토글)
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Tuple, List

PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_available_months(store: str) -> List[str]:
    """
    outputs CSV에서 사용 가능한 월 목록 추출

    Returns:
        ['2026-01', '2025-12', ...] 형태의 리스트 (최신순)
    """
    outputs_path = PROJECT_ROOT / "outputs" / f"{store}.csv"

    if not outputs_path.exists():
        return []

    try:
        df = pd.read_csv(outputs_path, usecols=['date_t'])
        df['month'] = pd.to_datetime(df['date_t']).dt.strftime('%Y-%m')
        months = sorted(df['month'].unique(), reverse=True)
        return months
    except Exception:
        return []


def get_raw_sales_last_update() -> Optional[datetime]:
    """raw_sales.csv 최종 수정 시간 반환"""
    raw_sales_path = PROJECT_ROOT / "data" / "raw_sales.csv"

    if not raw_sales_path.exists():
        return None

    try:
        mtime = raw_sales_path.stat().st_mtime
        return datetime.fromtimestamp(mtime)
    except Exception:
        return None


def render_store_month_selector() -> Tuple[str, str]:
    """
    점포 및 월 선택 UI 렌더링

    Returns:
        (store, month) 튜플 (예: ('210', '2026-01'))
    """
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        store = st.selectbox(
            "점포 선택",
            options=['210', '220', '480'],
            index=0,
            key="admin_store_select"
        )

    # 선택된 점포의 사용 가능한 월 목록
    available_months = get_available_months(store)

    with col2:
        if available_months:
            month = st.selectbox(
                "월 선택",
                options=available_months,
                index=0,
                key="admin_month_select"
            )
        else:
            month = datetime.now().strftime('%Y-%m')
            st.selectbox(
                "월 선택",
                options=[month],
                index=0,
                disabled=True,
                key="admin_month_select"
            )
            st.caption("데이터 없음")

    with col3:
        # 데이터 갱신 타임스탬프
        last_update = get_raw_sales_last_update()
        if last_update:
            st.markdown(
                f"<div style='text-align: right; color: #666; font-size: 0.85em; padding-top: 28px;'>"
                f"판매 데이터 최종 업데이트: {last_update.strftime('%Y-%m-%d %H:%M')}"
                f"</div>",
                unsafe_allow_html=True
            )

    return store, month


def render_admin_model_view():
    """모델 관리 페이지 메인 렌더링"""
    from components.admin_line_plot import render_line_plot
    from components.admin_model_info import render_model_info
    from components.admin_performance import render_performance_metrics
    from components.admin_pipeline_status import render_pipeline_status
    from components.admin_calendar import render_schedule_calendar

    st.title("⚙️ 모델 관리")
    st.markdown("---")

    # 1. 점포/월 선택
    store, month = render_store_month_selector()

    st.markdown("---")

    # 2. 파이프라인 상태 (날짜 선택 포함)
    render_pipeline_status(store)

    st.markdown("---")

    # 3. 메인 영역 (2열 레이아웃)
    col_left, col_right = st.columns([6, 4])

    with col_left:
        render_line_plot(store, month)

    with col_right:
        render_model_info(store)

    st.markdown("---")

    # 4. 성능 지표
    render_performance_metrics(store, month)

    st.markdown("---")

    # 5. 스케줄 캘린더 (토글)
    render_schedule_calendar(store, month)