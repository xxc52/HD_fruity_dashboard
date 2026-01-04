"""
Admin Line Plot Component
==========================
예측 vs 실제 판매량 시각화 (Plotly)

- Coverage (p10-p90) band
- p50 예측값 (점)
- actual 실제값 (점)
- horizon=1 기준
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
from typing import Optional, List

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_predictions_for_month(store: str, month: str) -> pd.DataFrame:
    """
    특정 월의 예측 데이터 로드 (horizon=1만)

    Args:
        store: 점포 코드
        month: 월 (YYYY-MM)

    Returns:
        DataFrame with columns: date_t, sku, sku_name, p10, p50, p90, recent_2w_mean
    """
    outputs_path = PROJECT_ROOT / "outputs" / f"{store}.csv"

    if not outputs_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(outputs_path)
        df['date_t'] = pd.to_datetime(df['date_t'])
        df['month'] = df['date_t'].dt.strftime('%Y-%m')

        # horizon=1만 필터링
        df = df[(df['month'] == month) & (df['horizon'] == 1)]

        # recent_2w_mean 컬럼이 없으면 기본값 0
        if 'recent_2w_mean' not in df.columns:
            df['recent_2w_mean'] = 0.0

        return df[['date_t', 'sku', 'sku_name', 'p10', 'p50', 'p90', 'recent_2w_mean']].copy()
    except Exception as e:
        st.error(f"예측 데이터 로드 실패: {e}")
        return pd.DataFrame()


def load_actuals_for_month(store: str, month: str) -> pd.DataFrame:
    """
    특정 월의 실제 판매 데이터 로드

    Args:
        store: 점포 코드
        month: 월 (YYYY-MM)

    Returns:
        DataFrame with columns: date, sku, actual
    """
    raw_sales_path = PROJECT_ROOT / "data" / "raw_sales.csv"

    if not raw_sales_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(raw_sales_path)

        # STORE_CD 필터링
        df = df[df['STORE_CD'] == int(store)]

        # 날짜 변환 (YYYYMMDD → datetime)
        df['date'] = pd.to_datetime(df['SALE_DT'].astype(str), format='%Y%m%d')
        df['month'] = df['date'].dt.strftime('%Y-%m')

        # 월 필터링
        df = df[df['month'] == month]

        # SKU별 일별 판매량 집계
        df['sku'] = df['PRDT_CD'].astype(str)
        result = df.groupby(['date', 'sku']).agg({
            'SELL_QTY': 'sum'
        }).reset_index()

        result.columns = ['date', 'sku', 'actual']

        # 오늘 날짜 제외 (아직 actual 없음)
        today = datetime.now().date()
        result = result[result['date'].dt.date < today]

        return result
    except Exception as e:
        st.error(f"실제 판매 데이터 로드 실패: {e}")
        return pd.DataFrame()


def get_sku_options(df_pred: pd.DataFrame) -> List[str]:
    """
    SKU 선택 옵션 목록 생성

    정렬: sku_code 오름차순 (찾기 쉽게)
    형식: '202193-사과(93)' (2주 평균값 포함)
    """
    if df_pred.empty:
        return ["전체 합계"]

    # SKU별 recent_2w_mean 평균 (여러 날짜에 걸쳐 있을 수 있으므로)
    sku_stats = df_pred.groupby(['sku', 'sku_name']).agg({
        'recent_2w_mean': 'mean'
    }).reset_index()

    # 정렬: sku 오름차순 (찾기 쉽게)
    sku_stats = sku_stats.sort_values('sku', ascending=True)

    # 옵션 생성: '202193-사과(93)' 형식
    options = ["전체 합계"]
    for _, row in sku_stats.iterrows():
        mean_val = int(round(row['recent_2w_mean']))
        options.append(f"{row['sku']}-{row['sku_name']}({mean_val})")

    return options


def create_line_plot(
    df_pred: pd.DataFrame,
    df_actual: pd.DataFrame,
    selected_sku: str,
    month: str
) -> go.Figure:
    """
    Plotly line plot 생성

    Args:
        df_pred: 예측 데이터
        df_actual: 실제 판매 데이터
        selected_sku: 선택된 SKU ("전체 합계" 또는 "SKU코드 - SKU명")
        month: 월 (YYYY-MM)
    """
    fig = go.Figure()

    if df_pred.empty:
        fig.add_annotation(
            text="예측 데이터가 없습니다.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig

    # 데이터 필터링/집계
    if selected_sku == "전체 합계":
        # 전체 SKU 합계
        pred_agg = df_pred.groupby('date_t').agg({
            'p10': 'sum',
            'p50': 'sum',
            'p90': 'sum'
        }).reset_index()

        if not df_actual.empty:
            actual_agg = df_actual.groupby('date').agg({
                'actual': 'sum'
            }).reset_index()
        else:
            actual_agg = pd.DataFrame(columns=['date', 'actual'])
    else:
        # 개별 SKU - 형식: "SKU코드-SKU명(2주평균)"
        sku_code = selected_sku.split('-')[0]
        pred_agg = df_pred[df_pred['sku'].astype(str) == sku_code].copy()
        pred_agg = pred_agg.rename(columns={'date_t': 'date_t'})

        if not df_actual.empty:
            actual_agg = df_actual[df_actual['sku'] == sku_code].copy()
        else:
            actual_agg = pd.DataFrame(columns=['date', 'actual'])

    # 날짜 정렬
    if not pred_agg.empty:
        pred_agg = pred_agg.sort_values('date_t')

    if not actual_agg.empty:
        actual_agg = actual_agg.sort_values('date')

    # p10-p90 band (fill)
    if not pred_agg.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([pred_agg['date_t'], pred_agg['date_t'][::-1]]),
            y=pd.concat([pred_agg['p90'], pred_agg['p10'][::-1]]),
            fill='toself',
            fillcolor='rgba(100, 149, 237, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='Coverage (p10-p90)'
        ))

        # p50 예측값 (점)
        fig.add_trace(go.Scatter(
            x=pred_agg['date_t'],
            y=pred_agg['p50'],
            mode='markers+lines',
            marker=dict(size=8, color='#4169E1'),
            line=dict(color='#4169E1', width=1, dash='dot'),
            name='예측 (p50)',
            hovertemplate='%{x|%m/%d}<br>예측: %{y:.0f}개<extra></extra>'
        ))

    # actual 실제값 (점)
    if not actual_agg.empty:
        fig.add_trace(go.Scatter(
            x=actual_agg['date'],
            y=actual_agg['actual'],
            mode='markers+lines',
            marker=dict(size=8, color='#FF6347', symbol='diamond'),
            line=dict(color='#FF6347', width=2),
            name='실제',
            hovertemplate='%{x|%m/%d}<br>실제: %{y:.0f}개<extra></extra>'
        ))

    # 레이아웃 설정
    fig.update_layout(
        title=dict(
            text=f"예측 vs 실제 ({month})",
            font=dict(size=16)
        ),
        xaxis=dict(
            title="날짜",
            tickformat="%m/%d",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title="판매량 (개)",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified',
        height=400,
        margin=dict(l=50, r=20, t=60, b=50)
    )

    return fig


def render_line_plot(store: str, month: str):
    """Line Plot 컴포넌트 렌더링"""
    # 데이터 로드
    df_pred = load_predictions_for_month(store, month)
    df_actual = load_actuals_for_month(store, month)

    # SKU 선택
    sku_options = get_sku_options(df_pred)
    selected_sku = st.selectbox(
        "SKU 선택",
        options=sku_options,
        index=0,
        key="admin_sku_select"
    )
    st.caption("괄호 안 숫자는 최근 2주 일평균 판매량")

    # 선택된 SKU를 session_state에 저장 (모델 정보에서 사용)
    st.session_state['selected_sku_for_model_info'] = selected_sku

    # 차트 생성 및 표시
    fig = create_line_plot(df_pred, df_actual, selected_sku, month)
    st.plotly_chart(fig)