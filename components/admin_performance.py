"""
Admin Performance Component
============================
월간 성능 지표 계산 및 표시
- MAE, RMSE, Coverage
- SKU별 성능 분포
- SKU별 성능 경고 시스템
- 모델 재학습 권장 알림
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent

# 성능 경고 임계값
COVERAGE_WARNING_THRESHOLD = 0.7  # Coverage 70% 미만이면 경고
MAE_RATIO_THRESHOLD = 0.3  # MAE / recent_2w_mean >= 30% 이면 경고
MAE_ABSOLUTE_THRESHOLD = 3.0  # recent_2w_mean < 1 일 때 fallback (MAE > 3)


def load_predictions_for_month(store: str, month: str) -> pd.DataFrame:
    """특정 월의 예측 데이터 로드 (horizon=1)"""
    outputs_path = PROJECT_ROOT / "outputs" / f"{store}.csv"

    if not outputs_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(outputs_path)
        df['date_t'] = pd.to_datetime(df['date_t'])
        df['month'] = df['date_t'].dt.strftime('%Y-%m')

        # horizon=1만
        df = df[(df['month'] == month) & (df['horizon'] == 1)]

        # recent_2w_mean 컬럼이 없으면 기본값 0
        if 'recent_2w_mean' not in df.columns:
            df['recent_2w_mean'] = 0.0

        return df[['date_t', 'sku', 'sku_name', 'p10', 'p50', 'p90', 'recent_2w_mean']].copy()
    except Exception:
        return pd.DataFrame()


def load_actuals_for_month(store: str, month: str) -> pd.DataFrame:
    """특정 월의 실제 판매 데이터 로드"""
    raw_sales_path = PROJECT_ROOT / "data" / "raw_sales.csv"

    if not raw_sales_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(raw_sales_path)
        df = df[df['STORE_CD'] == int(store)]

        df['date'] = pd.to_datetime(df['SALE_DT'].astype(str), format='%Y%m%d')
        df['month'] = df['date'].dt.strftime('%Y-%m')
        df = df[df['month'] == month]

        df['sku'] = df['PRDT_CD'].astype(str)
        result = df.groupby(['date', 'sku']).agg({'SELL_QTY': 'sum'}).reset_index()
        result.columns = ['date', 'sku', 'actual']

        # 오늘 제외
        today = datetime.now().date()
        result = result[result['date'].dt.date < today]

        return result
    except Exception:
        return pd.DataFrame()


def calculate_metrics(df_pred: pd.DataFrame, df_actual: pd.DataFrame) -> Dict:
    """
    전체 성능 지표 계산

    Returns:
        {
            'mae': float,
            'rmse': float,
            'coverage': float,
            'n_samples': int,
            'date_range': str
        }
    """
    if df_pred.empty or df_actual.empty:
        return {
            'mae': None,
            'rmse': None,
            'coverage': None,
            'n_samples': 0,
            'date_range': 'N/A'
        }

    # 예측-실제 조인
    df_pred['date'] = df_pred['date_t']
    df_pred['sku'] = df_pred['sku'].astype(str)
    df_actual['sku'] = df_actual['sku'].astype(str)

    merged = pd.merge(
        df_pred,
        df_actual,
        on=['date', 'sku'],
        how='inner'
    )

    if merged.empty:
        return {
            'mae': None,
            'rmse': None,
            'coverage': None,
            'n_samples': 0,
            'date_range': 'N/A'
        }

    # 지표 계산
    errors = merged['p50'] - merged['actual']
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    # Coverage: actual이 p10-p90 범위 내인 비율
    in_range = (merged['actual'] >= merged['p10']) & (merged['actual'] <= merged['p90'])
    coverage = in_range.mean()

    # 날짜 범위
    min_date = merged['date'].min().strftime('%m/%d')
    max_date = merged['date'].max().strftime('%m/%d')

    return {
        'mae': mae,
        'rmse': rmse,
        'coverage': coverage,
        'n_samples': len(merged),
        'date_range': f"{min_date} ~ {max_date}"
    }


def calculate_sku_metrics(df_pred: pd.DataFrame, df_actual: pd.DataFrame) -> pd.DataFrame:
    """
    SKU별 성능 지표 계산

    Returns:
        DataFrame with columns: sku, sku_name, recent_2w_mean, mae, coverage, n_samples, warning, warning_severity
    """
    if df_pred.empty or df_actual.empty:
        return pd.DataFrame()

    df_pred['date'] = df_pred['date_t']
    df_pred['sku'] = df_pred['sku'].astype(str)
    df_actual['sku'] = df_actual['sku'].astype(str)

    merged = pd.merge(
        df_pred,
        df_actual,
        on=['date', 'sku'],
        how='inner'
    )

    if merged.empty:
        return pd.DataFrame()

    results = []
    for sku, group in merged.groupby('sku'):
        errors = group['p50'] - group['actual']
        mae = np.mean(np.abs(errors))

        in_range = (group['actual'] >= group['p10']) & (group['actual'] <= group['p90'])
        coverage = in_range.mean()

        sku_name = group['sku_name'].iloc[0] if 'sku_name' in group.columns else ''
        recent_2w_mean = group['recent_2w_mean'].mean() if 'recent_2w_mean' in group.columns else 0.0

        # 경고 판정 및 심각도 계산
        warning = []
        warning_severity = 0
        has_coverage_warning = coverage < COVERAGE_WARNING_THRESHOLD

        # MAE 경고: 상대적 기준 (MAE / recent_2w_mean >= 30%)
        if recent_2w_mean >= 1:
            mae_ratio = mae / recent_2w_mean
            has_mae_warning = mae_ratio >= MAE_RATIO_THRESHOLD
        else:
            # recent_2w_mean이 너무 작으면 절대 기준 사용
            has_mae_warning = mae > MAE_ABSOLUTE_THRESHOLD

        if has_coverage_warning:
            warning.append('Coverage 낮음')
        if has_mae_warning:
            warning.append('MAE 높음')

        # 심각도: 2=둘 다, 1=Coverage만, 0=MAE만
        if has_coverage_warning and has_mae_warning:
            warning_severity = 2
        elif has_coverage_warning:
            warning_severity = 1
        elif has_mae_warning:
            warning_severity = 0

        results.append({
            'sku': sku,
            'sku_name': sku_name,
            'recent_2w_mean': round(recent_2w_mean, 1),
            'mae': round(mae, 2),
            'coverage': round(coverage * 100, 1),
            'n_samples': len(group),
            'warning': ', '.join(warning) if warning else '',
            'warning_severity': warning_severity if warning else -1,
        })

    return pd.DataFrame(results)


def render_performance_metrics(store: str, month: str):
    """성능 지표 컴포넌트 렌더링"""
    # 데이터 로드
    df_pred = load_predictions_for_month(store, month)
    df_actual = load_actuals_for_month(store, month)

    # 전체 지표 계산
    metrics = calculate_metrics(df_pred, df_actual)

    st.markdown(f"### 📊 {month} 모델 성능")

    if metrics['n_samples'] == 0:
        st.info("해당 월에 비교 가능한 데이터가 없습니다.")
        return

    st.caption(f"기간: {metrics['date_range']} ({metrics['n_samples']}건)")

    # 지표 표시
    col1, col2, col3 = st.columns(3)

    with col1:
        if metrics['mae'] is not None:
            st.metric("MAE", f"{metrics['mae']:.2f}개")
        else:
            st.metric("MAE", "N/A")

    with col2:
        if metrics['rmse'] is not None:
            st.metric("RMSE", f"{metrics['rmse']:.2f}개")
        else:
            st.metric("RMSE", "N/A")

    with col3:
        if metrics['coverage'] is not None:
            coverage_pct = metrics['coverage'] * 100
            delta_color = "normal" if coverage_pct >= 80 else "inverse"
            st.metric(
                "Coverage (p10-p90)",
                f"{coverage_pct:.1f}%",
                delta=f"{'✓ 양호' if coverage_pct >= 80 else '주의'}",
                delta_color=delta_color
            )
        else:
            st.metric("Coverage (p10-p90)", "N/A")

    # SKU별 분석
    st.markdown("#### 📋 SKU별 성능 분석")
    st.caption("경고 기준: Coverage < 70% / MAE ≥ 2주평균의 30%")

    df_sku = calculate_sku_metrics(df_pred, df_actual)

    if df_sku.empty:
        st.info("SKU별 데이터가 없습니다.")
    else:
        # 경고 SKU 하이라이트
        warning_skus = df_sku[df_sku['warning'] != ''].copy()
        if not warning_skus.empty:
            # 정렬: 경고 심각도 내림차순 → Coverage 오름차순 → 2주평균 내림차순
            warning_skus = warning_skus.sort_values(
                ['warning_severity', 'coverage', 'recent_2w_mean'],
                ascending=[False, True, False]
            )
            st.markdown("**⚠️ 주의 필요 SKU**")
            st.dataframe(
                warning_skus[['sku', 'sku_name', 'recent_2w_mean', 'mae', 'coverage', 'warning']].rename(columns={
                    'sku': 'SKU',
                    'sku_name': '상품명',
                    'recent_2w_mean': '2주 평균',
                    'mae': 'MAE',
                    'coverage': 'Coverage(%)',
                    'warning': '경고'
                }),
                hide_index=True
            )

        # 전체 SKU 테이블 (SKU 오름차순, 사용자가 테이블에서 직접 정렬 가능)
        st.markdown("**전체 SKU 성능**")
        df_all = df_sku.sort_values('sku', ascending=True)
        st.dataframe(
            df_all[['sku', 'sku_name', 'recent_2w_mean', 'mae', 'coverage', 'n_samples']].rename(columns={
                'sku': 'SKU',
                'sku_name': '상품명',
                'recent_2w_mean': '2주 평균',
                'mae': 'MAE',
                'coverage': 'Coverage(%)',
                'n_samples': '샘플 수'
            }),
            hide_index=True
        )