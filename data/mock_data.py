"""
Mock Data Generator
===================
대시보드 테스트용 더미 데이터 생성
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

# Top 10 SKU 더미 예측 데이터
MOCK_PREDICTIONS = {
    '269211': {  # 특선바나나
        'base_qty': 150,
        'model': 'XGBoost',
        'short_reason': '주말 수요 증가 예상',
        'full_report': """
### 특선바나나 수요 예측 리포트

**예측 모델**: XGBoost (RMSE: 38.16, MAPE: 17.83%)

**주요 영향 요인**:
1. **요일 효과**: 토요일은 평일 대비 평균 +23% 판매량
2. **날씨**: 맑음 예보, 외출 증가로 매장 방문객 상승 예상
3. **시즌**: 11월 말 바나나 성수기 진입

**최근 트렌드**:
- 7일 평균: 142개
- 전주 동요일: 158개
- 전년 동기: 145개

**권장 발주량**: 150~165개
"""
    },
    '202309': {  # 국산청포도
        'base_qty': 45,
        'model': 'LightGBM',
        'short_reason': '시즌 종료, 수요 감소',
        'full_report': """
### 국산청포도 수요 예측 리포트

**예측 모델**: LightGBM (RMSE: 42.31, MAPE: 21.5%)

**주요 영향 요인**:
1. **시즌성**: 청포도 시즌 종료기, 수요 하락세
2. **가격**: 최근 단가 상승으로 구매 감소
3. **대체재**: 샤인머스캣으로 수요 이동

**최근 트렌드**:
- 7일 평균: 52개
- 전주 동요일: 48개
- 전년 동기: 65개 (↓26%)

**권장 발주량**: 40~50개 (재고 소진 우선)
"""
    },
    '400189': {  # 금실딸기
        'base_qty': 85,
        'model': 'XGBoost',
        'short_reason': '딸기 시즌 시작, 급상승',
        'full_report': """
### 금실딸기 수요 예측 리포트

**예측 모델**: XGBoost (RMSE: 35.22, MAPE: 15.8%)

**주요 영향 요인**:
1. **시즌 시작**: 11월 말 딸기 시즌 본격 개시
2. **품질**: 금실 품종 인기 상승 중
3. **프로모션**: 이번 주 카드사 할인 행사

**최근 트렌드**:
- 7일 평균: 72개
- 전주 동요일: 78개
- 전년 동기: 68개 (↑25%)

**권장 발주량**: 85~95개 (적극 발주 권장)
"""
    },
    '400220': {  # 죽향 딸기
        'base_qty': 65,
        'model': 'RandomForest',
        'short_reason': '프리미엄 수요 안정적',
        'full_report': """
### 죽향 딸기 수요 예측 리포트

**예측 모델**: RandomForest (RMSE: 28.45, MAPE: 18.2%)

**주요 영향 요인**:
1. **프리미엄 포지션**: 고정 고객층 확보
2. **선물 수요**: 연말 선물세트 수요 증가
3. **품질 안정**: 공급처 품질 일정

**최근 트렌드**:
- 7일 평균: 62개
- 전주 동요일: 64개
- 전년 동기: 58개 (↑12%)

**권장 발주량**: 60~70개
"""
    },
    '400053': {  # 사과(4입)
        'base_qty': 120,
        'model': 'XGBoost',
        'short_reason': '사과 성수기, 꾸준한 수요',
        'full_report': """
### 사과(4입) 수요 예측 리포트

**예측 모델**: XGBoost (RMSE: 45.12, MAPE: 19.3%)

**주요 영향 요인**:
1. **시즌**: 11월 사과 최성수기
2. **포장 단위**: 4입 소포장 선호도 높음
3. **가격 경쟁력**: 경쟁사 대비 5% 저렴

**최근 트렌드**:
- 7일 평균: 115개
- 전주 동요일: 122개
- 전년 동기: 110개 (↑9%)

**권장 발주량**: 115~125개
"""
    },
    '400293': {  # 블루베리(200g)
        'base_qty': 95,
        'model': 'LightGBM',
        'short_reason': '건강식품 트렌드 지속',
        'full_report': """
### 블루베리(200g) 수요 예측 리포트

**예측 모델**: LightGBM (RMSE: 32.18, MAPE: 16.7%)

**주요 영향 요인**:
1. **건강 트렌드**: 항산화 식품 인기 지속
2. **아침식사 대용**: 요거트 토핑용 수요
3. **수입 안정**: 칠레산 공급 원활

**최근 트렌드**:
- 7일 평균: 92개
- 전주 동요일: 95개
- 전년 동기: 82개 (↑16%)

**권장 발주량**: 90~100개
"""
    },
    '400342': {  # 국내산 블루베리
        'base_qty': 55,
        'model': 'SARIMAX',
        'short_reason': '시즌 오프, 최소 발주',
        'full_report': """
### 국내산 블루베리 수요 예측 리포트

**예측 모델**: SARIMAX (RMSE: 28.92, MAPE: 22.1%)

**주요 영향 요인**:
1. **비시즌**: 국내산 블루베리 비수기
2. **가격**: 수입산 대비 2배 가격
3. **재고**: 냉동 재고 소진 필요

**최근 트렌드**:
- 7일 평균: 48개
- 전주 동요일: 52개
- 전년 동기: 55개 (동일)

**권장 발주량**: 50~60개 (최소 유지)
"""
    },
    '202284': {  # 불로초감귤
        'base_qty': 75,
        'model': 'Holt-Winters',
        'short_reason': '감귤 시즌, 수요 증가',
        'full_report': """
### 불로초감귤 수요 예측 리포트

**예측 모델**: Holt-Winters (RMSE: 38.55, MAPE: 20.3%)

**주요 영향 요인**:
1. **시즌 피크**: 11월 감귤 최성수기
2. **브랜드 인지도**: 불로초 브랜드 충성 고객
3. **당도 보장**: 품질 관리 우수

**최근 트렌드**:
- 7일 평균: 71개
- 전주 동요일: 74개
- 전년 동기: 68개 (↑10%)

**권장 발주량**: 70~80개
"""
    },
    '202193': {  # 사과
        'base_qty': 180,
        'model': 'XGBoost',
        'short_reason': '벌크 사과 대량 수요',
        'full_report': """
### 사과(벌크) 수요 예측 리포트

**예측 모델**: XGBoost (RMSE: 52.33, MAPE: 18.9%)

**주요 영향 요인**:
1. **대용량 선호**: 가정용 벌크 수요 증가
2. **가격 메리트**: kg당 최저가
3. **시즌**: 사과 성수기

**최근 트렌드**:
- 7일 평균: 175개
- 전주 동요일: 182개
- 전년 동기: 165개 (↑9%)

**권장 발주량**: 175~190개
"""
    },
    '202599': {  # 골드키위
        'base_qty': 88,
        'model': 'LightGBM',
        'short_reason': '뉴질랜드산 입고, 안정',
        'full_report': """
### 골드키위 수요 예측 리포트

**예측 모델**: LightGBM (RMSE: 29.87, MAPE: 17.2%)

**주요 영향 요인**:
1. **수입 안정**: 뉴질랜드산 정기 입고
2. **비타민 수요**: 겨울철 비타민C 수요
3. **단가 안정**: 환율 영향 최소화

**최근 트렌드**:
- 7일 평균: 85개
- 전주 동요일: 88개
- 전년 동기: 80개 (↑10%)

**권장 발주량**: 85~95개
"""
    }
}


def get_predictions_df(
    base_date: datetime,
    order_date: datetime,
    store_id: str = '210'
) -> pd.DataFrame:
    """
    발주 예측 데이터프레임 생성 (Mock 데이터)

    Parameters
    ----------
    base_date : datetime
        오늘 날짜 (t 시점)
    order_date : datetime
        발주일 (t+n 시점)
    store_id : str
        점포 코드

    Returns
    -------
    pd.DataFrame
        예측 데이터가 포함된 발주 목록 (새 컬럼 구조)
    """
    from config import TOP_10_SKUS, CONFIDENCE_INTERVAL

    # horizon 계산
    horizon = (order_date - base_date).days
    if horizon < 1:
        horizon = 1
    elif horizon > 4:
        horizon = 4

    rows = []
    for i, sku in enumerate(TOP_10_SKUS, 1):
        code = sku['code']
        pred_info = MOCK_PREDICTIONS.get(code, {})

        # 기본 예측값 (horizon에 따라 약간의 변동)
        base_qty = pred_info.get('base_qty', 100)
        # horizon이 길어질수록 불확실성 증가
        noise = np.random.normal(0, base_qty * 0.05 * horizon)
        p50 = int(base_qty + noise)

        # p10, p90 계산
        p10 = int(p50 * (1 - CONFIDENCE_INTERVAL * 2))
        p90 = int(p50 * (1 + CONFIDENCE_INTERVAL * 2))

        # Mock row_data for chatbot
        mock_row_data = {
            'sku_code': code,
            'sku_name': sku['name'],
            'p10': p10,
            'p50': p50,
            'p90': p90,
            'model_name': pred_info.get('model', 'XGBoost'),
            'val_rmse': round(np.random.uniform(20, 50), 2),
            'val_train_rmse': round(np.random.uniform(15, 40), 2),
            'n_train_samples': np.random.randint(300, 500),
            'top_3_features': '["lag_1", "rolling_mean_6", "is_weekend_t+1"]',
            'lag_1': np.random.randint(50, 200),
            'rolling_mean_6': round(np.random.uniform(80, 180), 2),
            'lag': '{"lag_1": 120, "lag_2": 115}',
            'rolling': '{"rolling_mean_6": 125.5, "rolling_std_6": 15.2}',
            'weather': '{"TEMP_AVG_t+1": 15.5, "HM_AVG_t+1": 55.0, "RN_DAY_t+1": 0.0}',
            'holiday': '{"is_weekend_t+1": 0, "is_hd_holiday_t+1": 0}',
            'shap_values': '{"lag_1": 0.25, "rolling_mean_6": 0.18, "is_weekend_t+1": 0.12}',
            'hyperparameters': '{"n_estimators": 150, "max_depth": 6}'
        }

        rows.append({
            '순번': i,
            '단품코드': code,
            '단품명': sku['name'],
            '단위': sku['unit'],
            '의뢰수량': 0,
            '예측값(p50)': p50,
            '하한값(p10)': p10,
            '상한값(p90)': p90,
            '주요 영향 변수': pred_info.get('short_reason', 'lag_1, rolling_mean_6'),
            '비고': '',
            '_row_data': mock_row_data
        })

    return pd.DataFrame(rows)


def get_sku_chat_context(sku_code: str) -> str:
    """
    LLM 챗봇용 SKU 컨텍스트 생성

    Parameters
    ----------
    sku_code : str
        단품코드

    Returns
    -------
    str
        LLM에 전달할 컨텍스트
    """
    from config import TOP_10_SKUS

    sku_info = next((s for s in TOP_10_SKUS if s['code'] == sku_code), None)
    pred_info = MOCK_PREDICTIONS.get(sku_code, {})

    if not sku_info:
        return "해당 SKU 정보를 찾을 수 없습니다."

    context = f"""
당신은 현대백화점 청과 발주 담당자를 돕는 AI 어시스턴트입니다.

현재 상품 정보:
- 단품코드: {sku_code}
- 단품명: {sku_info['name']}
- 카테고리: {sku_info['category']}
- 누적 매출: {sku_info['total_sales']:,}원

예측 정보:
- 예측 모델: {pred_info.get('model', 'XGBoost')}
- 기본 예측량: {pred_info.get('base_qty', 100)}개
- 예측 근거: {pred_info.get('short_reason', '-')}

상세 리포트:
{pred_info.get('full_report', '리포트 없음')}

발주 담당자의 질문에 친절하고 전문적으로 답변해주세요.
수요 예측, 발주량 조정, 시장 트렌드 등에 대해 조언할 수 있습니다.
"""
    return context
