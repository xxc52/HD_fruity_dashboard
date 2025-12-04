# FRUITY Dashboard - 발주의뢰 시스템

현대백화점 청과 수요 예측 기반 **발주 지원 대시보드**.

---

## Overview

### 시스템 목적
- 발주 담당자가 AI 예측값을 참고하여 발주 수량 결정
- 예측값에 대한 의문점은 AI 챗봇으로 즉시 해소
- FRUITY 모델링 모듈과 완전 분리 (독립 배포)

### 핵심 기능
1. **예측값 조회**: Supabase에서 SKU별 t+1~t+4 예측 조회
2. **발주 입력**: 의뢰수량 직접 입력, 예측값과 비교
3. **상세 리포트**: SHAP 기반 예측 근거 확인
4. **AI 챗봇**: 예측값에 대한 Q&A (OpenAI GPT)
5. **대화 저장**: 발주 담당자 질문 로그 저장 (분석용)

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT CLOUD                          │
│                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │ app.py      │───▶│ Supabase     │───▶│ predictions_ │   │
│  │ (메인 UI)   │    │ Client       │    │ display/     │   │
│  └─────────────┘    └──────────────┘    │ context      │   │
│         │                               └──────────────┘   │
│         ▼                                                   │
│  ┌─────────────┐    ┌──────────────┐                       │
│  │ 챗봇 버튼   │───▶│ OpenAI GPT   │                       │
│  │ (context    │    │ (응답 생성)  │                       │
│  │  전달)      │    └──────────────┘                       │
│  └─────────────┘                                           │
│                                                             │
│  URL: https://xxc52-hd-dummy-dashboard-app-*.streamlit.app │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │           SUPABASE            │
              │                               │
              │  predictions_display (조회)   │
              │  predictions_context (챗봇)   │
              │  chat_history (대화 저장)     │
              └───────────────────────────────┘
```

---

## 디렉토리 구조

```
dashboard/
├── app.py                      # 메인 Streamlit 앱
├── config.py                   # 설정 (점포, 부서, USE_SUPABASE 등)
├── requirements.txt            # 배포용 의존성
│
├── components/
│   ├── header.py               # 헤더 (필터 영역)
│   ├── order_table.py          # 발주 테이블 + 챗봇 UI
│   └── chatbot.py              # AI 챗봇 로직
│
├── data/
│   ├── supabase_client.py      # Supabase 연동
│   └── mock_data.py            # Fallback용 Mock 데이터
│
└── .streamlit/
    └── secrets.toml            # Secrets (git 제외)
```

---

## UI 구성

### 헤더 (필터 영역)

| 필드 | 설명 |
|------|------|
| 점 | 점포 선택 (210 무역, 220 킨텍스, 480 더현대서울) |
| 팀/파트/PC/코너 | 신선식품 고정 |
| 오늘 날짜 (t) | 예측 기준일 (기본: 오늘) |
| 발주일 | 발주 대상일 (t+1 ~ t+4 선택) |
| 예측 Horizon | 자동 계산 (발주일 - 오늘) |

### 발주 테이블

| 컬럼 | 설명 |
|------|------|
| 순번 | 행 번호 |
| 단품코드 | SKU 코드 |
| 단품명 | 상품명 |
| 단위 | EA |
| **의뢰수량** | 발주 담당자 입력 (editable) |
| **예측값** | AI 예측 수량 (강조 표시) |
| 예측_min | 95% 신뢰구간 하한 |
| 예측_max | 95% 신뢰구간 상한 |
| 예측설명 | SHAP Top 1 변수 |
| 상세 | 📊 버튼 → 상세 리포트 |
| 챗봇 | 💬 버튼 → AI 챗봇 |
| 비고 | 메모 입력 |

### 챗봇 인터페이스

```
💬 특선바나나 AI 어시스턴트
────────────────────────────
수요 예측에 대해 질문하거나, 발주량 조정 시나리오를 물어보세요.
💡 대화는 서비스 개선을 위해 저장될 수 있습니다.

🧑 나: 왜 오늘 예측값이 평소보다 높아?
🤖 AI: 특선바나나의 t+1 예측값이 165개로 7일 평균(142개) 대비
      16% 높게 나왔습니다. 주요 영향 요인:
      1. 주말 효과 (is_weekend): 토요일은 평일 대비 +23%
      2. 날씨 (TEMP_AVG): 예보 기온 18도, 야외 활동 증가
      ...

[예측 근거가 뭐야?] [공격적 발주 시 리스크는?] [작년 대비 트렌드는?]
```

---

## Supabase 연동

### 테이블 조회

**predictions_display**: 대시보드 표시용
```python
# 필터: store_cd + prediction_date (= 오늘)
# 반환: sku_code, predicted_value, pred_min, pred_max, top_features, ...
```

**predictions_context**: 챗봇 context용
```python
# 필터: store_cd + sku_code + horizon
# 반환: 전체 context (SHAP, hyperparameters, recent_trend, feature_info)
```

### 데이터 저장

**chat_history**: 대화 로그
```python
save_chat_history(
    store_cd='210',
    sku_code='269211',
    prediction_date='2024-12-04',
    horizon='t+1',
    user_message='왜 예측값이 높아?',
    assistant_message='...',
    session_id='abc123'
)
```

---

## 챗봇 설계

### System Prompt

```python
SYSTEM_PROMPT = """
당신은 현대백화점 청과 수요 예측 시스템의 AI 어시스턴트입니다.
발주 담당자의 질문에 친절하고 전문적으로 답변합니다.

역할:
- 예측값의 근거 설명 (SHAP 기반)
- 발주량 조정 시나리오 분석
- 최근 트렌드 및 시즌성 정보 제공
- 모델 성능 및 신뢰구간 설명

답변 스타일:
- 간결하고 명확하게
- 숫자와 근거를 함께 제시
- 발주 담당자 관점에서 실용적인 조언
"""
```

### Context 전달 방식

챗봇 열 때 `predictions_context`의 전체 데이터를 한 번에 전달:

```
[현재 상품 정보]
- 단품코드: 269211
- 단품명: 특선바나나
- 예측일: 2024-12-04
- 예측 horizon: t+1

[예측 결과]
- 예측값: 165개
- 신뢰구간: 142 ~ 188개
- 예측 모델: LightGBM

[주요 영향 요인 - SHAP Top 10]
1. lag_1 (1일 전 판매량): 0.35
2. is_weekend (주말 여부): 0.22
...

[최근 판매 트렌드]
- 7일 평균: 142개
- 전주 대비: +5.2%

[t+1 예측에 사용된 변수 정보]
- 총 변수 수: 38개
- 날씨 변수: TEMP_AVG_t+1, HM_AVG_t+1, ...
...
```

### Feature 한글 매핑

```python
FEATURE_DESCRIPTIONS = {
    'rolling_mean_6': '최근 6일 평균 판매량',
    'lag_1': '1일 전 판매량',
    'lag_6': '6일 전 판매량 (1주 전)',
    'is_weekend_t+1': '주말 여부',
    'TEMP_AVG_t+1': '예보 평균 기온',
    'RN_DAY_t+1': '예보 강수 여부',
    # ...
}
```

---

## 배포

### 로컬 실행

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud 배포

1. GitHub repo 연결 (hd_dummy_dashboard)
2. Main file: `app.py`
3. Secrets 설정:
   ```toml
   [supabase]
   url = "https://xxx.supabase.co"
   key = "sb_xxx"

   [openai]
   api_key = "sk-proj-xxx"
   ```
4. Deploy

### 주의사항

- `secrets.toml`은 `.gitignore`에 포함됨 (git에 올라가지 않음)
- Streamlit Cloud에서는 웹 UI로 Secrets 설정 필요

---

## 설정 파일

### config.py

```python
# Supabase 사용 여부 (False면 mock_data 사용)
USE_SUPABASE = True

# 점포 정보
STORES = {
    '210': '210 무역',
    '220': '220 킨텍스',
    '480': '480 더현대서울'
}

# 부서/파트/PC/코너 (현재 고정값)
DEPARTMENTS = {'10': '신선식품팀'}
PARTS = {'01': '청과파트'}
# ...
```

---

## Fallback 처리

Supabase 연결 실패 시 `mock_data.py`의 더미 데이터 사용:

```python
if config.USE_SUPABASE:
    try:
        df = get_predictions_from_supabase(...)
    except Exception as e:
        st.warning(f"Supabase 연결 실패: {e}")
        df = None

if df is None or df.empty:
    df = get_predictions_df(...)  # Mock 데이터 fallback
```

---

## TODO / 개선 필요

### 기능
- [ ] 발주 확정 시 실제 저장 로직 구현
- [ ] 임시저장 기능 구현
- [ ] 발주 히스토리 조회 기능

### UI/UX
- [ ] 예측값 vs 의뢰수량 차이 시각화 (그래프)
- [ ] SKU별 과거 예측 정확도 표시
- [ ] 모바일 반응형 개선

### 챗봇
- [ ] 답변 품질 개선 (프롬프트 튜닝)
- [ ] 대화 컨텍스트 유지 개선
- [ ] 빠른 질문 버튼 커스터마이징
