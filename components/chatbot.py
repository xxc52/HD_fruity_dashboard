"""
AI Chatbot Module
=================

LLM API를 사용한 예측값 Q&A 챗봇 (Claude / Gemini 전환 가능)
"""

import streamlit as st
from typing import Dict, List, Optional
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# LLM 클라이언트 로드 (config 기반)
def get_llm_client():
    """설정에 따라 LLM 클라이언트 반환"""
    if config.LLM_PROVIDER == "claude":
        from .llm_claude import ClaudeClient
        return ClaudeClient()
    else:
        from .llm_gemini import GeminiClient
        return GeminiClient()


# Feature 한글 매핑 (*_future 패턴은 동적으로 처리)
FEATURE_DESCRIPTIONS = {
    # === Lag 피처 (fitting date 기준) ===
    'lag_1': '1일 전 판매량',
    'lag_2': '2일 전 판매량',

    # === Lag 피처 (prediction date 기준, 전주 동일요일) ===
    'lag_h-7': '1주 전 동일요일 판매량',
    'lag_h-14': '2주 전 동일요일 판매량',
    'lag_h-21': '3주 전 동일요일 판매량',
    'lag_h-28': '4주 전 동일요일 판매량',

    # === Rolling 피처 (이동 통계) ===
    'rolling_mean_6': '최근 6일 평균 판매량',
    'rolling_std_6': '최근 6일 판매량 표준편차',
    'rolling_max_6': '최근 6일 최대 판매량',
    'rolling_mean_27': '최근 27일 평균 판매량',
    'rolling_std_27': '최근 27일 판매량 표준편차',
    'rolling_max_27': '최근 27일 최대 판매량',

    # === SKU 인코딩 ===
    'SKU_encoded': '상품 고유 특성값',

    # === 날씨 피처 (*_future) ===
    'TEMP_AVG_future': '예측일 평균 기온(℃)',
    'HM_AVG_future': '예측일 평균 습도(%)',
    'WIND_AVG_future': '예측일 평균 풍속(m/s)',
    'RN_DAY_future': '예측일 강수량(mm)',

    # === 시간 사이클 피처 (*_future) ===
    'day_cos_future': '일(day) 주기 코사인',
    'day_sin_future': '일(day) 주기 사인',
    'month_cos_future': '월 주기 코사인',
    'month_sin_future': '월 주기 사인',
    'weekday_cos_future': '요일 주기 코사인',
    'weekday_sin_future': '요일 주기 사인',

    # === 휴일 피처 (*_future) ===
    'is_weekend_future': '예측일 금토일 여부',
    'is_hd_holiday_future': '예측일 현대백화점 휴무',
    'holiday_korea_t_future': '예측일 한국 공휴일',
    'holiday_korea_t-1_future': '예측일 전일 공휴일',
    'holiday_korea_t+1_future': '예측일 익일 공휴일',
    'holiday_christ_t_future': '예측일 크리스마스',
    'holiday_christ_t-1_future': '예측일 전일 크리스마스',
    'holiday_newyear_t_future': '예측일 신년',
    'holiday_newyear_t-1_future': '예측일 전일 신년',
    'holiday_etc_future': '예측일 기타 휴일',

    # === 가격/판매 피처 (t-1: 전일) ===
    'AVG_SELL_UPRC_t-1': '전일 평균 판매단가',
    'PRCH_QTY_t-1': '전일 매입수량',
    'RMRGN_RATE_t-1': '전일 매익률',
    'RMRGN_t-1': '전일 매익금액',
    'TR_CNT_t-1': '전일 거래건수',
    'TSALE_AMT_t-1': '전일 총판매금액',
}


def get_feature_description(feature_name: str) -> str:
    """피처명을 한글 설명으로 변환"""
    # 정확히 일치하면 바로 반환
    if feature_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature_name]

    # 휴일 관련 피처 (패턴 매칭)
    if 'holiday_korea' in feature_name:
        return '한국 공휴일 관련'
    if 'holiday_christ' in feature_name:
        return '크리스마스 관련'
    if 'holiday_newyear' in feature_name:
        return '신년 관련'
    if 'holiday_etc' in feature_name:
        return '기타 휴일'

    # 날씨 관련 피처
    if 'TEMP_AVG' in feature_name:
        return '평균 기온'
    if 'HM_AVG' in feature_name:
        return '평균 습도'
    if 'WIND_AVG' in feature_name:
        return '평균 풍속'
    if 'RN_DAY' in feature_name:
        return '강수량'

    # 시간 사이클 피처
    if 'weekday_cos' in feature_name or 'weekday_sin' in feature_name:
        return '요일 주기'
    if 'month_cos' in feature_name or 'month_sin' in feature_name:
        return '월 주기'
    if 'day_cos' in feature_name or 'day_sin' in feature_name:
        return '일 주기'

    # 주말/휴무
    if 'is_weekend' in feature_name:
        return '금토일 여부'
    if 'is_hd_holiday' in feature_name:
        return '현대백화점 휴무'

    # 못 찾으면 원본 반환
    return feature_name


# 초기 리포트 생성 프롬프트 (고정)
INITIAL_REPORT_PROMPT = """당신은 현대백화점 청과 수요 예측 시스템의 AI 어시스턴트입니다.
아래 예측 정보를 바탕으로 발주 담당자를 위한 리포트를 작성해주세요.

**중요: 발주 담당자는 데이터 전문가가 아닙니다. 쉬운 말로 작성하세요.**

**형식 규칙:**
- # 또는 ## 헤더 금지
- 범위 표시: 물결표(~) 대신 하이픈(-) 사용 (예: 65-85개)
- 취소선(~~) 금지
- 기술 용어 금지: MAE, RMSE, SHAP, feature, rolling 등 → 쉬운 말로 바꿔 쓰기
- 변수명 표기: 한글 설명 먼저, 영문은 괄호 안에 (예: "최근 6일 평균 판매량(rolling_mean_6)")

다음 형식으로 작성하세요:

**[예측 모델 정보]**
- 사용 모델명과 예측 정확도를 쉬운 말로 (예: "평균 7개 정도 오차")
- "과거 판매량, 날씨, 요일 정보 등을 종합해서 예측했습니다" 정도로 간단히

**[예측에 가장 큰 영향을 준 요인 3가지]**
1. 한글 설명 (영문 변수명): 왜 영향을 주는지 쉬운 해석 1문장
2. 한글 설명 (영문 변수명): 해석 1문장
3. 한글 설명 (영문 변수명): 해석 1문장
(영향도 숫자는 생략하거나, "가장 큰 영향", "두 번째로 큰 영향" 정도로만 표현)

**[발주 권장]**
- 권장 발주 수량: XX개
- 참고사항: 쉬운 말로 1-2문장 (예: "최근 일주일 평균보다 낮게 예측되었으니 재고 확인 권장")

=== 예측 정보 ===
{context}
"""

# 후속 질문용 시스템 프롬프트
SYSTEM_PROMPT = """당신은 현대백화점 청과 수요 예측 시스템의 AI 어시스턴트입니다.
발주 담당자의 질문에 친절하게 답변합니다.

**중요: 발주 담당자는 데이터 전문가가 아닙니다. 쉬운 말로 답변하세요.**

**형식 규칙:**
- # 또는 ## 헤더 금지
- 범위 표시: 물결표(~) 대신 하이픈(-) 사용
- 기술 용어(MAE, RMSE, SHAP, feature 등) 금지 → 쉬운 말로
- 변수명은 한글 설명 먼저 (예: "최근 6일 평균 판매량")

**답변 스타일:**
- 간결하고 명확하게 (2-4문장)
- 숫자와 이유를 함께 제시
- 전문용어 없이 쉬운 한국어로

=== 이 상품의 예측 정보 ===
{context}
"""


class PredictionChatbot:
    """예측값 설명 챗봇 (Claude / Gemini 전환 가능)"""

    def __init__(self):
        self.llm = get_llm_client()
        print(f"[Chatbot] Using {config.LLM_PROVIDER} as LLM provider")

    def _parse_json(self, value) -> dict:
        """JSON 문자열을 dict로 파싱"""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except:
                return {}
        return {}

    def _format_row_data(self, row_data: Dict, sku_name: str, horizon: int) -> str:
        """row_data를 AI에게 전달할 텍스트로 변환"""
        if not row_data:
            return "예측 정보 없음"

        lines = []

        # 기본 정보
        lines.append(f"단품코드: {row_data.get('sku_code', 'N/A')}")
        lines.append(f"단품명: {sku_name}")
        lines.append(f"예측 기준일: {row_data.get('date_t', 'N/A')}")
        lines.append(f"예측 horizon: t+{horizon}")
        lines.append("")

        # 예측 결과
        lines.append(f"예측값 p50 (중간값): {row_data.get('p50', 'N/A')}개")
        lines.append(f"하한값 p10: {row_data.get('p10', 'N/A')}개")
        lines.append(f"상한값 p90: {row_data.get('p90', 'N/A')}개")
        lines.append(f"예측 모델: {row_data.get('model_name', 'N/A')}")
        lines.append(f"학습 샘플 수: {row_data.get('n_train_samples', 'N/A')}")

        # 검증 성능 (pinball loss)
        val_loss_sku = row_data.get('val_loss_sku')
        val_loss_global = row_data.get('val_loss_global')
        if val_loss_sku is not None:
            lines.append(f"이 SKU 검증 MAE (p50): {val_loss_sku:.2f}개")
        if val_loss_global is not None:
            lines.append(f"전체 SKU 평균 MAE (p50): {val_loss_global:.2f}개")
        lines.append("")

        # ========================================
        # 예측에 사용된 정보
        # ========================================
        used_features = []
        if row_data.get('lag'):
            used_features.append("과거 판매량")
        if row_data.get('rolling'):
            used_features.append("최근 판매 트렌드")
        if row_data.get('weather'):
            used_features.append("날씨 정보")
        if row_data.get('holiday'):
            used_features.append("휴일/요일 정보")
        if row_data.get('price_sales'):
            used_features.append("가격/판매 정보")

        if used_features:
            lines.append(f"예측에 사용된 정보: {', '.join(used_features)}")
            lines.append("")

        # Top 3 Features
        top_3 = row_data.get('top_3_features')
        if top_3:
            if isinstance(top_3, str):
                try:
                    top_3 = json.loads(top_3)
                except:
                    top_3 = []
            if isinstance(top_3, list):
                lines.append("예측에 가장 큰 영향을 준 요인 Top 3:")
                for i, feat in enumerate(top_3[:3], 1):
                    feat_name = get_feature_description(feat)
                    lines.append(f"  {i}. {feat_name} ({feat})")
                lines.append("")

        # Lag 피처
        lag_data = self._parse_json(row_data.get('lag'))
        if lag_data:
            lines.append("과거 판매량:")
            lag_1_val = lag_data.get('lag_1', row_data.get('lag_1', 'N/A'))
            lines.append(f"  어제 판매량: {lag_1_val}개")
            if lag_data.get('lag_2'):
                lines.append(f"  그저께 판매량: {lag_data.get('lag_2', 'N/A')}개")
            lines.append("")

        # Rolling 피처
        rolling_data = self._parse_json(row_data.get('rolling'))
        if rolling_data:
            lines.append("최근 판매 트렌드:")
            mean_6 = rolling_data.get('rolling_mean_6', row_data.get('rolling_mean_6', 'N/A'))
            lines.append(f"  최근 6일 평균: {mean_6}개")
            if rolling_data.get('rolling_std_6'):
                std_6 = rolling_data.get('rolling_std_6', 'N/A')
                lines.append(f"  최근 6일 판매 변동폭: {std_6}")
            lines.append("")

        # 날씨
        weather = self._parse_json(row_data.get('weather'))
        if weather:
            lines.append(f"날씨 예보 (t+{horizon}):")
            temp = weather.get('TEMP_AVG_future', weather.get(f'TEMP_AVG_t+{horizon}', 'N/A'))
            hm = weather.get('HM_AVG_future', weather.get(f'HM_AVG_t+{horizon}', 'N/A'))
            wind = weather.get('WIND_AVG_future', weather.get(f'WIND_AVG_t+{horizon}', 'N/A'))
            rn = weather.get('RN_DAY_future', weather.get(f'RN_DAY_t+{horizon}', 'N/A'))
            lines.append(f"  평균 기온: {temp}℃")
            lines.append(f"  평균 습도: {hm}%")
            lines.append(f"  평균 풍속: {wind}m/s")
            lines.append(f"  강수량: {rn}mm")
            lines.append("")

        # 휴일
        holiday = self._parse_json(row_data.get('holiday'))
        if holiday:
            lines.append(f"휴일 정보 (t+{horizon}):")
            # _future 형태 또는 기존 형태 모두 지원
            is_weekend = holiday.get('is_weekend_future', holiday.get(f'is_weekend_t+{horizon}', 0))
            is_hd = holiday.get('is_hd_holiday_future', holiday.get(f'is_hd_holiday_t+{horizon}', 0))
            holiday_korea = holiday.get('holiday_korea_t_future', holiday.get(f'holiday_korea_t_t+{horizon}', 0))
            lines.append(f"  금토일 여부: {'예' if is_weekend else '아니오'}")
            lines.append(f"  현대백화점 휴무: {'예' if is_hd else '아니오'}")
            lines.append(f"  한국 공휴일: {'예' if holiday_korea else '아니오'}")
            lines.append("")

        # 가격/판매 피처
        price_sales = self._parse_json(row_data.get('price_sales'))
        if price_sales:
            lines.append("전일 판매 정보 (t-1):")
            if price_sales.get('AVG_SELL_UPRC_t-1'):
                lines.append(f"  평균 판매단가: {price_sales.get('AVG_SELL_UPRC_t-1'):,}원")
            if price_sales.get('TR_CNT_t-1'):
                lines.append(f"  거래건수: {price_sales.get('TR_CNT_t-1')}건")
            if price_sales.get('PRCH_QTY_t-1'):
                lines.append(f"  매입수량: {price_sales.get('PRCH_QTY_t-1')}개")
            if price_sales.get('RMRGN_RATE_t-1'):
                lines.append(f"  매익률: {price_sales.get('RMRGN_RATE_t-1') * 100:.1f}%")
            lines.append("")

        # SHAP
        shap = self._parse_json(row_data.get('shap_values'))
        if shap:
            lines.append("예측에 영향을 준 요인 (영향도순):")
            sorted_shap = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for i, (feat, val) in enumerate(sorted_shap, 1):
                feat_name = get_feature_description(feat)
                direction = "↑" if val > 0 else "↓"
                lines.append(f"  {i}. {feat_name} ({feat}): 영향도 {abs(val):.2f} {direction}")
            lines.append("")

        # 하이퍼파라미터는 발주 담당자에게 불필요하므로 생략

        return "\n".join(lines)

    def generate_initial_report(self, row_data: Dict, sku_name: str, horizon: int) -> str:
        """챗봇 열릴 때 초기 리포트 생성 (API 호출)"""
        context_str = self._format_row_data(row_data, sku_name, horizon)

        if not self.llm.is_available():
            return self._fallback_report(row_data, sku_name)

        try:
            prompt = INITIAL_REPORT_PROMPT.format(context=context_str)
            return self.llm.generate(prompt)

        except Exception as e:
            print(f"[Chatbot] Report generation error: {e}")
            return self._fallback_report(row_data, sku_name)

    def _fallback_report(self, row_data: Dict, sku_name: str) -> str:
        """API 실패 시 폴백 리포트"""
        p50 = row_data.get('p50', 'N/A')
        p10 = row_data.get('p10', 'N/A')
        p90 = row_data.get('p90', 'N/A')
        model = row_data.get('model_name', 'Unknown')
        rmse = row_data.get('val_rmse', 'N/A')

        return f"""**[{sku_name} 예측 리포트]**

**모델**: {model} (RMSE: {rmse})

**예측값**:
- 하한(p10): {p10}개
- 중간(p50): {p50}개
- 상한(p90): {p90}개

**권장**: p50 기준 {p50}개 발주를 권장합니다.

(AI 연결 실패로 간략 리포트가 표시됩니다. 질문을 입력하시면 다시 시도합니다.)
"""

    def get_response(self, user_message: str, context: Dict, chat_history: List[Dict], horizon: int = 1) -> str:
        """사용자 질문에 대한 응답 생성"""
        context_str = self._format_row_data(context, context.get('sku_name', ''), horizon)

        if not self.llm.is_available():
            return self._fallback_response(user_message, context)

        try:
            system_prompt = SYSTEM_PROMPT.format(context=context_str)
            user_prompt = f"사용자 질문: {user_message}"
            return self.llm.generate(user_prompt, system_prompt)

        except Exception as e:
            print(f"[Chatbot] Response error: {e}")
            return self._fallback_response(user_message, context)

    def _fallback_response(self, question: str, context: Dict) -> str:
        """API 실패 시 폴백 응답"""
        sku_name = context.get('sku_name', '상품')
        p50 = context.get('p50', 'N/A')

        if "날씨" in question or "기온" in question:
            weather = self._parse_json(context.get('weather'))
            if weather:
                temp = weather.get('TEMP_AVG_t+1', 'N/A')
                return f"예보 기온 {temp}도입니다. 날씨 요인은 모델에 반영되어 있습니다."

        if "트렌드" in question or "최근" in question:
            lag = self._parse_json(context.get('lag'))
            lag_1 = lag.get('lag_1', context.get('lag_1', 'N/A'))
            return f"1일 전 판매량은 {lag_1}개입니다."

        return f"[{sku_name}] 예측값 {p50}개입니다. 더 구체적인 질문을 해주세요."

    def get_quick_suggestions(self) -> List[str]:
        """빠른 질문 제안"""
        return ["예측 근거는?", "날씨 영향?", "트렌드는?"]


def get_chatbot() -> PredictionChatbot:
    """챗봇 인스턴스 반환"""
    if 'chatbot_instance' not in st.session_state:
        st.session_state.chatbot_instance = PredictionChatbot()
    return st.session_state.chatbot_instance