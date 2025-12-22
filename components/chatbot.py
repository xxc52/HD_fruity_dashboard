"""
AI Chatbot Module
=================

Google Gemini 2.5 Flash API를 사용한 예측값 Q&A 챗봇
"""

import streamlit as st
from typing import Dict, List
import json

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[Chatbot] google-genai not installed. Run: pip install google-genai")


# Feature 한글 매핑 (t+n 패턴은 동적으로 처리)
FEATURE_DESCRIPTIONS = {
    # Lag 피처 (과거 판매량)
    'lag_1': '1일 전 판매량',
    'lag_2': '2일 전 판매량',
    'lag_3': '3일 전 판매량',
    'lag_4': '4일 전 판매량',
    'lag_5': '5일 전 판매량',
    'lag_6': '6일 전 판매량',
    'lag_10': '10일 전 판매량',
    'lag_11': '11일 전 판매량',
    'lag_12': '12일 전 판매량',
    'lag_13': '13일 전 판매량',
    'lag_17': '17일 전 판매량',
    'lag_18': '18일 전 판매량',
    'lag_19': '19일 전 판매량',
    'lag_20': '20일 전 판매량',
    'lag_24': '24일 전 판매량',
    'lag_25': '25일 전 판매량',
    'lag_26': '26일 전 판매량',
    'lag_27': '27일 전 판매량',

    # Rolling 피처 (이동 통계)
    'rolling_mean_6': '최근 6일 평균 판매량',
    'rolling_std_6': '최근 6일 판매량 표준편차',
    'rolling_max_6': '최근 6일 최대 판매량',
    'rolling_mean_27': '최근 27일 평균 판매량',
    'rolling_std_27': '최근 27일 판매량 표준편차',
    'rolling_max_27': '최근 27일 최대 판매량',

    # 날씨 피처 (t+n은 동적 처리)
    'TEMP_AVG': '평균 기온(℃)',
    'HM_AVG': '평균 습도(%)',
    'WIND_AVG': '평균 풍속(m/s)',
    'RN_DAY': '일 강수량(mm)',

    # 휴일 피처
    'is_weekend': '금토일 여부',
    'is_hd_holiday': '현대백화점 휴무일',
    'holiday_korea_t-1': '전일 한국 공휴일',
    'holiday_korea_t': '당일 한국 공휴일',
    'holiday_korea_t+1': '익일 한국 공휴일',
    'holiday_christ_t-1': '전일 크리스마스',
    'holiday_christ_t': '당일 크리스마스',
    'holiday_newyear_t-1': '전일 신년',
    'holiday_newyear_t': '당일 신년',
    'holiday_etc': '기타 휴일',

    # 가격/판매 피처 (t-1: 전일)
    'AVG_SELL_UPRC_t-1': '전일 평균 판매단가',
    'PRCH_QTY_t-1': '전일 매입수량',
    'RMRGN_RATE_t-1': '전일 매익률',
    'RMRGN_t-1': '전일 매익금액',
    'TR_CNT_t-1': '전일 거래건수',
    'TSALE_AMT_t-1': '전일 총판매금액',
}


def get_feature_description(feature_name: str) -> str:
    """피처명을 한글 설명으로 변환 (t+n 패턴 동적 처리)"""
    # 정확히 일치하면 바로 반환
    if feature_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature_name]

    # t+n 패턴 처리 (예: TEMP_AVG_t+1 → 평균 기온(℃))
    import re
    match = re.match(r'(.+?)_t\+(\d+)$', feature_name)
    if match:
        base_name = match.group(1)
        horizon = match.group(2)
        if base_name in FEATURE_DESCRIPTIONS:
            return f"{FEATURE_DESCRIPTIONS[base_name]} (t+{horizon})"

    # 언더스코어 + t+n 패턴 (예: is_weekend_t+1)
    match = re.match(r'(.+)_t\+(\d+)$', feature_name)
    if match:
        base_name = match.group(1)
        horizon = match.group(2)
        if base_name in FEATURE_DESCRIPTIONS:
            return f"{FEATURE_DESCRIPTIONS[base_name]} (t+{horizon})"

    # holiday_korea_t+1_t+2 같은 복잡한 패턴
    if 'holiday_korea' in feature_name:
        return '한국 공휴일 관련'
    if 'holiday_christ' in feature_name:
        return '크리스마스 관련'
    if 'holiday_newyear' in feature_name:
        return '신년 관련'

    # 못 찾으면 원본 반환
    return feature_name


# 초기 리포트 생성 프롬프트 (고정)
INITIAL_REPORT_PROMPT = """당신은 현대백화점 청과 수요 예측 시스템의 AI 어시스턴트입니다.
아래 예측 정보를 바탕으로 발주 담당자를 위한 리포트를 작성해주세요.

다음 형식으로 작성하세요:

**[모델 정보]**
- 사용된 예측 모델과 검증 성능(RMSE)을 간단히 설명
- "모델이 사용한 Feature 카테고리"에 있는 변수 유형들을 1줄로 요약 (예: 날씨, 휴일, 판매이력 등 사용됨)

**[주요 영향 요인 Top 3 해석]**
- top_3_features에 있는 3가지 요인을 한글로 번역하고, 각각이 예측에 어떤 영향을 미치는지 간단히 해석
- (참고: SHAP 순위가 낮다고 해서 해당 변수가 사용되지 않은 것은 아님)

**[최근 트렌드]**
- lag_1(1일 전 판매량)과 rolling_mean_6(최근 6일 평균), rolling_mean_27(최근 27일 평균)을 활용해 최근 판매 트렌드 설명

**[예측 범위 설명]**
- p10(하한값), p50(중간값), p90(상한값)의 의미를 간단히 설명하고, 예측 불확실성 해석

**[발주 권장]**
- 결론으로 권장 발주 수량(p50 기준)과 발주 시 고려사항을 1-2문장으로 정리

=== 예측 정보 ===
{context}
"""

# 후속 질문용 시스템 프롬프트
SYSTEM_PROMPT = """당신은 현대백화점 청과 수요 예측 시스템의 AI 어시스턴트입니다.
발주 담당자의 질문에 친절하고 전문적으로 답변합니다.

역할:
- 예측값의 근거 설명
- 발주량 조정 시나리오 분석
- 날씨, 휴일 등 외부 요인 영향 설명
- 모델 신뢰도 및 예측 범위 해석

**중요 구분**:
- "모델이 사용한 Feature 카테고리"에 있으면 = 해당 변수가 모델의 입력으로 사용됨 (예: 날씨 변수 사용됨)
- "SHAP 영향도"는 해당 예측에서 변수의 상대적 중요도를 나타냄 (순위가 낮다고 사용 안 된 게 아님)
- 사용자가 "~를 고려했나?", "~가 변수로 사용되었나?" 질문 시 → Feature 카테고리 확인 후 답변

답변 스타일:
- 간결하고 명확하게 (3-5문장)
- 숫자와 근거를 함께 제시
- 발주 담당자 관점에서 실용적인 조언
- 한국어로 답변

=== 이 상품의 예측 정보 ===
{context}
"""


class PredictionChatbot:
    """예측값 설명 챗봇 (Gemini 2.5 Flash)"""

    def __init__(self):
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Gemini 클라이언트 초기화"""
        if not GEMINI_AVAILABLE:
            print("[Chatbot] google-genai not installed")
            return

        try:
            # secrets.toml에서 API 키 읽기
            api_key = None
            if "gemini" in st.secrets and "api_key" in st.secrets["gemini"]:
                api_key = st.secrets["gemini"]["api_key"]

            if api_key:
                self.client = genai.Client(api_key=api_key)
                print("[Chatbot] Gemini Client initialized OK")
            else:
                print("[Chatbot] API key not found in secrets.toml [gemini] section")
        except Exception as e:
            print(f"[Chatbot] Init failed: {e}")
            self.client = None

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
        lines.append(f"검증 RMSE: {row_data.get('val_rmse', 'N/A')}")
        lines.append(f"학습 샘플 수: {row_data.get('n_train_samples', 'N/A')}")
        lines.append("")

        # ========================================
        # 모델이 사용한 Feature 카테고리 (중요!)
        # ========================================
        used_features = []
        if row_data.get('lag'):
            used_features.append("과거 판매량 (lag_1~lag_27)")
        if row_data.get('rolling'):
            used_features.append("이동통계 (rolling_mean, rolling_std, rolling_max)")
        if row_data.get('weather'):
            used_features.append("날씨 (기온, 습도, 풍속, 강수량)")
        if row_data.get('holiday'):
            used_features.append("휴일 (주말, 공휴일, 현대백화점휴무)")
        if row_data.get('price_sales'):
            used_features.append("가격/판매 (판매단가, 거래건수, 매입수량, 매익률)")

        if used_features:
            lines.append("** 모델이 사용한 Feature 카테고리 (입력 변수) **")
            lines.append("(아래 변수들은 모두 모델 학습에 사용됨. SHAP 순위와 무관하게 모델 입력으로 활용됨)")
            for feat in used_features:
                lines.append(f"  ✓ {feat}")
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
                lines.append("주요 영향 요인 Top 3:")
                for i, feat in enumerate(top_3[:3], 1):
                    feat_name = get_feature_description(feat)
                    lines.append(f"  {i}. {feat} ({feat_name})")
                lines.append("")

        # Lag 피처
        lag_data = self._parse_json(row_data.get('lag'))
        if lag_data:
            lines.append("과거 판매량 (Lag):")
            lines.append(f"  lag_1 (1일 전): {lag_data.get('lag_1', row_data.get('lag_1', 'N/A'))}개")
            if lag_data.get('lag_2'):
                lines.append(f"  lag_2 (2일 전): {lag_data.get('lag_2', 'N/A')}개")
            lines.append("")

        # Rolling 피처
        rolling_data = self._parse_json(row_data.get('rolling'))
        if rolling_data:
            lines.append("최근 트렌드 (Rolling):")
            lines.append(f"  rolling_mean_6: {rolling_data.get('rolling_mean_6', row_data.get('rolling_mean_6', 'N/A'))}개")
            if rolling_data.get('rolling_std_6'):
                lines.append(f"  rolling_std_6: {rolling_data.get('rolling_std_6', 'N/A')}")
            lines.append("")

        # 날씨
        weather = self._parse_json(row_data.get('weather'))
        if weather:
            lines.append(f"날씨 예보 (t+{horizon}):")
            lines.append(f"  평균 기온: {weather.get(f'TEMP_AVG_t+{horizon}', 'N/A')}℃")
            lines.append(f"  평균 습도: {weather.get(f'HM_AVG_t+{horizon}', 'N/A')}%")
            lines.append(f"  평균 풍속: {weather.get(f'WIND_AVG_t+{horizon}', 'N/A')}m/s")
            lines.append(f"  강수량: {weather.get(f'RN_DAY_t+{horizon}', 'N/A')}mm")
            lines.append("")

        # 휴일
        holiday = self._parse_json(row_data.get('holiday'))
        if holiday:
            lines.append(f"휴일 정보 (t+{horizon}):")
            suffix = f"t+{horizon}"
            is_weekend = holiday.get(f'is_weekend_{suffix}', 0)
            is_hd = holiday.get(f'is_hd_holiday_{suffix}', 0)
            holiday_korea = holiday.get(f'holiday_korea_t_{suffix}', 0)
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
            lines.append("SHAP 영향도 Top 5:")
            sorted_shap = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for feat, val in sorted_shap:
                feat_name = get_feature_description(feat)
                lines.append(f"  {feat} ({feat_name}): {val:.4f}")
            lines.append("")

        # 하이퍼파라미터
        hp = self._parse_json(row_data.get('hyperparameters'))
        if hp:
            hp_str = ", ".join([f"{k}={v}" for k, v in list(hp.items())[:4]])
            lines.append(f"하이퍼파라미터: {hp_str}")

        return "\n".join(lines)

    def generate_initial_report(self, row_data: Dict, sku_name: str, horizon: int) -> str:
        """챗봇 열릴 때 초기 리포트 생성 (API 호출)"""
        context_str = self._format_row_data(row_data, sku_name, horizon)

        if not self.client:
            return self._fallback_report(row_data, sku_name)

        try:
            prompt = INITIAL_REPORT_PROMPT.format(context=context_str)

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text

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

        if not self.client:
            return self._fallback_response(user_message, context)

        try:
            prompt = SYSTEM_PROMPT.format(context=context_str)
            prompt += f"\n\n사용자 질문: {user_message}"

            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text

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
