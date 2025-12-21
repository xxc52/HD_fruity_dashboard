"""
Cold Start Chatbot Module
=========================

신규 SKU(판매 이력 없는 상품)의 발주량 예측을 위한 AI 챗봇
유사 상품의 첫 주 판매 데이터를 기반으로 예측값 산출
"""

import streamlit as st
from typing import Dict, List, Optional
import pandas as pd

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    types = None

# 품목 그룹 정의
PRODUCT_GROUPS: Dict[str, List[str]] = {
    '딸기': ['딸기', '킹스베리', '아리향', '비타베리'],

    '일반 감귤': ['귤', '감귤', '썬레드', '만다린'],

    '프리미엄 감귤': [
        '한라봉', '천혜향', '레드향', '황금향',
        '카라향', '설국향', '달코미', '한라향',
        '동백향', '소원향', '진지향', '탐라향',
        '가을향', '루비향'
    ],

    '오렌지': ['오렌지'],
    '레몬/라임/유자': ['레몬', '라임', '유자'],
    '자몽': ['자몽', '허니포멜로'],

    '포도': [
        '포도', '샤인', '캠벨', '머스캣', '사파이어',
        '캔디하트', '캔디 하트', '코튼 캔디', '코튼캔디', '레드클라렛',
        '골드스위트', '슈팅스타', '글로리스타', '킹데라웨어', '캔디스냅',
        '마이하트', '루비스위트', '바이올렛킹',
        '쥬얼머스켓', '머스켓써틴', '홍주씨들리스'
    ],

    '사과': ['사과', '피치애플'],
    '석류': ['석류'],
    '토마토': ['토마토', '방울토마토', '완숙토마토'],
    '바나나': ['바나나'],
    '수박': ['수박'],
    '멜론': ['멜론', '메론', '하미과'],
    '배': ['배', '조이스킨'],
    '복숭아': ['복숭아', '천도복숭아', '엑셀라', '홍설도', '황도'],
    '키위': ['키위', '참다래'],
    '망고': ['망고'],
    '파인애플': ['파인애플'],
    '블루베리': ['블루베리', '코튼베리', '베리스냅'],
    '체리': ['체리'],
    '자두/살구': ['자두', '살구', '플럼코트'],
    '감': ['감', '단감', '곶감', '흑시'],
    '참외': ['참외'],
    '무화과': ['무화과'],
    '아보카도': ['아보카도'],
    '용과': ['용과'],
    '오디': ['오디'],
    '파파야': ['파파야'],
    '패션후르츠': ['패션후르츠'],
    '두리안': ['두리안'],
    '리치': ['리치'],
    '모과': ['모과'],
    '매실': ['매실'],
    '앵두': ['앵두']
}

# 품목 그룹 목록 (LLM에게 전달)
PRODUCT_GROUP_LIST = list(PRODUCT_GROUPS.keys())


def get_gemini_client():
    """Gemini 클라이언트 반환"""
    if not GEMINI_AVAILABLE:
        return None
    try:
        api_key = st.secrets.get("gemini", {}).get("api_key")
        if api_key:
            return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"[ColdStart] Gemini init error: {e}")
    return None


def get_supabase_client():
    """Supabase 클라이언트 반환"""
    try:
        from supabase import create_client
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        print(f"[ColdStart] Supabase init error: {e}")
        return None


# ========== Step 1: 품목 그룹 추론 ==========

def infer_product_group_with_search(client, fruit_name: str) -> tuple[Optional[str], Optional[str]]:
    """웹 검색을 활용하여 과일 정보를 찾고 품목 그룹 판단

    Returns:
        tuple: (품목 그룹명, 검색으로 알아낸 과일 정보)
    """
    prompt = f""""{fruit_name}"이(가) 무슨 과일인지 검색해서 알려주세요.

아래 품목 그룹 중 어디에 속하는지 판단해주세요:
{', '.join(PRODUCT_GROUP_LIST)}

다음 형식으로 정확히 답변하세요:
과일정보: [검색 결과 요약 1-2문장]
품목그룹: [위 목록 중 하나 또는 None]"""

    try:
        # Google Search grounding 활성화
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config
        )
        result = response.text.strip()

        # 결과 파싱
        fruit_info = None
        group_name = None

        for line in result.split('\n'):
            if '과일정보:' in line:
                fruit_info = line.split('과일정보:')[-1].strip()
            elif '품목그룹:' in line:
                group_text = line.split('품목그룹:')[-1].strip()
                if group_text.lower() != 'none':
                    # 유효한 그룹명인지 확인
                    for group in PRODUCT_GROUP_LIST:
                        if group in group_text or group_text in group:
                            group_name = group
                            break

        return group_name, fruit_info

    except Exception as e:
        print(f"[ColdStart] Search inference error: {e}")
        return None, None


def infer_product_group(client, fruit_name: str) -> Optional[str]:
    """LLM으로 입력된 과일명이 어느 품목 그룹에 속하는지 판단 (기본 버전)"""
    prompt = f"""당신은 과일 분류 전문가입니다.
사용자가 입력한 과일명이 아래 품목 그룹 중 어디에 속하는지 판단해주세요.

품목 그룹 목록:
{', '.join(PRODUCT_GROUP_LIST)}

입력된 과일명: "{fruit_name}"

규칙:
1. 가장 적합한 품목 그룹명 하나만 출력
2. 어느 그룹에도 속하지 않으면 "None" 출력
3. 그룹명만 출력 (설명 없이)

예시:
- "타이백귤" → 일반 감귤
- "레드향" → 프리미엄 감귤
- "샤인머스캣" → 포도
- "블랙사파이어" → 포도
- "두리안" → 두리안
- "용눈알" → None

출력:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        result = response.text.strip()
        # 유효한 그룹명인지 확인
        if result in PRODUCT_GROUP_LIST:
            return result
        if result.lower() == "none":
            return None
        # 부분 매칭 시도
        for group in PRODUCT_GROUP_LIST:
            if group in result or result in group:
                return group
        return None
    except Exception as e:
        print(f"[ColdStart] Group inference error: {e}")
        return None


# ========== Step 2: 키워드 추출 ==========

def get_keywords_for_group(group_name: str) -> List[str]:
    """품목 그룹명으로 키워드 리스트 가져오기"""
    return PRODUCT_GROUPS.get(group_name, [])


# ========== Step 3: Supabase에서 유사 상품 검색 ==========

def search_similar_products(keywords: List[str]) -> List[Dict]:
    """after_preprocessing 테이블에서 키워드로 유사 상품 검색

    Note: 각 SKU당 30일치 데이터가 있어 중복이 많음 → limit 늘려서 검색 후 중복 제거
    """
    client = get_supabase_client()
    if not client:
        return []

    found_products = []
    seen_skus = set()

    try:
        for keyword in keywords:
            # sku_name에서 LIKE 검색 (limit 500으로 늘려서 더 많은 unique SKU 확보)
            response = client.table("after_preprocessing") \
                .select("sku, sku_name") \
                .ilike("sku_name", f"%{keyword}%") \
                .limit(500) \
                .execute()

            if response.data:
                for row in response.data:
                    sku = row.get('sku')
                    if sku and sku not in seen_skus:
                        seen_skus.add(sku)
                        found_products.append({
                            'sku': sku,
                            'sku_name': row.get('sku_name', str(sku))
                        })
    except Exception as e:
        print(f"[ColdStart] Search error: {e}")

    return found_products


# ========== Step 4: Top 3 유사 상품 선정 (LLM) ==========

def select_top3_similar(client, input_name: str, products: List[Dict]) -> List[Dict]:
    """LLM으로 입력 상품과 가장 유사한 Top 3 선정"""
    if not products:
        return []
    if len(products) <= 3:
        return products

    product_list = "\n".join([
        f"- {p['sku_name']} (SKU: {p['sku']})"
        for p in products[:30]  # 최대 30개만 전달
    ])

    prompt = f"""당신은 현대백화점 청과 바이어입니다.
신규 상품 "{input_name}"과 가장 유사한 기존 상품 3개를 선정해주세요.

기존 상품 목록:
{product_list}

선정 기준:
1. 과일 종류가 같거나 유사
2. 규격/용량이 비슷
3. 가격대가 비슷할 것으로 예상되는 상품

아래 형식으로 정확히 출력해주세요:
1. [SKU코드] - [상품명] - [선정 근거 한 줄]
2. [SKU코드] - [상품명] - [선정 근거 한 줄]
3. [SKU코드] - [상품명] - [선정 근거 한 줄]

출력:"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        result = response.text.strip()

        # 파싱: SKU 코드 추출
        selected = []
        lines = result.split('\n')
        for line in lines:
            if line.strip() and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                # SKU 추출 시도
                for p in products:
                    sku_str = str(p['sku'])
                    if sku_str in line:
                        selected.append({
                            'sku': p['sku'],
                            'sku_name': p['sku_name'],
                            'reason': line.split('-')[-1].strip() if '-' in line else ''
                        })
                        break

        # 부족하면 상위 상품으로 채움
        while len(selected) < 3 and len(products) > len(selected):
            candidate = products[len(selected)]
            if candidate['sku'] not in [s['sku'] for s in selected]:
                selected.append({
                    'sku': candidate['sku'],
                    'sku_name': candidate['sku_name'],
                    'reason': '키워드 매칭'
                })

        return selected[:3]

    except Exception as e:
        print(f"[ColdStart] Top3 selection error: {e}")
        return products[:3]


# ========== Step 5: 첫 주 통계 계산 ==========

def get_first_week_stats(sku_list: List[str]) -> Dict[str, Dict]:
    """각 SKU의 첫 7일 판매 통계 계산"""
    client = get_supabase_client()
    if not client:
        return {}

    stats = {}

    for sku in sku_list:
        try:
            # 해당 SKU의 모든 데이터 조회 (sale_dt 오름차순)
            response = client.table("after_preprocessing") \
                .select("sale_dt, sell_qty") \
                .eq("sku", sku) \
                .order("sale_dt", desc=False) \
                .limit(30) \
                .execute()

            if response.data and len(response.data) > 0:
                df = pd.DataFrame(response.data)
                # 첫 7일만 추출
                first_week = df.head(7)

                if len(first_week) > 0:
                    qty = first_week['sell_qty'].astype(float)
                    stats[sku] = {
                        'mean': round(qty.mean(), 1),
                        'min': int(qty.min()),
                        'max': int(qty.max()),
                        'std': round(qty.std(), 1) if len(qty) > 1 else 0.0,
                        'days': len(first_week)
                    }
        except Exception as e:
            print(f"[ColdStart] Stats error for SKU {sku}: {e}")
            continue

    return stats


# ========== Step 6: 최종 예측값 산출 (LLM) ==========

def generate_final_prediction(
    client,
    input_name: str,
    group_name: str,
    top3: List[Dict],
    stats: Dict[str, Dict]
) -> str:
    """통계 데이터 기반 최종 발주량 예측"""

    # 통계 테이블 문자열 생성
    stats_lines = []
    for item in top3:
        sku = item['sku']
        if sku in stats:
            s = stats[sku]
            stats_lines.append(
                f"| {item['sku_name'][:15]} | {s['mean']} | {s['min']} | {s['max']} | {s['std']} |"
            )

    stats_table = "\n".join(stats_lines) if stats_lines else "통계 데이터 없음"

    prompt = f"""당신은 현대백화점 청과 발주 전문가를 돕고 있습니다.
신규 상품 "{input_name}"의 첫 주 예상 발주량을 산출해주세요.

품목 그룹: {group_name}

유사 상품 첫 주 판매 통계:
| 상품명 | 평균 | 최소 | 최대 | 표준편차 |
|--------|-----|-----|-----|---------|
{stats_table}

다음 형식으로 답변해주세요:

▶ 추천 발주량: [하한]~[상한]개
  (근거: [2-3문장으로 근거 설명])

추가 고려사항:
- [주의사항이나 조정 필요 상황 1-2개]"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"[ColdStart] Prediction error: {e}")
        # 폴백: 통계 기반 단순 계산
        if stats:
            means = [s['mean'] for s in stats.values()]
            avg_mean = sum(means) / len(means)
            return f"""▶ 추천 발주량: {int(avg_mean * 0.8)}-{int(avg_mean * 1.2)}개
  (근거: 유사 상품 첫 주 평균 {avg_mean:.0f}개 기준, 신상품 초기 보수적 접근 권장)"""
        return "통계 데이터가 부족하여 예측이 어렵습니다. 10-20개 소량 발주 후 추이 관찰을 권장합니다."


# ========== 메인 챗봇 함수 ==========

def run_cold_start_prediction(fruit_name: str) -> None:
    """Cold Start 예측 실행 (스트리밍 형태로 단계별 출력)"""

    gemini = get_gemini_client()
    if not gemini:
        st.error("AI 연결 실패. Gemini API 키를 확인해주세요.")
        return

    # Step 1: 품목 그룹 추론
    with st.status(f"'{fruit_name}' 분석 중...", expanded=True) as status:
        st.write("📋 품목 그룹을 찾고 있습니다...")

        # 먼저 기본 추론 시도
        group_name = infer_product_group(gemini, fruit_name)
        fruit_info = None

        if group_name:
            st.write(f"✅ **'{group_name}'** 그룹으로 분류되었습니다.")
        else:
            # 기본 추론 실패 시 웹 검색 활용
            st.write("🔍 웹 검색으로 과일 정보를 찾고 있습니다...")
            group_name, fruit_info = infer_product_group_with_search(gemini, fruit_name)

            if group_name:
                st.write(f"✅ **'{group_name}'** 그룹으로 분류되었습니다.")
                if fruit_info:
                    st.info(f"📖 {fruit_info}")
            else:
                st.write("⚠️ 매칭되는 품목 그룹이 없습니다. 기본 추천으로 진행합니다.")
                if fruit_info:
                    st.info(f"📖 {fruit_info}")
                st.warning("추천 발주량: 10-20개 (신규 품목, 소량 테스트 권장)")
                status.update(label="분석 완료", state="complete")
                return

        # Step 2: 키워드 추출
        keywords = get_keywords_for_group(group_name)
        st.write(f"🔍 검색 키워드: {', '.join(keywords[:5])}")

        # Step 3: 유사 상품 검색
        st.write("🔎 유사 상품을 검색 중입니다...")
        similar_products = search_similar_products(keywords)

        if not similar_products:
            st.write("⚠️ 유사 상품을 찾을 수 없습니다.")
            st.info("추천 발주량: 10-20개 (데이터 부족, 소량 테스트 권장)")
            status.update(label="분석 완료", state="complete")
            return

        st.write(f"✅ **{len(similar_products)}개** 상품을 찾았습니다.")

        # Step 4: Top 3 선정
        st.write("🎯 가장 유사한 상품 3개를 선정 중입니다...")
        top3 = select_top3_similar(gemini, fruit_name, similar_products)

        if top3:
            st.write("**유사 상품 Top 3:**")
            for i, item in enumerate(top3, 1):
                reason = item.get('reason', '')
                st.write(f"   {i}. {item['sku_name']} - {reason}")

        # Step 5: 첫 주 통계 계산
        st.write("📊 첫 주 판매 데이터를 분석 중입니다...")
        sku_list = [str(item['sku']) for item in top3]
        stats = get_first_week_stats(sku_list)

        if stats:
            # 통계 테이블 표시
            stats_data = []
            for item in top3:
                sku = str(item['sku'])
                if sku in stats:
                    s = stats[sku]
                    stats_data.append({
                        '상품명': item['sku_name'][:20],
                        '평균': s['mean'],
                        '최소': s['min'],
                        '최대': s['max'],
                        '표준편차': s['std']
                    })

            if stats_data:
                st.write("**첫 주 판매 통계:**")
                st.dataframe(pd.DataFrame(stats_data), hide_index=True)

        # Step 6: 최종 예측
        st.write("🧠 예측값을 산출 중입니다...")
        prediction = generate_final_prediction(gemini, fruit_name, group_name, top3, stats)

        status.update(label="분석 완료!", state="complete")

    # 최종 결과 표시
    st.markdown("---")
    st.markdown(f"### 📦 [{fruit_name}] 예측 결과")
    st.markdown(f"**품목 그룹:** {group_name}")
    # ~ 문자가 마크다운 취소선으로 해석되지 않도록 처리
    prediction_safe = prediction.replace("~", "\\~")
    st.markdown(prediction_safe)


# ========== Modal Dialog ==========

@st.dialog("🆕 신규 SKU 예측", width="large")
def cold_start_dialog():
    """신규 SKU 예측 모달 다이얼로그"""
    st.markdown("판매 이력이 없는 신규 상품의 예상 발주량을 예측합니다.")
    st.markdown("유사 상품의 첫 주 판매 데이터를 기반으로 분석합니다.")

    st.markdown("---")

    # 입력 영역
    fruit_name = st.text_input(
        "신규 상품명을 입력하세요",
        placeholder="예: 타이백귤, 조이베리, 천중도...",
        key="cold_start_input"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        predict_btn = st.button("🔮 예측하기", type="primary", use_container_width=True)
    with col2:
        if st.button("닫기", use_container_width=True):
            st.rerun()

    # 예측 실행
    if predict_btn and fruit_name:
        run_cold_start_prediction(fruit_name.strip())
    elif predict_btn and not fruit_name:
        st.warning("상품명을 입력해주세요.")


def show_cold_start_button():
    """신규 SKU 예측 버튼 표시 (app.py에서 호출)"""
    if st.button("🆕 신규 SKU 예측", use_container_width=True):
        cold_start_dialog()
