"""
Cold Start Chatbot Module
=========================

ì‹ ê·œ SKU(íŒë§¤ ì´ë ¥ ì—†ëŠ” ìƒí’ˆ)ì˜ ë°œì£¼ëŸ‰ ì˜ˆì¸¡ì„ ìœ„í•œ AI ì±—ë´‡
ìœ ì‚¬ ìƒí’ˆì˜ ì²« ì£¼ íŒë§¤ ë°ì´í„°ë¥¼ ê¸°ë°˜ìœ¼ë¡œ ì˜ˆì¸¡ê°’ ì‚°ì¶œ
"""

import streamlit as st
from typing import Dict, List, Optional
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Gemini
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None

# Claude
try:
    import anthropic
    import httpx
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    anthropic = None
    httpx = None

# í’ˆëª© ê·¸ë£¹ ì •ì˜
PRODUCT_GROUPS: Dict[str, List[str]] = {
    'ë”¸ê¸°': ['ë”¸ê¸°', 'í‚¹ìŠ¤ë² ë¦¬', 'ì•„ë¦¬í–¥', 'ë¹„íƒ€ë² ë¦¬'],
    'ì¼ë°˜ ê°ê·¤': ['ê·¤', 'ê°ê·¤', 'ì¬ë ˆë“œ', 'ë§Œë‹¤ë¦°'],
    'í”„ë¦¬ë¯¸ì—„ ê°ê·¤': [
        'í•œë¼ë´‰', 'ì²œí˜œí–¥', 'ë ˆë“œí–¥', 'í™©ê¸ˆí–¥',
        'ì¹´ë¼í–¥', 'ì„¤êµ­í–¥', 'ë‹¬ì½”ë¯¸', 'í•œë¼í–¥',
        'ë™ë°±í–¥', 'ì†Œì›í–¥', 'ì§„ì§€í–¥', 'íƒë¼í–¥',
        'ê°€ì„í–¥', 'ë£¨ë¹„í–¥'
    ],
    'ì˜¤ë Œì§€': ['ì˜¤ë Œì§€'],
    'ë ˆëª¬/ë¼ìž„/ìœ ìž': ['ë ˆëª¬', 'ë¼ìž„', 'ìœ ìž'],
    'ìžëª½': ['ìžëª½', 'í—ˆë‹ˆí¬ë©œë¡œ'],
    'í¬ë„': [
        'í¬ë„', 'ìƒ¤ì¸', 'ìº ë²¨', 'ë¨¸ìŠ¤ìº£', 'ì‚¬íŒŒì´ì–´',
        'ìº”ë””í•˜íŠ¸', 'ìº”ë”” í•˜íŠ¸', 'ì½”íŠ¼ ìº”ë””', 'ì½”íŠ¼ìº”ë””', 'ë ˆë“œí´ë¼ë ›',
        'ê³¨ë“œìŠ¤ìœ„íŠ¸', 'ìŠˆíŒ…ìŠ¤íƒ€', 'ê¸€ë¡œë¦¬ìŠ¤íƒ€', 'í‚¹ë°ë¼ì›¨ì–´', 'ìº”ë””ìŠ¤ëƒ…',
        'ë§ˆì´í•˜íŠ¸', 'ë£¨ë¹„ìŠ¤ìœ„íŠ¸', 'ë°”ì´ì˜¬ë ›í‚¹',
        'ì¥¬ì–¼ë¨¸ìŠ¤ì¼“', 'ë¨¸ìŠ¤ì¼“ì¨í‹´', 'í™ì£¼ì”¨ë“¤ë¦¬ìŠ¤'
    ],
    'ì‚¬ê³¼': ['ì‚¬ê³¼', 'í”¼ì¹˜ì• í”Œ'],
    'ì„ë¥˜': ['ì„ë¥˜'],
    'í† ë§ˆí† ': ['í† ë§ˆí† ', 'ë°©ìš¸í† ë§ˆí† ', 'ì™„ìˆ™í† ë§ˆí† '],
    'ë°”ë‚˜ë‚˜': ['ë°”ë‚˜ë‚˜'],
    'ìˆ˜ë°•': ['ìˆ˜ë°•'],
    'ë©œë¡ ': ['ë©œë¡ ', 'ë©”ë¡ ', 'í•˜ë¯¸ê³¼'],
    'ë°°': ['ë°°', 'ì¡°ì´ìŠ¤í‚¨'],
    'ë³µìˆ­ì•„': ['ë³µìˆ­ì•„', 'ì²œë„ë³µìˆ­ì•„', 'ì—‘ì…€ë¼', 'í™ì„¤ë„', 'í™©ë„'],
    'í‚¤ìœ„': ['í‚¤ìœ„', 'ì°¸ë‹¤ëž˜'],
    'ë§ê³ ': ['ë§ê³ '],
    'íŒŒì¸ì• í”Œ': ['íŒŒì¸ì• í”Œ'],
    'ë¸”ë£¨ë² ë¦¬': ['ë¸”ë£¨ë² ë¦¬', 'ì½”íŠ¼ë² ë¦¬', 'ë² ë¦¬ìŠ¤ëƒ…'],
    'ì²´ë¦¬': ['ì²´ë¦¬'],
    'ìžë‘/ì‚´êµ¬': ['ìžë‘', 'ì‚´êµ¬', 'í”ŒëŸ¼ì½”íŠ¸'],
    'ê°': ['ê°', 'ë‹¨ê°', 'ê³¶ê°', 'í‘ì‹œ'],
    'ì°¸ì™¸': ['ì°¸ì™¸'],
    'ë¬´í™”ê³¼': ['ë¬´í™”ê³¼'],
    'ì•„ë³´ì¹´ë„': ['ì•„ë³´ì¹´ë„'],
    'ìš©ê³¼': ['ìš©ê³¼'],
    'ì˜¤ë””': ['ì˜¤ë””'],
    'íŒŒíŒŒì•¼': ['íŒŒíŒŒì•¼'],
    'íŒ¨ì…˜í›„ë¥´ì¸ ': ['íŒ¨ì…˜í›„ë¥´ì¸ '],
    'ë‘ë¦¬ì•ˆ': ['ë‘ë¦¬ì•ˆ'],
    'ë¦¬ì¹˜': ['ë¦¬ì¹˜'],
    'ëª¨ê³¼': ['ëª¨ê³¼'],
    'ë§¤ì‹¤': ['ë§¤ì‹¤'],
    'ì•µë‘': ['ì•µë‘']
}

PRODUCT_GROUP_LIST = list(PRODUCT_GROUPS.keys())


# ========== LLM í´ë¼ì´ì–¸íŠ¸ ==========

def get_llm_client():
    """config.LLM_PROVIDERì— ë”°ë¼ LLM í´ë¼ì´ì–¸íŠ¸ ë°˜í™˜"""
    provider = getattr(config, 'LLM_PROVIDER', 'claude')

    if provider == "claude":
        if not CLAUDE_AVAILABLE:
            return None, "claude"
        try:
            api_key = st.secrets.get("claude", {}).get("claude_api_key")
            if api_key:
                # SSL ê²€ì¦ ë¹„í™œì„±í™” (ë‚´ë¶€ë§ í™˜ê²½)
                http_client = httpx.Client(verify=False)
                return anthropic.Anthropic(api_key=api_key, http_client=http_client), "claude"
        except Exception as e:
            print(f"[ColdStart] Claude init error: {e}")
        return None, "claude"

    else:  # gemini
        if not GEMINI_AVAILABLE:
            return None, "gemini"
        try:
            api_key = st.secrets.get("gemini", {}).get("gemini_api_key")
            if api_key:
                return genai.Client(api_key=api_key), "gemini"
        except Exception as e:
            print(f"[ColdStart] Gemini init error: {e}")
        return None, "gemini"


def call_llm(client, provider: str, prompt: str, use_search: bool = False) -> Optional[str]:
    """LLM í˜¸ì¶œ (providerì— ë”°ë¼ ë¶„ê¸°)"""
    try:
        if provider == "claude":
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()

        else:  # gemini
            if use_search and types:
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                config_obj = types.GenerateContentConfig(tools=[grounding_tool])
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=config_obj
                )
            else:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
            return response.text.strip()

    except Exception as e:
        print(f"[ColdStart] LLM call error: {e}")
        return None


def get_raw_sales_df():
    """ë¡œì»¬ raw_sales.csv ë¡œë“œ (ìºì‹±)"""
    csv_path = Path(__file__).parent.parent.parent / "data" / "raw_sales.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        return df
    except Exception as e:
        print(f"[ColdStart] raw_sales load error: {e}")
        return None


# ========== Step 1: í’ˆëª© ê·¸ë£¹ ì¶”ë¡  ==========

def keyword_based_group_match(fruit_name: str) -> Optional[str]:
    """í‚¤ì›Œë“œ ê¸°ë°˜ í’ˆëª© ê·¸ë£¹ ë§¤ì¹­ (LLM í´ë°±ìš©)"""
    fruit_lower = fruit_name.lower()
    for group_name, keywords in PRODUCT_GROUPS.items():
        for kw in keywords:
            if kw.lower() in fruit_lower or fruit_lower in kw.lower():
                return group_name
    return None


def infer_product_group(client, provider: str, fruit_name: str) -> Optional[str]:
    """LLMìœ¼ë¡œ ìž…ë ¥ëœ ê³¼ì¼ëª…ì´ ì–´ëŠ í’ˆëª© ê·¸ë£¹ì— ì†í•˜ëŠ”ì§€ íŒë‹¨"""

    # LLM ì¶”ë¡  ë¨¼ì € ì‹œë„ (ì‹ ê·œ í’ˆì¢…ë„ ë¶„ë¥˜ ê°€ëŠ¥)
    prompt = f"""ë‹¹ì‹ ì€ ê³¼ì¼ ë¶„ë¥˜ ì „ë¬¸ê°€ìž…ë‹ˆë‹¤.
ì‚¬ìš©ìžê°€ ìž…ë ¥í•œ ê³¼ì¼ëª…ì´ ì•„ëž˜ í’ˆëª© ê·¸ë£¹ ì¤‘ ì–´ë””ì— ì†í•˜ëŠ”ì§€ íŒë‹¨í•´ì£¼ì„¸ìš”.

í’ˆëª© ê·¸ë£¹ ëª©ë¡:
{', '.join(PRODUCT_GROUP_LIST)}

ìž…ë ¥ëœ ê³¼ì¼ëª…: "{fruit_name}"

ê·œì¹™:
1. ê°€ìž¥ ì í•©í•œ í’ˆëª© ê·¸ë£¹ëª… í•˜ë‚˜ë§Œ ì¶œë ¥
2. ì–´ëŠ ê·¸ë£¹ì—ë„ ì†í•˜ì§€ ì•Šìœ¼ë©´ "None" ì¶œë ¥
3. ê·¸ë£¹ëª…ë§Œ ì¶œë ¥ (ì„¤ëª… ì—†ì´)

ì˜ˆì‹œ:
- "íƒ€ì´ë°±ê·¤" â†’ ì¼ë°˜ ê°ê·¤
- "ë ˆë“œí–¥" â†’ í”„ë¦¬ë¯¸ì—„ ê°ê·¤
- "ìƒ¤ì¸ë¨¸ìŠ¤ìº£" â†’ í¬ë„
- "ë¸”ëž™ì‚¬íŒŒì´ì–´" â†’ í¬ë„

ì¶œë ¥:"""

    result = call_llm(client, provider, prompt)
    if not result:
        print(f"[ColdStart] LLM returned empty result for '{fruit_name}', trying keyword fallback")
        # LLM ì‹¤íŒ¨ ì‹œ í‚¤ì›Œë“œ ê¸°ë°˜ í´ë°±
        return keyword_based_group_match(fruit_name)

    print(f"[ColdStart] LLM raw response: '{result}'")

    # ì •í™•ížˆ ì¼ì¹˜
    if result in PRODUCT_GROUP_LIST:
        return result
    if result.lower() == "none":
        return None

    # ë¶€ë¶„ ë§¤ì¹­ ì‹œë„ (ì‘ë‹µì—ì„œ ê·¸ë£¹ëª… ì°¾ê¸°)
    for group in PRODUCT_GROUP_LIST:
        if group in result or result in group:
            return group

    # ë” ìœ ì—°í•œ ë§¤ì¹­: ì‘ë‹µ ì •ë¦¬ í›„ ìž¬ì‹œë„
    clean_result = result.strip().replace('"', '').replace("'", "")
    if clean_result in PRODUCT_GROUP_LIST:
        return clean_result

    return None


def infer_product_group_with_search(client, provider: str, fruit_name: str) -> tuple[Optional[str], Optional[str]]:
    """ì›¹ ê²€ìƒ‰ì„ í™œìš©í•˜ì—¬ ê³¼ì¼ ì •ë³´ë¥¼ ì°¾ê³  í’ˆëª© ê·¸ë£¹ íŒë‹¨"""

    prompt = f""""{fruit_name}"ì´(ê°€) ë¬´ìŠ¨ ê³¼ì¼ì¸ì§€ ê²€ìƒ‰í•´ì„œ ì•Œë ¤ì£¼ì„¸ìš”.

ì•„ëž˜ í’ˆëª© ê·¸ë£¹ ì¤‘ ì–´ë””ì— ì†í•˜ëŠ”ì§€ íŒë‹¨í•´ì£¼ì„¸ìš”:
{', '.join(PRODUCT_GROUP_LIST)}

ë‹¤ìŒ í˜•ì‹ìœ¼ë¡œ ì •í™•ížˆ ë‹µë³€í•˜ì„¸ìš”:
ê³¼ì¼ì •ë³´: [ê²€ìƒ‰ ê²°ê³¼ ìš”ì•½ 1-2ë¬¸ìž¥]
í’ˆëª©ê·¸ë£¹: [ìœ„ ëª©ë¡ ì¤‘ í•˜ë‚˜ ë˜ëŠ” None]"""

    try:
        if provider == "claude":
            # Claude ì›¹ ê²€ìƒ‰ (web_search tool ì‚¬ìš©)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                tools=[{"type": "web_search_20250305"}],
                messages=[{"role": "user", "content": prompt}]
            )
            # ì‘ë‹µì—ì„œ í…ìŠ¤íŠ¸ ì¶”ì¶œ
            result = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    result += block.text
            result = result.strip()

        elif provider == "gemini":
            # Gemini ì›¹ ê²€ìƒ‰ (Google Search grounding)
            if types:
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                config_obj = types.GenerateContentConfig(tools=[grounding_tool])
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=config_obj
                )
                result = response.text.strip()
            else:
                return None, None
        else:
            return None, None

    except Exception as e:
        print(f"[ColdStart] Web search error: {e}")
        return None, None

    if not result:
        return None, None

    fruit_info = None
    group_name = None

    for line in result.split('\n'):
        if 'ê³¼ì¼ì •ë³´:' in line:
            fruit_info = line.split('ê³¼ì¼ì •ë³´:')[-1].strip()
        elif 'í’ˆëª©ê·¸ë£¹:' in line:
            group_text = line.split('í’ˆëª©ê·¸ë£¹:')[-1].strip()
            if group_text.lower() != 'none':
                for group in PRODUCT_GROUP_LIST:
                    if group in group_text or group_text in group:
                        group_name = group
                        break

    return group_name, fruit_info


# ========== Step 2: í‚¤ì›Œë“œ ì¶”ì¶œ ==========

def get_keywords_for_group(group_name: str) -> List[str]:
    """í’ˆëª© ê·¸ë£¹ëª…ìœ¼ë¡œ í‚¤ì›Œë“œ ë¦¬ìŠ¤íŠ¸ ê°€ì ¸ì˜¤ê¸°"""
    return PRODUCT_GROUPS.get(group_name, [])


# ========== Step 3: ë¡œì»¬ CSVì—ì„œ ìœ ì‚¬ ìƒí’ˆ ê²€ìƒ‰ ==========

def search_similar_products(keywords: List[str], store: str = None) -> List[Dict]:
    """raw_sales.csvì—ì„œ í‚¤ì›Œë“œë¡œ ìœ ì‚¬ ìƒí’ˆ ê²€ìƒ‰"""
    df = get_raw_sales_df()
    if df is None or df.empty:
        return []

    # ì í¬ í•„í„° (ì„ íƒì‚¬í•­)
    if store:
        df = df[df['STORE_CD'].astype(str) == str(store)]

    found_products = []
    seen_skus = set()

    try:
        for keyword in keywords:
            # ìƒí’ˆëª…ì—ì„œ í‚¤ì›Œë“œ ê²€ìƒ‰ (ëŒ€ì†Œë¬¸ìž ë¬´ì‹œ)
            mask = df['PRDT_NM'].str.contains(keyword, case=False, na=False)
            matches = df[mask][['PRDT_CD', 'PRDT_NM']].drop_duplicates()

            for _, row in matches.iterrows():
                sku = str(row['PRDT_CD'])
                if sku not in seen_skus:
                    seen_skus.add(sku)
                    found_products.append({
                        'sku': sku,
                        'sku_name': row['PRDT_NM']
                    })

                # ìµœëŒ€ 100ê°œê¹Œì§€
                if len(found_products) >= 100:
                    break

            if len(found_products) >= 100:
                break

    except Exception as e:
        print(f"[ColdStart] Search error: {e}")

    return found_products


# ========== Step 4: Top 3 ìœ ì‚¬ ìƒí’ˆ ì„ ì • (LLM) ==========

def select_top3_similar(client, provider: str, input_name: str, products: List[Dict]) -> List[Dict]:
    """LLMìœ¼ë¡œ ìž…ë ¥ ìƒí’ˆê³¼ ê°€ìž¥ ìœ ì‚¬í•œ Top 3 ì„ ì •"""
    if not products:
        return []
    if len(products) <= 3:
        return products

    product_list = "\n".join([
        f"- {p['sku_name']} (SKU: {p['sku']})"
        for p in products[:30]
    ])

    prompt = f"""ë‹¹ì‹ ì€ í˜„ëŒ€ë°±í™”ì  ì²­ê³¼ ë°”ì´ì–´ìž…ë‹ˆë‹¤.
ì‹ ê·œ ìƒí’ˆ "{input_name}"ê³¼ ê°€ìž¥ ìœ ì‚¬í•œ ê¸°ì¡´ ìƒí’ˆ 3ê°œë¥¼ ì„ ì •í•´ì£¼ì„¸ìš”.

ê¸°ì¡´ ìƒí’ˆ ëª©ë¡:
{product_list}

ì„ ì • ê¸°ì¤€:
1. ê³¼ì¼ ì¢…ë¥˜ê°€ ê°™ê±°ë‚˜ ìœ ì‚¬
2. ê·œê²©/ìš©ëŸ‰ì´ ë¹„ìŠ·
3. ê°€ê²©ëŒ€ê°€ ë¹„ìŠ·í•  ê²ƒìœ¼ë¡œ ì˜ˆìƒë˜ëŠ” ìƒí’ˆ

ì•„ëž˜ í˜•ì‹ìœ¼ë¡œ ì •í™•ížˆ ì¶œë ¥í•´ì£¼ì„¸ìš”:
1. [SKUì½”ë“œ] - [ìƒí’ˆëª…] - [ì„ ì • ê·¼ê±° í•œ ì¤„]
2. [SKUì½”ë“œ] - [ìƒí’ˆëª…] - [ì„ ì • ê·¼ê±° í•œ ì¤„]
3. [SKUì½”ë“œ] - [ìƒí’ˆëª…] - [ì„ ì • ê·¼ê±° í•œ ì¤„]

ì¶œë ¥:"""

    result = call_llm(client, provider, prompt)
    if not result:
        return products[:3]

    selected = []
    lines = result.split('\n')
    for line in lines:
        if line.strip() and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
            for p in products:
                sku_str = str(p['sku'])
                if sku_str in line:
                    selected.append({
                        'sku': p['sku'],
                        'sku_name': p['sku_name'],
                        'reason': line.split('-')[-1].strip() if '-' in line else ''
                    })
                    break

    while len(selected) < 3 and len(products) > len(selected):
        candidate = products[len(selected)]
        if candidate['sku'] not in [s['sku'] for s in selected]:
            selected.append({
                'sku': candidate['sku'],
                'sku_name': candidate['sku_name'],
                'reason': 'í‚¤ì›Œë“œ ë§¤ì¹­'
            })

    return selected[:3]


# ========== Step 5: ì²« ì£¼ í†µê³„ ê³„ì‚° ==========

def get_first_week_stats(sku_list: List[str], store: str = None) -> Dict[str, Dict]:
    """ê° SKUì˜ ì²« 7ì¼ íŒë§¤ í†µê³„ ê³„ì‚° (ë¡œì»¬ CSV)"""
    df = get_raw_sales_df()
    if df is None or df.empty:
        return {}

    # ì í¬ í•„í„°
    if store:
        df = df[df['STORE_CD'].astype(str) == str(store)]

    stats = {}

    for sku in sku_list:
        try:
            # í•´ë‹¹ SKU ë°ì´í„° í•„í„°
            sku_df = df[df['PRDT_CD'].astype(str) == str(sku)].copy()
            if sku_df.empty:
                continue

            # ë‚ ì§œ ì •ë ¬
            sku_df['SALE_DT'] = pd.to_datetime(sku_df['SALE_DT'], format='%Y%m%d')
            sku_df = sku_df.sort_values('SALE_DT')

            # ì²« 7ì¼ë§Œ ì¶”ì¶œ
            first_week = sku_df.head(7)

            if len(first_week) > 0:
                qty = first_week['SELL_QTY'].astype(float)
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


# ========== Step 6: ìµœì¢… ì˜ˆì¸¡ê°’ ì‚°ì¶œ (LLM) ==========

def generate_final_prediction(
    client,
    provider: str,
    input_name: str,
    group_name: str,
    top3: List[Dict],
    stats: Dict[str, Dict]
) -> str:
    """í†µê³„ ë°ì´í„° ê¸°ë°˜ ìµœì¢… ë°œì£¼ëŸ‰ ì˜ˆì¸¡"""

    stats_lines = []
    for item in top3:
        sku = item['sku']
        if sku in stats:
            s = stats[sku]
            stats_lines.append(
                f"| {item['sku_name'][:15]} | {s['mean']} | {s['min']} | {s['max']} | {s['std']} |"
            )

    stats_table = "\n".join(stats_lines) if stats_lines else "í†µê³„ ë°ì´í„° ì—†ìŒ"

    prompt = f"""ë‹¹ì‹ ì€ í˜„ëŒ€ë°±í™”ì  ì²­ê³¼ ë°œì£¼ ì „ë¬¸ê°€ë¥¼ ë•ê³  ìžˆìŠµë‹ˆë‹¤.
ì‹ ê·œ ìƒí’ˆ "{input_name}"ì˜ ì²« ì£¼ ì˜ˆìƒ ë°œì£¼ëŸ‰ì„ ì‚°ì¶œí•´ì£¼ì„¸ìš”.

í’ˆëª© ê·¸ë£¹: {group_name}

ìœ ì‚¬ ìƒí’ˆ ì²« ì£¼ íŒë§¤ í†µê³„:
| ìƒí’ˆëª… | í‰ê·  | ìµœì†Œ | ìµœëŒ€ | í‘œì¤€íŽ¸ì°¨ |
|--------|-----|-----|-----|---------|
{stats_table}

ë‹¤ìŒ í˜•ì‹ìœ¼ë¡œ ë‹µë³€í•´ì£¼ì„¸ìš” (ì£¼ì˜: ë²”ìœ„ í‘œì‹œ ì‹œ ~ ëŒ€ì‹  - ì‚¬ìš©):

â–¶ ì¶”ì²œ ë°œì£¼ëŸ‰: [í•˜í•œ]-[ìƒí•œ]ê°œ
  (ê·¼ê±°: [2-3ë¬¸ìž¥ìœ¼ë¡œ ê·¼ê±° ì„¤ëª…])

ì¶”ê°€ ê³ ë ¤ì‚¬í•­:
- [ì£¼ì˜ì‚¬í•­ì´ë‚˜ ì¡°ì • í•„ìš” ìƒí™© 1-2ê°œ]"""

    result = call_llm(client, provider, prompt)
    if result:
        return result

    # í´ë°±: í†µê³„ ê¸°ë°˜ ë‹¨ìˆœ ê³„ì‚°
    if stats:
        means = [s['mean'] for s in stats.values()]
        avg_mean = sum(means) / len(means)
        return f"""â–¶ ì¶”ì²œ ë°œì£¼ëŸ‰: {int(avg_mean * 0.8)}-{int(avg_mean * 1.2)}ê°œ
  (ê·¼ê±°: ìœ ì‚¬ ìƒí’ˆ ì²« ì£¼ í‰ê·  {avg_mean:.0f}ê°œ ê¸°ì¤€, ì‹ ìƒí’ˆ ì´ˆê¸° ë³´ìˆ˜ì  ì ‘ê·¼ ê¶Œìž¥)"""
    return "í†µê³„ ë°ì´í„°ê°€ ë¶€ì¡±í•˜ì—¬ ì˜ˆì¸¡ì´ ì–´ë µìŠµë‹ˆë‹¤. 10-20ê°œ ì†ŒëŸ‰ ë°œì£¼ í›„ ì¶”ì´ ê´€ì°°ì„ ê¶Œìž¥í•©ë‹ˆë‹¤."


# ========== ë©”ì¸ ì±—ë´‡ í•¨ìˆ˜ ==========

def run_cold_start_prediction(fruit_name: str, store: str = None) -> None:
    """Cold Start ì˜ˆì¸¡ ì‹¤í–‰

    Parameters
    ----------
    fruit_name : str
        ì‹ ê·œ ìƒí’ˆëª…
    store : str, optional
        ì í¬ ì½”ë“œ (Noneì´ë©´ ì „ì²´ ë°ì´í„°ì—ì„œ ê²€ìƒ‰)
    """
    client, provider = get_llm_client()
    if not client:
        st.error(f"AI ì—°ê²° ì‹¤íŒ¨. {provider.upper()} API í‚¤ë¥¼ í™•ì¸í•´ì£¼ì„¸ìš”.")
        return

    with st.status(f"'{fruit_name}' ë¶„ì„ ì¤‘...", expanded=True) as status:
        st.write(f"ðŸ¤– {provider.upper()} ëª¨ë¸ ì‚¬ìš©")
        if store:
            st.write(f"ðŸ¬ ì í¬: {store}")
        st.write("ðŸ“‹ í’ˆëª© ê·¸ë£¹ì„ ì°¾ê³  ìžˆìŠµë‹ˆë‹¤...")

        # Step 1: ì›¹ ê²€ìƒ‰ìœ¼ë¡œ ì •í™•í•œ ê³¼ì¼ ì •ë³´ íŒŒì•…
        st.write("ðŸ” ì›¹ ê²€ìƒ‰ìœ¼ë¡œ ê³¼ì¼ ì •ë³´ë¥¼ ì°¾ê³  ìžˆìŠµë‹ˆë‹¤...")
        group_name, fruit_info = infer_product_group_with_search(client, provider, fruit_name)

        if group_name:
            st.write(f"âœ… **'{group_name}'** ê·¸ë£¹ìœ¼ë¡œ ë¶„ë¥˜ë˜ì—ˆìŠµë‹ˆë‹¤.")
            if fruit_info:
                st.info(f"ðŸ“– {fruit_info}")
        else:
            # ì›¹ ê²€ìƒ‰ ì‹¤íŒ¨ ì‹œ LLM ì¶”ë¡ ìœ¼ë¡œ í´ë°±
            st.write("ðŸ“‹ LLM ì¶”ë¡ ìœ¼ë¡œ í’ˆëª© ê·¸ë£¹ì„ íŒë‹¨í•©ë‹ˆë‹¤...")
            group_name = infer_product_group(client, provider, fruit_name)

            if group_name:
                st.write(f"âœ… **'{group_name}'** ê·¸ë£¹ìœ¼ë¡œ ë¶„ë¥˜ë˜ì—ˆìŠµë‹ˆë‹¤.")
            else:
                # ë§ˆì§€ë§‰ ì‹œë„: raw_sales.csvì—ì„œ ìƒí’ˆëª… ì§ì ‘ ê²€ìƒ‰
                st.write("ðŸ”Ž ë°ì´í„°ë² ì´ìŠ¤ì—ì„œ ìƒí’ˆëª…ì„ ì§ì ‘ ê²€ìƒ‰í•©ë‹ˆë‹¤...")
                direct_products = search_similar_products([fruit_name], store)
                if direct_products:
                    st.write(f"âœ… **{len(direct_products)}ê°œ** ìœ ì‚¬ ìƒí’ˆì„ ì§ì ‘ ê²€ìƒ‰ìœ¼ë¡œ ì°¾ì•˜ìŠµë‹ˆë‹¤.")
                    # ì§ì ‘ ê²€ìƒ‰ ê²°ê³¼ë¡œ ì§„í–‰
                    top3 = select_top3_similar(client, provider, fruit_name, direct_products)
                    if top3:
                        st.write("**ìœ ì‚¬ ìƒí’ˆ Top 3:**")
                        for i, item in enumerate(top3, 1):
                            reason = item.get('reason', '')
                            st.write(f"   {i}. {item['sku_name']} - {reason}")

                        st.write("ðŸ“Š ì²« ì£¼ íŒë§¤ ë°ì´í„°ë¥¼ ë¶„ì„ ì¤‘ìž…ë‹ˆë‹¤...")
                        sku_list = [str(item['sku']) for item in top3]
                        stats = get_first_week_stats(sku_list, store)

                        if stats:
                            stats_data = []
                            for item in top3:
                                sku = str(item['sku'])
                                if sku in stats:
                                    s = stats[sku]
                                    stats_data.append({
                                        'ìƒí’ˆëª…': item['sku_name'][:20],
                                        'í‰ê· ': s['mean'],
                                        'ìµœì†Œ': s['min'],
                                        'ìµœëŒ€': s['max'],
                                        'í‘œì¤€íŽ¸ì°¨': s['std']
                                    })
                            if stats_data:
                                st.write("**ì²« ì£¼ íŒë§¤ í†µê³„:**")
                                st.dataframe(pd.DataFrame(stats_data), hide_index=True)

                        st.write("ðŸ§  ì˜ˆì¸¡ê°’ì„ ì‚°ì¶œ ì¤‘ìž…ë‹ˆë‹¤...")
                        prediction = generate_final_prediction(client, provider, fruit_name, "ì§ì ‘ê²€ìƒ‰", top3, stats)
                        status.update(label="ë¶„ì„ ì™„ë£Œ!", state="complete")

                        st.markdown("---")
                        st.markdown(f"### ðŸ“¦ [{fruit_name}] ì˜ˆì¸¡ ê²°ê³¼")
                        st.markdown("**í’ˆëª© ê·¸ë£¹:** ì§ì ‘ ê²€ìƒ‰ (ê·¸ë£¹ ë¯¸ë¶„ë¥˜)")
                        prediction_safe = prediction.replace("~", "-")
                        st.markdown(prediction_safe)
                        return

                st.write("âš ï¸ ë§¤ì¹­ë˜ëŠ” í’ˆëª© ê·¸ë£¹ì´ ì—†ìŠµë‹ˆë‹¤. ê¸°ë³¸ ì¶”ì²œìœ¼ë¡œ ì§„í–‰í•©ë‹ˆë‹¤.")
                st.warning("ì¶”ì²œ ë°œì£¼ëŸ‰: 10-20ê°œ (ì‹ ê·œ í’ˆëª©, ì†ŒëŸ‰ í…ŒìŠ¤íŠ¸ ê¶Œìž¥)")
                status.update(label="ë¶„ì„ ì™„ë£Œ", state="complete")
                return

        # Step 2: í‚¤ì›Œë“œ ì¶”ì¶œ
        keywords = get_keywords_for_group(group_name)
        st.write(f"ðŸ” ê²€ìƒ‰ í‚¤ì›Œë“œ: {', '.join(keywords[:5])}")

        # Step 3: ìœ ì‚¬ ìƒí’ˆ ê²€ìƒ‰ (ë¡œì»¬ CSV)
        st.write("ðŸ”Ž ìœ ì‚¬ ìƒí’ˆì„ ê²€ìƒ‰ ì¤‘ìž…ë‹ˆë‹¤...")
        similar_products = search_similar_products(keywords, store)

        if not similar_products:
            st.write("âš ï¸ ìœ ì‚¬ ìƒí’ˆì„ ì°¾ì„ ìˆ˜ ì—†ìŠµë‹ˆë‹¤.")
            st.info("ì¶”ì²œ ë°œì£¼ëŸ‰: 10-20ê°œ (ë°ì´í„° ë¶€ì¡±, ì†ŒëŸ‰ í…ŒìŠ¤íŠ¸ ê¶Œìž¥)")
            status.update(label="ë¶„ì„ ì™„ë£Œ", state="complete")
            return

        st.write(f"âœ… **{len(similar_products)}ê°œ** ìƒí’ˆì„ ì°¾ì•˜ìŠµë‹ˆë‹¤.")

        # Step 4: Top 3 ì„ ì •
        st.write("ðŸŽ¯ ê°€ìž¥ ìœ ì‚¬í•œ ìƒí’ˆ 3ê°œë¥¼ ì„ ì • ì¤‘ìž…ë‹ˆë‹¤...")
        top3 = select_top3_similar(client, provider, fruit_name, similar_products)

        if top3:
            st.write("**ìœ ì‚¬ ìƒí’ˆ Top 3:**")
            for i, item in enumerate(top3, 1):
                reason = item.get('reason', '')
                st.write(f"   {i}. {item['sku_name']} - {reason}")

        # Step 5: ì²« ì£¼ í†µê³„ ê³„ì‚° (ë¡œì»¬ CSV)
        st.write("ðŸ“Š ì²« ì£¼ íŒë§¤ ë°ì´í„°ë¥¼ ë¶„ì„ ì¤‘ìž…ë‹ˆë‹¤...")
        sku_list = [str(item['sku']) for item in top3]
        stats = get_first_week_stats(sku_list, store)

        if stats:
            stats_data = []
            for item in top3:
                sku = str(item['sku'])
                if sku in stats:
                    s = stats[sku]
                    stats_data.append({
                        'ìƒí’ˆëª…': item['sku_name'][:20],
                        'í‰ê· ': s['mean'],
                        'ìµœì†Œ': s['min'],
                        'ìµœëŒ€': s['max'],
                        'í‘œì¤€íŽ¸ì°¨': s['std']
                    })

            if stats_data:
                st.write("**ì²« ì£¼ íŒë§¤ í†µê³„:**")
                st.dataframe(pd.DataFrame(stats_data), hide_index=True)

        # Step 6: ìµœì¢… ì˜ˆì¸¡
        st.write("ðŸ§  ì˜ˆì¸¡ê°’ì„ ì‚°ì¶œ ì¤‘ìž…ë‹ˆë‹¤...")
        prediction = generate_final_prediction(client, provider, fruit_name, group_name, top3, stats)

        status.update(label="ë¶„ì„ ì™„ë£Œ!", state="complete")

    # ìµœì¢… ê²°ê³¼ í‘œì‹œ
    st.markdown("---")
    st.markdown(f"### ðŸ“¦ [{fruit_name}] ì˜ˆì¸¡ ê²°ê³¼")
    st.markdown(f"**í’ˆëª© ê·¸ë£¹:** {group_name}")
    prediction_safe = prediction.replace("~", "-")
    st.markdown(prediction_safe)


# ========== Modal Dialog ==========

STORE_OPTIONS = {
    "ì „ì²´": None,
    "210 (ë³¸ì )": "210",
    "220 (ë¬´ì—­ì„¼í„°ì )": "220",
    "480 (íŒêµì )": "480",
}


@st.dialog("ðŸ†• ì‹ ê·œ SKU ì˜ˆì¸¡", width="large")
def cold_start_dialog():
    """ì‹ ê·œ SKU ì˜ˆì¸¡ ëª¨ë‹¬ ë‹¤ì´ì–¼ë¡œê·¸"""
    st.markdown("íŒë§¤ ì´ë ¥ì´ ì—†ëŠ” ì‹ ê·œ ìƒí’ˆì˜ ì˜ˆìƒ ë°œì£¼ëŸ‰ì„ ì˜ˆì¸¡í•©ë‹ˆë‹¤.")
    st.markdown("ìœ ì‚¬ ìƒí’ˆì˜ ì²« ì£¼ íŒë§¤ ë°ì´í„°ë¥¼ ê¸°ë°˜ìœ¼ë¡œ ë¶„ì„í•©ë‹ˆë‹¤.")

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        fruit_name = st.text_input(
            "ì‹ ê·œ ìƒí’ˆëª…",
            placeholder="ì˜ˆ: íƒ€ì´ë°±ê·¤, ì¡°ì´ë² ë¦¬, ì²œì¤‘ë„...",
            key="cold_start_input"
        )
    with col2:
        store_label = st.selectbox(
            "ì í¬",
            options=list(STORE_OPTIONS.keys()),
            index=0,
            key="cold_start_store"
        )

    store = STORE_OPTIONS[store_label]

    col1, col2 = st.columns([1, 1])
    with col1:
        predict_btn = st.button("ðŸ”® ì˜ˆì¸¡í•˜ê¸°", type="primary", use_container_width=True)
    with col2:
        if st.button("ë‹«ê¸°", use_container_width=True):
            st.rerun()

    if predict_btn and fruit_name:
        run_cold_start_prediction(fruit_name.strip(), store)
    elif predict_btn and not fruit_name:
        st.warning("ìƒí’ˆëª…ì„ ìž…ë ¥í•´ì£¼ì„¸ìš”.")


def show_cold_start_button():
    """ì‹ ê·œ SKU ì˜ˆì¸¡ ë²„íŠ¼ í‘œì‹œ"""
    if st.button("ðŸ†• ì‹ ê·œ SKU ì˜ˆì¸¡", use_container_width=True):
        cold_start_dialog()