"""
Admin Schedule Calendar Component
==================================
ìŠ¤ì¼€ì¤„ ìº˜ë¦°ë”
- ì›”ê°„ ìº˜ë¦°ë” UI (T=Tuning, F=Fitting, P=Predicting)
- ë‚ ì§œë³„ ì˜ˆì•½ ê¸°ëŠ¥
- Tuning/Fitting/Predicting ì¦‰ì‹œ ì‹¤í–‰
"""

import streamlit as st
import json
import subprocess
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import calendar

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCHEDULED_TASKS_FILE = PROJECT_ROOT / "scheduled_tasks.json"


def load_scheduled_tasks() -> List[Dict]:
    """scheduled_tasks.json ë¡œë“œ"""
    if not SCHEDULED_TASKS_FILE.exists():
        return []

    try:
        with open(SCHEDULED_TASKS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('tasks', [])
    except Exception:
        return []


def save_scheduled_tasks(tasks: List[Dict]):
    """scheduled_tasks.json ì €ìž¥"""
    try:
        with open(SCHEDULED_TASKS_FILE, 'w', encoding='utf-8') as f:
            json.dump({'tasks': tasks}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"ì €ìž¥ ì‹¤íŒ¨: {e}")


def get_scheduled_task(date_str: str, store: str, include_deleted: bool = False) -> Optional[Dict]:
    """íŠ¹ì • ë‚ ì§œ/ì í¬ì˜ ì˜ˆì•½ ìž‘ì—… ì¡°íšŒ"""
    tasks = load_scheduled_tasks()
    for task in tasks:
        if task.get('date') == date_str and task.get('store') == store:
            # deletedëœ ìž‘ì—…ì€ ê¸°ë³¸ì ìœ¼ë¡œ ì œì™¸
            if not include_deleted and task.get('deleted', False):
                continue
            return task
    return None


def add_scheduled_task(date_str: str, store: str, mode: str):
    """ì˜ˆì•½ ìž‘ì—… ì¶”ê°€/ìˆ˜ì •/ì·¨ì†Œ"""
    tasks = load_scheduled_tasks()

    # ê¸°ì¡´ ìž‘ì—… ì°¾ê¸°
    existing_idx = None
    for i, t in enumerate(tasks):
        if t.get('date') == date_str and t.get('store') == store:
            existing_idx = i
            break

    if mode:  # ìƒˆ ì˜ˆì•½ ë˜ëŠ” ìˆ˜ì •
        new_task = {
            'date': date_str,
            'store': store,
            'mode': mode,
            'created_at': datetime.now().isoformat(),
            'created_by': 'admin',
            'deleted': False
        }
        if existing_idx is not None:
            tasks[existing_idx] = new_task
        else:
            tasks.append(new_task)
    else:  # ì·¨ì†Œ (deleted í”Œëž˜ê·¸ ì„¤ì •)
        if existing_idx is not None:
            tasks[existing_idx]['deleted'] = True
            tasks[existing_idx]['deleted_at'] = datetime.now().isoformat()

    save_scheduled_tasks(tasks)


def get_default_mode(d: date) -> str:
    """ë‚ ì§œ ê¸°ë°˜ ê¸°ë³¸ ëª¨ë“œ ê²°ì •"""
    if d.day == 1:
        return 'tuning'
    elif d.weekday() == 0:  # ì›”ìš”ì¼
        return 'fitting'
    else:
        return 'predicting'


def get_mode_for_date(date_str: str, store: str) -> str:
    """íŠ¹ì • ë‚ ì§œì˜ ì‹¤í–‰ ëª¨ë“œ (ì˜ˆì•½ ìš°ì„ , ì—†ìœ¼ë©´ ê¸°ë³¸)"""
    task = get_scheduled_task(date_str, store)
    if task:
        return task.get('mode', '')

    d = datetime.strptime(date_str, '%Y-%m-%d').date()
    return get_default_mode(d)


def get_mode_display(mode: str) -> str:
    """ëª¨ë“œ í‘œì‹œ ë¬¸ìž"""
    if mode == 'tuning':
        return 'T'
    elif mode == 'fitting':
        return 'F'
    elif mode == 'predicting':
        return 'P'
    return ''


def get_mode_color(mode: str) -> str:
    """ëª¨ë“œë³„ ìƒ‰ìƒ"""
    if mode == 'tuning':
        return '#FF6347'  # ë¹¨ê°•
    elif mode == 'fitting':
        return '#4169E1'  # íŒŒëž‘
    elif mode == 'predicting':
        return '#32CD32'  # ì´ˆë¡
    return '#888'


def run_pipeline_async(store: str, mode: str, date_str: str, skip_extract: bool = False, skip_preprocess: bool = False):
    """íŒŒì´í”„ë¼ì¸ ë¹„ë™ê¸° ì‹¤í–‰ (Tuning/Fitting/Predicting)"""
    python_exe = r"C:\ml_env\.venv\Scripts\python.exe"
    scheduler_path = PROJECT_ROOT / "scheduler.py"

    cmd = [
        python_exe,
        str(scheduler_path),
        "--mode", mode,
        "--store", store,
        "--date", date_str
    ]

    if skip_extract:
        cmd.append("--skip-extract")
    if skip_preprocess:
        cmd.append("--skip-preprocess")

    try:
        # ë°±ê·¸ë¼ìš´ë“œ ì‹¤í–‰
        subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )
        skip_msg = " (ì¶”ì¶œ/ì „ì²˜ë¦¬ ìƒëžµ)" if (skip_extract or skip_preprocess) else ""
        st.success(f"âœ… {mode.title()} íŒŒì´í”„ë¼ì¸ì´ ì‹œìž‘ë˜ì—ˆìŠµë‹ˆë‹¤.{skip_msg} (ì í¬: {store})")
        st.info("ì§„í–‰ ìƒí™©ì€ 'íŒŒì´í”„ë¼ì¸ ìƒíƒœ'ì—ì„œ í™•ì¸í•  ìˆ˜ ìžˆìŠµë‹ˆë‹¤.")
    except Exception as e:
        st.error(f"ì‹¤í–‰ ì‹¤íŒ¨: {e}")


def render_calendar_grid(store: str, year: int, month: int):
    """ì›”ê°„ ìº˜ë¦°ë” ê·¸ë¦¬ë“œ ë Œë”ë§"""
    cal = calendar.Calendar(firstweekday=6)  # ì¼ìš”ì¼ ì‹œìž‘
    month_days = cal.monthdayscalendar(year, month)

    # ìš”ì¼ í—¤ë”
    weekdays = ['ì¼', 'ì›”', 'í™”', 'ìˆ˜', 'ëª©', 'ê¸ˆ', 'í† ']
    header_cols = st.columns(7)
    for i, wd in enumerate(weekdays):
        with header_cols[i]:
            color = '#FF6347' if i == 0 else ('#4169E1' if i == 6 else '#333')
            st.markdown(f"<div style='text-align:center;color:{color};font-weight:bold;'>{wd}</div>", unsafe_allow_html=True)

    # ë‚ ì§œ ê·¸ë¦¬ë“œ
    today = date.today()

    for week in month_days:
        cols = st.columns(7)
        for i, day in enumerate(week):
            with cols[i]:
                if day == 0:
                    st.markdown("<div style='height:60px;'></div>", unsafe_allow_html=True)
                else:
                    d = date(year, month, day)
                    date_str = d.strftime('%Y-%m-%d')
                    mode = get_mode_for_date(date_str, store)
                    mode_display = get_mode_display(mode)
                    mode_color = get_mode_color(mode)

                    # ì˜¤ëŠ˜ í‘œì‹œ
                    is_today = d == today
                    border = "2px solid #333" if is_today else "1px solid #ddd"
                    bg_color = "#fffacd" if is_today else "#fff"

                    # ì˜ˆì•½ëœ ìž‘ì—… í‘œì‹œ
                    task = get_scheduled_task(date_str, store)
                    is_scheduled = task is not None

                    st.markdown(f"""
                        <div style='
                            border:{border};
                            border-radius:5px;
                            padding:5px;
                            text-align:center;
                            height:60px;
                            background-color:{bg_color};
                        '>
                            <div style='font-size:14px;'>{day}</div>
                            <div style='
                                font-size:18px;
                                font-weight:bold;
                                color:{mode_color};
                            '>{mode_display}{'*' if is_scheduled else ''}</div>
                        </div>
                    """, unsafe_allow_html=True)


def render_schedule_calendar(store: str, month: str):
    """ìŠ¤ì¼€ì¤„ ìº˜ë¦°ë” ì»´í¬ë„ŒíŠ¸ ë Œë”ë§"""
    st.markdown("### ðŸ“… ìŠ¤ì¼€ì¤„ ìº˜ë¦°ë”")

    # ë²”ë¡€
    st.markdown("""
        <div style='display:flex;gap:20px;margin-bottom:10px;'>
            <span><b style='color:#FF6347;'>T</b> = Tuning (ë§¤ì›” 1ì¼)</span>
            <span><b style='color:#4169E1;'>F</b> = Fitting (ë§¤ì£¼ ì›”ìš”ì¼)</span>
            <span><b style='color:#32CD32;'>P</b> = Predicting (ë§¤ì¼)</span>
            <span>* = ì˜ˆì•½ë¨</span>
        </div>
    """, unsafe_allow_html=True)

    # ì›” íŒŒì‹±
    year, mon = map(int, month.split('-'))

    # ìº˜ë¦°ë” ê·¸ë¦¬ë“œ
    st.markdown(f"#### {year}ë…„ {mon}ì›”")
    render_calendar_grid(store, year, mon)

    # ìž‘ì—… ì˜ˆì•½/ì‹¤í–‰ UI
    st.markdown("#### ìž‘ì—… ê´€ë¦¬")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        selected_date = st.date_input(
            "ë‚ ì§œ ì„ íƒ",
            value=date.today(),
            key="calendar_date_select"
        )

    with col2:
        current_task = get_scheduled_task(selected_date.strftime('%Y-%m-%d'), store)
        current_mode = current_task.get('mode') if current_task else ''

        mode_options = ['', 'tuning', 'fitting', 'predicting']
        mode_labels = ['ê¸°ë³¸ ìŠ¤ì¼€ì¤„', 'Tuning', 'Fitting', 'Predicting']

        selected_mode = st.selectbox(
            "ì˜ˆì•½ ëª¨ë“œ",
            options=mode_options,
            format_func=lambda x: mode_labels[mode_options.index(x)],
            index=mode_options.index(current_mode) if current_mode in mode_options else 0,
            key="calendar_mode_select"
        )

    with col3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if st.button("ì˜ˆì•½ ì €ìž¥", key="save_schedule"):
            add_scheduled_task(
                selected_date.strftime('%Y-%m-%d'),
                store,
                selected_mode
            )
            st.success("ì˜ˆì•½ì´ ì €ìž¥ë˜ì—ˆìŠµë‹ˆë‹¤.")
            st.rerun()

    # ì¦‰ì‹œ ì‹¤í–‰ (ê¸ˆì¼)
    st.markdown("#### ì¦‰ì‹œ ì‹¤í–‰ (ê¸ˆì¼)")

    today_str = date.today().strftime('%Y-%m-%d')

    # ì²´í¬ë°•ìŠ¤: ì¶”ì¶œ/ì „ì²˜ë¦¬ í¬í•¨ ì—¬ë¶€ (ê¸°ë³¸ ì²´í¬ í•´ì œ = skip)
    check_col1, check_col2 = st.columns(2)
    with check_col1:
        include_extract = st.checkbox("ì¶”ì¶œ í¬í•¨", value=False, key="include_extract")
    with check_col2:
        include_preprocess = st.checkbox("ì „ì²˜ë¦¬ í¬í•¨", value=False, key="include_preprocess")

    skip_extract = not include_extract
    skip_preprocess = not include_preprocess

    # ì‹¤í–‰ ë²„íŠ¼
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("ðŸŽ¯ Tuning"):
            run_pipeline_async(store, 'tuning', today_str, skip_extract=skip_extract, skip_preprocess=skip_preprocess)
        st.caption("ì•½ 3ì‹œê°„")

    with col2:
        if st.button("ðŸ”„ Fitting"):
            run_pipeline_async(store, 'fitting', today_str, skip_extract=skip_extract, skip_preprocess=skip_preprocess)
        st.caption("ì•½ 5ë¶„")

    with col3:
        if st.button("ðŸ“Š Predicting"):
            run_pipeline_async(store, 'predicting', today_str, skip_extract=skip_extract, skip_preprocess=skip_preprocess)
        st.caption("ì•½ 5ë¶„")

    # ì˜ˆì•½ëœ ìž‘ì—… ëª©ë¡
    tasks = load_scheduled_tasks()
    store_tasks = [t for t in tasks if t.get('store') == store and not t.get('deleted', False)]

    if store_tasks:
        st.markdown("#### ì˜ˆì•½ëœ ìž‘ì—…")
        for task in sorted(store_tasks, key=lambda x: x['date']):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"ðŸ“… {task['date']}")
            with col2:
                st.markdown(f"**{task['mode'].title()}**")
            with col3:
                if st.button("âŒ", key=f"del_{task['date']}_{task['store']}"):
                    add_scheduled_task(task['date'], store, '')
                    st.rerun()