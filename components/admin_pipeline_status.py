"""
Admin Pipeline Status Component
================================
íŒŒì´í”„ë¼ì¸ ìƒíƒœ í‘œì‹œ (í† ê¸€ í˜•íƒœ)
- ì í¬ë³„ ë¶„ë¦¬
- ë‹¨ê³„ë³„ ìƒíƒœ + ì†Œìš” ì‹œê°„: ì¶”ì¶œ, ì „ì²˜ë¦¬, íŠœë‹, í•™ìŠµ, ì˜ˆì¸¡
- ê²½ê³ /ì—ëŸ¬ í‘œì‹œ
- ê°™ì€ ë‚  ì—¬ëŸ¬ ì‹¤í–‰ ì§€ì› (ì¶”ê°€ ì‹¤í–‰ í‘œì‹œ)
"""

import streamlit as st
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# ìž‘ì—… ìš°ì„ ìˆœìœ„ (ë†’ì„ìˆ˜ë¡ ìƒìœ„)
MODE_PRIORITY = {'tuning': 3, 'fitting': 2, 'predicting': 1}


def get_available_log_dates() -> List[str]:
    """ë¡œê·¸ íŒŒì¼ì´ ì¡´ìž¬í•˜ëŠ” ë‚ ì§œ ëª©ë¡ ë°˜í™˜ (ìµœì‹ ìˆœ ì •ë ¬)"""
    if not LOGS_DIR.exists():
        return []

    dates = []
    for log_file in LOGS_DIR.glob("scheduler_[0-9]*.log"):
        filename = log_file.stem
        if filename.startswith("scheduler_") and not filename.startswith("scheduler_runner"):
            date_str = filename.replace("scheduler_", "")
            if len(date_str) == 8 and date_str.isdigit():
                formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                dates.append(formatted)

    dates.sort(reverse=True)
    return dates


def get_log_file_by_date(date_str: str) -> Optional[Path]:
    """íŠ¹ì • ë‚ ì§œì˜ scheduler ë¡œê·¸ íŒŒì¼ ë°˜í™˜"""
    if not LOGS_DIR.exists():
        return None

    date_key = date_str.replace('-', '')
    log_path = LOGS_DIR / f"scheduler_{date_key}.log"

    if log_path.exists():
        return log_path
    return None


def _extract_timestamp(line: str) -> Optional[str]:
    """ë¡œê·¸ ë¼ì¸ì—ì„œ íƒ€ìž„ìŠ¤íƒ¬í”„ ì¶”ì¶œ"""
    match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
    return match.group(1) if match else None


def _parse_pipeline_runs(log_path: Path) -> List[Dict]:
    """
    ë¡œê·¸ íŒŒì¼ì—ì„œ ëª¨ë“  íŒŒì´í”„ë¼ì¸ ì‹¤í–‰ ë¸”ë¡ì„ íŒŒì‹±

    ê° ë¸”ë¡: "íŒŒì´í”„ë¼ì¸ ì‹œìž‘" ~ "íŒŒì´í”„ë¼ì¸ ì™„ë£Œ"

    Returns:
        List of runs, each containing:
        - start_time, end_time
        - global_mode (ì í¬ë³„ìžë™, tuning, fitting, predicting)
        - target_store (all or specific store)
        - lines: í•´ë‹¹ ë¸”ë¡ì˜ ë¼ì¸ë“¤ (start_idx, end_idx)
    """
    if not log_path or not log_path.exists():
        return []

    runs = []

    try:
        content = log_path.read_text(encoding='utf-8')
        lines = content.split('\n')

        current_run = None

        for i, line in enumerate(lines):
            # íŒŒì´í”„ë¼ì¸ ì‹œìž‘ ê°ì§€
            if '[scheduler]' in line and 'íŒŒì´í”„ë¼ì¸ ì‹œìž‘' in line:
                timestamp = _extract_timestamp(line)

                # mode íŒŒì‹±
                mode_match = re.search(r'mode=([^\s,]+)', line)
                global_mode = mode_match.group(1).strip() if mode_match else 'unknown'

                # target store íŒŒì‹±
                store_match = re.search(r'store=([^\s,]+)', line)
                target_store = store_match.group(1).strip() if store_match else 'all'

                current_run = {
                    'start_time': timestamp,
                    'end_time': None,
                    'global_mode': global_mode,
                    'target_store': target_store,
                    'start_idx': i,
                    'end_idx': None,
                    'completed': False,
                }

            # íŒŒì´í”„ë¼ì¸ ì™„ë£Œ ê°ì§€
            if current_run and '[scheduler]' in line and 'íŒŒì´í”„ë¼ì¸ ì™„ë£Œ' in line:
                timestamp = _extract_timestamp(line)
                current_run['end_time'] = timestamp
                current_run['end_idx'] = i
                current_run['completed'] = True
                runs.append(current_run)
                current_run = None

        # ì™„ë£Œë˜ì§€ ì•Šì€ ì‹¤í–‰ (ì§„í–‰ ì¤‘)
        if current_run:
            current_run['end_idx'] = len(lines) - 1
            runs.append(current_run)

    except Exception:
        pass

    return runs


def _parse_store_processing(lines: List[str], start_idx: int, end_idx: int, store: str) -> Optional[Dict]:
    """
    íŠ¹ì • íŒŒì´í”„ë¼ì¸ ë¸”ë¡ì—ì„œ ì í¬ë³„ ì²˜ë¦¬ ì •ë³´ ì¶”ì¶œ

    "ì²˜ë¦¬ ì¤‘: ì í¬ XXX (mode=YYY)" ~ "[ì í¬ XXX] ì²˜ë¦¬ ì™„ë£Œ âœ“"

    Returns:
        {
            'mode': 'tuning/fitting/predicting',
            'store_start': timestamp,
            'store_end': timestamp,
            'tuning': {'start_time', 'end_time', 'status'},
            'fitting': {'start_time', 'end_time', 'status'},
            'predict': {'start_time', 'end_time', 'status'},
        }
        or None if store not processed in this block
    """
    result = {
        'mode': None,
        'store_start': None,
        'store_end': None,
        'tuning': {'start_time': None, 'end_time': None, 'status': 'pending'},
        'fitting': {'start_time': None, 'end_time': None, 'status': 'pending'},
        'predict': {'start_time': None, 'end_time': None, 'status': 'pending'},
    }

    in_store_section = False

    for i in range(start_idx, min(end_idx + 1, len(lines))):
        line = lines[i]
        timestamp = _extract_timestamp(line)

        # ì í¬ ì²˜ë¦¬ ì‹œìž‘
        if f'ì²˜ë¦¬ ì¤‘: ì í¬ {store}' in line:
            in_store_section = True
            result['store_start'] = timestamp

            mode_match = re.search(r'mode=(\w+)', line)
            if mode_match:
                result['mode'] = mode_match.group(1)

        # ë‹¤ë¥¸ ì í¬ ì‹œìž‘ = í˜„ìž¬ ì í¬ ì„¹ì…˜ ë
        if in_store_section and 'ì²˜ë¦¬ ì¤‘: ì í¬' in line and f'ì²˜ë¦¬ ì¤‘: ì í¬ {store}' not in line:
            in_store_section = False

        # ì í¬ ì²˜ë¦¬ ì™„ë£Œ
        if f'[ì í¬ {store}] ì²˜ë¦¬ ì™„ë£Œ' in line:
            result['store_end'] = timestamp
            in_store_section = False

        # ì í¬ ì„¹ì…˜ ë‚´ì—ì„œ íŠœë‹/í•™ìŠµ/ì˜ˆì¸¡ íŒŒì‹±
        if in_store_section or (result['store_start'] and not result['store_end']):
            # íŠœë‹ ì‹œìž‘ (scheduler ë ˆë²¨)
            if '[scheduler]' in line and '[íŠœë‹]' in line and f'store={store}' in line:
                result['tuning']['start_time'] = timestamp
            # íŠœë‹ ì™„ë£Œ
            if '[scheduler]' in line and '[íŠœë‹] ì™„ë£Œ' in line:
                result['tuning']['end_time'] = timestamp
                result['tuning']['status'] = 'success'
            if '[íŠœë‹] ì—ëŸ¬' in line:
                result['tuning']['end_time'] = timestamp
                result['tuning']['status'] = 'error'

            # í•™ìŠµ ì‹œìž‘
            if '[scheduler]' in line and '[í•™ìŠµ]' in line and f'store={store}' in line:
                result['fitting']['start_time'] = timestamp
            # í•™ìŠµ ì™„ë£Œ
            if '[scheduler]' in line and '[í•™ìŠµ] ì™„ë£Œ' in line:
                result['fitting']['end_time'] = timestamp
                result['fitting']['status'] = 'success'
            if '[í•™ìŠµ] ì—ëŸ¬' in line:
                result['fitting']['end_time'] = timestamp
                result['fitting']['status'] = 'error'

            # ì˜ˆì¸¡ ì‹œìž‘
            if '[scheduler]' in line and '[ì˜ˆì¸¡]' in line and f'store={store}' in line:
                result['predict']['start_time'] = timestamp
            # ì˜ˆì¸¡ ì™„ë£Œ
            if '[scheduler]' in line and '[ì˜ˆì¸¡] ì™„ë£Œ' in line:
                result['predict']['end_time'] = timestamp
                result['predict']['status'] = 'success'
            if '[ì˜ˆì¸¡] ì—ëŸ¬' in line:
                result['predict']['end_time'] = timestamp
                result['predict']['status'] = 'error'

    # ì í¬ê°€ ì²˜ë¦¬ë˜ì§€ ì•Šì•˜ìœ¼ë©´ None ë°˜í™˜
    if not result['store_start']:
        return None

    return result


def _parse_shared_stages(lines: List[str], start_idx: int, end_idx: int) -> Dict:
    """
    ê³µìœ  ë‹¨ê³„ (ì¶”ì¶œ, ì „ì²˜ë¦¬) íŒŒì‹±
    """
    result = {
        'extract': {'start_time': None, 'end_time': None, 'status': 'pending'},
        'preprocess': {'start_time': None, 'end_time': None, 'status': 'pending'},
    }

    for i in range(start_idx, min(end_idx + 1, len(lines))):
        line = lines[i]
        timestamp = _extract_timestamp(line)

        # ì¶”ì¶œ
        if '[ì¶”ì¶œ] ë°ì´í„° ì¶”ì¶œ ì‹œìž‘' in line:
            result['extract']['start_time'] = timestamp
        if '[scheduler]' in line and '[ì¶”ì¶œ] ì™„ë£Œ' in line:
            result['extract']['end_time'] = timestamp
            result['extract']['status'] = 'success'
        if '[ì¶”ì¶œ] ì‹¤íŒ¨' in line or ('[ì¶”ì¶œ]' in line and '[ERROR]' in line):
            result['extract']['end_time'] = timestamp
            result['extract']['status'] = 'error'

        # ì „ì²˜ë¦¬
        if '[ì „ì²˜ë¦¬] ë°ì´í„° ì „ì²˜ë¦¬ ì‹œìž‘' in line:
            result['preprocess']['start_time'] = timestamp
        if '[scheduler]' in line and '[ì „ì²˜ë¦¬] ì™„ë£Œ' in line:
            result['preprocess']['end_time'] = timestamp
            result['preprocess']['status'] = 'success'
        if '[ì „ì²˜ë¦¬] ì‹¤íŒ¨' in line:
            result['preprocess']['end_time'] = timestamp
            result['preprocess']['status'] = 'error'

    return result


def _collect_warnings_errors(lines: List[str], start_idx: int, end_idx: int, store: str = None) -> Tuple[List[str], List[str]]:
    """ê²½ê³ /ì—ëŸ¬ ìˆ˜ì§‘"""
    warnings = []
    errors = []

    in_store_section = False if store else True

    for i in range(start_idx, min(end_idx + 1, len(lines))):
        line = lines[i]

        if store:
            if f'ì²˜ë¦¬ ì¤‘: ì í¬ {store}' in line:
                in_store_section = True
            elif 'ì²˜ë¦¬ ì¤‘: ì í¬' in line and f'ì²˜ë¦¬ ì¤‘: ì í¬ {store}' not in line:
                in_store_section = False

        # ê³µìœ  ë‹¨ê³„ ë˜ëŠ” ì í¬ ì„¹ì…˜ ë‚´ì—ì„œë§Œ ìˆ˜ì§‘
        is_shared = store and not in_store_section and i < start_idx + 200  # ì•žë¶€ë¶„ì€ ê³µìœ  ë‹¨ê³„

        if in_store_section or is_shared or not store:
            if '[WARNING]' in line:
                msg_match = re.search(r'\[WARNING\]\s*(?:\[[^\]]+\])?\s*(.+)$', line)
                if msg_match:
                    msg = msg_match.group(1).strip()
                    if msg and len(msg) < 200:
                        warnings.append(msg)

            if '[ERROR]' in line:
                msg_match = re.search(r'\[ERROR\]\s*(?:\[[^\]]+\])?\s*(.+)$', line)
                if msg_match:
                    msg = msg_match.group(1).strip()
                    if msg and len(msg) < 200:
                        errors.append(msg)

    return warnings, errors


def parse_log_file(log_path: Path, store: str = None) -> Dict:
    """
    ë¡œê·¸ íŒŒì¼ íŒŒì‹± (ì í¬ë³„, ë‹¤ì¤‘ ì‹¤í–‰ ì§€ì›)

    ê°™ì€ ë‚  ì—¬ëŸ¬ ë²ˆ ì‹¤í–‰ëœ ê²½ìš°:
    - ì¶”ì¶œ/ì „ì²˜ë¦¬: ë‹¹ì¼ ëª¨ë“  ì‹¤í–‰ ì¤‘ ê°€ìž¥ ìµœê·¼ ì„±ê³µí•œ ê²ƒ (ì í¬ ë¬´ê´€, ê³µìœ  ë‹¨ê³„)
    - íŠœë‹/í•™ìŠµ/ì˜ˆì¸¡: í•´ë‹¹ ì í¬ê°€ ì²˜ë¦¬ëœ ì‹¤í–‰ ì¤‘ ìš°ì„ ìˆœìœ„ê°€ ê°€ìž¥ ë†’ì€ ê²ƒ
    - ë‚˜ë¨¸ì§€ëŠ” additional_runsë¡œ ë°˜í™˜
    """
    result = {
        'date': None,
        'mode': None,
        'store_mode': None,
        'mode_detail': None,
        'extract': {'start_time': None, 'end_time': None, 'status': 'pending'},
        'preprocess': {'start_time': None, 'end_time': None, 'status': 'pending'},
        'tuning': {'start_time': None, 'end_time': None, 'status': 'pending'},
        'fitting': {'start_time': None, 'end_time': None, 'status': 'pending'},
        'predict': {'start_time': None, 'end_time': None, 'status': 'pending'},
        'completed': False,
        'warnings': [],
        'errors': [],
        'last_run_start': None,
        'last_run_end': None,
        'additional_runs': [],
    }

    if not log_path or not log_path.exists():
        return result

    try:
        content = log_path.read_text(encoding='utf-8')
        lines = content.split('\n')

        # 1. ëª¨ë“  íŒŒì´í”„ë¼ì¸ ì‹¤í–‰ ë¸”ë¡ íŒŒì‹±
        runs = _parse_pipeline_runs(log_path)

        if not runs:
            return result

        # 2. ëª¨ë“  ì‹¤í–‰ì—ì„œ ê³µìœ  ë‹¨ê³„(ì¶”ì¶œ/ì „ì²˜ë¦¬) íŒŒì‹± â†’ ê°€ìž¥ ìµœê·¼ ì„±ê³µí•œ ê²ƒ ì„ íƒ
        all_shared = []
        for run in runs:
            shared = _parse_shared_stages(lines, run['start_idx'], run['end_idx'])
            all_shared.append({
                'run': run,
                'shared': shared,
            })

        # ì¶”ì¶œ: ê°€ìž¥ ìµœê·¼ ì„±ê³µí•œ ê²ƒ (ì‹œê°„ìˆœ ì—­ì •ë ¬ í›„ ì²« ë²ˆì§¸ success)
        latest_extract = {'start_time': None, 'end_time': None, 'status': 'pending'}
        for item in sorted(all_shared, key=lambda x: x['run']['start_time'] or '', reverse=True):
            if item['shared']['extract']['status'] == 'success':
                latest_extract = item['shared']['extract']
                break

        # ì „ì²˜ë¦¬: ê°€ìž¥ ìµœê·¼ ì„±ê³µí•œ ê²ƒ
        latest_preprocess = {'start_time': None, 'end_time': None, 'status': 'pending'}
        for item in sorted(all_shared, key=lambda x: x['run']['start_time'] or '', reverse=True):
            if item['shared']['preprocess']['status'] == 'success':
                latest_preprocess = item['shared']['preprocess']
                break

        # 3. í•´ë‹¹ ì í¬ê°€ ì²˜ë¦¬ëœ ì‹¤í–‰ ë¸”ë¡ë§Œ í•„í„°ë§ + ìƒì„¸ íŒŒì‹±
        store_runs = []
        for run in runs:
            # ì í¬ê°€ ì²˜ë¦¬ë˜ì—ˆëŠ”ì§€ í™•ì¸
            store_info = _parse_store_processing(lines, run['start_idx'], run['end_idx'], store) if store else None

            if store and not store_info:
                # ì´ ì‹¤í–‰ì—ì„œ í•´ë‹¹ ì í¬ê°€ ì²˜ë¦¬ë˜ì§€ ì•ŠìŒ
                continue

            # í•´ë‹¹ ì‹¤í–‰ì˜ ê³µìœ  ë‹¨ê³„ íŒŒì‹±
            shared = _parse_shared_stages(lines, run['start_idx'], run['end_idx'])

            # ê²½ê³ /ì—ëŸ¬ ìˆ˜ì§‘
            warnings, errors = _collect_warnings_errors(lines, run['start_idx'], run['end_idx'], store)

            store_runs.append({
                'run': run,
                'store_info': store_info,
                'shared': shared,
                'warnings': warnings,
                'errors': errors,
            })

        if not store_runs:
            # ì í¬ ì²˜ë¦¬ ê¸°ë¡ì€ ì—†ì§€ë§Œ ì¶”ì¶œ/ì „ì²˜ë¦¬ëŠ” ìžˆì„ ìˆ˜ ìžˆìŒ
            if latest_extract['status'] == 'success' or latest_preprocess['status'] == 'success':
                result['extract'] = latest_extract
                result['preprocess'] = latest_preprocess
                if runs:
                    result['date'] = runs[0]['start_time'].split(' ')[0] if runs[0]['start_time'] else None
            return result

        # 4. ìš°ì„ ìˆœìœ„ë¡œ ë©”ì¸ ì‹¤í–‰ ì„ íƒ
        def get_priority(sr):
            mode = sr['store_info']['mode'] if sr['store_info'] else sr['run']['global_mode']
            if mode == 'ì í¬ë³„ìžë™':
                mode = 'predicting'
            return MODE_PRIORITY.get(mode, 0)

        store_runs.sort(key=lambda x: (get_priority(x), x['run']['start_time']), reverse=True)

        main_run = store_runs[0]
        additional = store_runs[1:] if len(store_runs) > 1 else []

        # 5. ê²°ê³¼ êµ¬ì„±
        run = main_run['run']
        store_info = main_run['store_info']
        main_shared = main_run['shared']

        result['date'] = run['start_time'].split(' ')[0] if run['start_time'] else None
        result['mode'] = run['global_mode']
        result['store_mode'] = store_info['mode'] if store_info else None
        result['completed'] = run['completed']
        result['last_run_start'] = run['start_time']
        result['last_run_end'] = run['end_time']

        # ê³µìœ  ë‹¨ê³„: ë©”ì¸ ì‹¤í–‰ì—ì„œ ìžˆìœ¼ë©´ ì‚¬ìš©, ì—†ìœ¼ë©´ fallback
        if main_shared['extract']['status'] == 'success':
            result['extract'] = main_shared['extract']
        else:
            result['extract'] = latest_extract  # fallback

        if main_shared['preprocess']['status'] == 'success':
            result['preprocess'] = main_shared['preprocess']
        else:
            result['preprocess'] = latest_preprocess  # fallback

        # ì í¬ë³„ ë‹¨ê³„: ë©”ì¸ ì‹¤í–‰ì—ì„œ ê°€ì ¸ì˜´
        if store_info:
            result['tuning'] = store_info['tuning']
            result['fitting'] = store_info['fitting']
            result['predict'] = store_info['predict']

        # ê²½ê³ /ì—ëŸ¬
        result['warnings'] = main_run['warnings']
        result['errors'] = main_run['errors']

        # ì¶”ê°€ ì‹¤í–‰ ì •ë³´
        for add_run in additional:
            add_store_info = add_run['store_info']
            add_time = add_store_info['store_start'] if add_store_info else add_run['run']['start_time']
            add_mode = add_store_info['mode'] if add_store_info else add_run['run']['global_mode']

            if add_mode == 'ì í¬ë³„ìžë™':
                add_mode = 'predicting'

            if add_time:
                time_str = add_time.split(' ')[1][:5] if ' ' in add_time else add_time[:5]
                result['additional_runs'].append({
                    'time': time_str,
                    'mode': add_mode.title() if add_mode else '-',
                })

        # ì‹œê°„ìˆœ ì •ë ¬ (ì˜¤ëž˜ëœ ê²ƒ ë¨¼ì €)
        result['additional_runs'].sort(key=lambda x: x['time'])

    except Exception:
        pass

    return result


def _load_scheduled_tasks() -> Dict:
    """scheduled_tasks.json ë¡œë“œ (deleted ì œì™¸)"""
    import json
    scheduled_file = PROJECT_ROOT / "scheduled_tasks.json"

    if not scheduled_file.exists():
        return {}

    try:
        with open(scheduled_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            tasks_by_date = {}
            for task in data.get('tasks', []):
                if task.get('deleted', False):
                    continue
                date = task.get('date')
                if date:
                    if date not in tasks_by_date:
                        tasks_by_date[date] = []
                    tasks_by_date[date].append(task)
            return tasks_by_date
    except Exception:
        return {}


def _compute_mode_detail(result: Dict, store: str = None) -> str:
    """ì‹¤í–‰ ëª¨ë“œ ìƒì„¸ ì •ë³´ ê³„ì‚°"""
    mode = result.get('mode', '')
    date = result.get('date')
    store_mode = result.get('store_mode')

    # ì‹¤ì œ ì‹¤í–‰ëœ ëª¨ë“œ
    if store_mode:
        actual_mode = store_mode
    else:
        tuning_ran = result['tuning']['status'] != 'pending'
        fitting_ran = result['fitting']['status'] != 'pending'
        predict_ran = result['predict']['status'] != 'pending'

        if tuning_ran:
            actual_mode = 'tuning'
        elif fitting_ran:
            actual_mode = 'fitting'
        elif predict_ran:
            actual_mode = 'predicting'
        else:
            actual_mode = None

    # ì˜ˆì•½ëœ ìž‘ì—… í™•ì¸
    scheduled_tasks = _load_scheduled_tasks()
    has_scheduled = False
    scheduled_mode = None

    if date and date in scheduled_tasks and store:
        for task in scheduled_tasks[date]:
            if task.get('store') == store:
                has_scheduled = True
                scheduled_mode = task.get('mode', actual_mode)
                break

    # mode_detail ê²°ì •
    if mode == 'ì í¬ë³„ìžë™':
        if has_scheduled:
            return f"ì˜ˆì•½ ({scheduled_mode})"
        elif actual_mode:
            return f"ìžë™ ({actual_mode})"
        else:
            return "ìžë™ (-)"
    elif '(ìžë™)' in mode:
        base_mode = mode.replace('(ìžë™)', '').strip()
        if has_scheduled:
            return f"ì˜ˆì•½ ({base_mode})"
        return f"ìžë™ ({base_mode})"
    elif mode in ['tuning', 'fitting', 'predicting']:
        return mode
    else:
        return mode or '-'


def calculate_duration(start_time: str, end_time: str) -> str:
    """ì†Œìš” ì‹œê°„ ê³„ì‚°"""
    if not start_time or not end_time:
        return ""

    try:
        start = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
        end = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
        duration = end - start
        total_seconds = int(duration.total_seconds())

        if total_seconds < 0:
            return ""
        elif total_seconds < 60:
            return f"{total_seconds}ì´ˆ"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}ë¶„ {seconds}ì´ˆ"
        else:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}ì‹œê°„ {minutes}ë¶„"
    except Exception:
        return ""


def get_status_style(status: str) -> tuple:
    """ìƒíƒœë³„ ìŠ¤íƒ€ì¼ (emoji, bg_color, text_color)"""
    if status == 'success':
        return ('âœ…', '#d4edda', '#155724')
    elif status == 'error':
        return ('âŒ', '#f8d7da', '#721c24')
    elif status == 'pending':
        return ('â¬œ', '#e9ecef', '#6c757d')
    else:
        return ('âšª', '#fff', '#333')


def get_pipeline_summary(store: str) -> Optional[Dict]:
    """ê°€ìž¥ ìµœê·¼ íŒŒì´í”„ë¼ì¸ ìƒíƒœ ìš”ì•½ ë°˜í™˜"""
    available_dates = get_available_log_dates()

    if not available_dates:
        return None

    latest_date = available_dates[0]
    log_path = get_log_file_by_date(latest_date)
    status = parse_log_file(log_path, store)

    if not log_path or not log_path.exists():
        return None

    status['mode_detail'] = _compute_mode_detail(status, store)

    # ìš”ì•½ ìƒíƒœ
    if status['completed']:
        if len(status['errors']) > 0:
            status_text = f"ì™„ë£Œ (ì—ëŸ¬ {len(status['errors'])}ê±´)"
            status_icon = "âš ï¸"
        elif len(status['warnings']) > 0:
            status_text = f"ì™„ë£Œ (ê²½ê³  {len(status['warnings'])}ê±´)"
            status_icon = "âœ…"
        else:
            status_text = "ì •ìƒ ì™„ë£Œ"
            status_icon = "âœ…"
    else:
        status_text = "ì§„í–‰ ì¤‘"
        status_icon = "ðŸ”„"

    total_duration = calculate_duration(status['last_run_start'], status['last_run_end'])
    date_display = latest_date.replace('-', '/')

    return {
        'date': date_display,
        'status_icon': status_icon,
        'status_text': status_text,
        'duration': total_duration,
        'mode_detail': status['mode_detail'],
    }


def _calculate_total_duration_from_stages(status: Dict) -> str:
    """ë‹¨ê³„ë³„ ì†Œìš” ì‹œê°„ì˜ í•©ê³„ ê³„ì‚°"""
    stages = ['extract', 'preprocess', 'tuning', 'fitting', 'predict']
    total_seconds = 0

    for key in stages:
        stage = status.get(key, {})
        start_time = stage.get('start_time')
        end_time = stage.get('end_time')

        if start_time and end_time:
            try:
                start = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                end = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
                duration = end - start
                secs = int(duration.total_seconds())
                if secs > 0:
                    total_seconds += secs
            except Exception:
                pass

    if total_seconds == 0:
        return ""
    elif total_seconds < 60:
        return f"{total_seconds}ì´ˆ"
    elif total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}ë¶„ {seconds}ì´ˆ"
    else:
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        return f"{hours}ì‹œê°„ {minutes}ë¶„"


def render_pipeline_status(store: str, selected_date: str = None):
    """íŒŒì´í”„ë¼ì¸ ìƒíƒœ ë Œë”ë§"""
    st.markdown("### ðŸ“Š íŒŒì´í”„ë¼ì¸ ìƒíƒœ")

    available_dates = get_available_log_dates()

    if not available_dates:
        st.warning("íŒŒì´í”„ë¼ì¸ ì‹¤í–‰ ê¸°ë¡ì´ ì—†ìŠµë‹ˆë‹¤.")
        return

    available_date_objs = [datetime.strptime(d, '%Y-%m-%d').date() for d in available_dates]
    min_date = min(available_date_objs)
    max_date = max(available_date_objs)

    if selected_date is None or selected_date not in available_dates:
        default_date = max_date
    else:
        default_date = datetime.strptime(selected_date, '%Y-%m-%d').date()

    col_date, col_spacer = st.columns([1, 3])
    with col_date:
        selected_date_obj = st.date_input(
            "ðŸ“… íŒŒì´í”„ë¼ì¸ ë‚ ì§œ",
            value=default_date,
            min_value=min_date,
            max_value=max_date,
            key="pipeline_date_selector",
            label_visibility="collapsed",
        )
    selected_date = selected_date_obj.strftime('%Y-%m-%d')

    log_path = get_log_file_by_date(selected_date)
    status = parse_log_file(log_path, store)

    status['mode_detail'] = _compute_mode_detail(status, store)

    if not log_path or not log_path.exists():
        st.info(f"ðŸ“… {selected_date}: íŒŒì´í”„ë¼ì¸ ì‹¤í–‰ ê¸°ë¡ì´ ì—†ìŠµë‹ˆë‹¤.")
        return

    # ì´ ì†Œìš” ì‹œê°„
    total_duration = _calculate_total_duration_from_stages(status)

    error_count = len(status['errors'])
    warning_count = len(status['warnings'])

    summary_parts = []
    if total_duration:
        summary_parts.append(total_duration)
    if error_count > 0:
        summary_parts.append(f"ì—ëŸ¬ {error_count}ê±´")
    if warning_count > 0:
        summary_parts.append(f"ê²½ê³  {warning_count}ê±´")

    if summary_parts:
        st.markdown(f"**{' / '.join(summary_parts)}**")

    # í—¤ë”
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(f"**ì í¬**: {store}")
    with col2:
        if status['mode_detail']:
            st.markdown(f"**ì‹¤í–‰ ëª¨ë“œ**: `{status['mode_detail']}`")

    # ë‹¨ê³„ë³„ ìƒíƒœ ì¹´ë“œ
    stages = [
        ('ì¶”ì¶œ', 'extract', 'ðŸ“¥'),
        ('ì „ì²˜ë¦¬', 'preprocess', 'âš™ï¸'),
        ('íŠœë‹', 'tuning', 'ðŸŽ¯'),
        ('í•™ìŠµ', 'fitting', 'ðŸ“š'),
        ('ì˜ˆì¸¡', 'predict', 'ðŸ“Š'),
    ]

    cols = st.columns(len(stages))
    for i, (name, key, icon) in enumerate(stages):
        with cols[i]:
            stage = status[key]
            emoji, bg_color, text_color = get_status_style(stage['status'])
            duration = calculate_duration(stage['start_time'], stage['end_time'])

            st.markdown(f"""
                <div style='
                    background-color: {bg_color};
                    border-radius: 8px;
                    padding: 10px;
                    text-align: center;
                    min-height: 100px;
                '>
                    <div style='font-size: 24px;'>{emoji}</div>
                    <div style='font-weight: bold; color: {text_color}; margin: 5px 0;'>{icon} {name}</div>
                    <div style='font-size: 11px; color: #666;'>
                        {duration if duration else '-'}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    # ì‹œê°„ ìƒì„¸
    st.markdown("**ì‹¤í–‰ ì‹œê°„ ìƒì„¸**")

    time_data = []
    for name, key, icon in stages:
        stage = status[key]
        if stage['start_time'] or stage['end_time']:
            start = stage['start_time'].split(' ')[1] if stage['start_time'] else '-'
            end = stage['end_time'].split(' ')[1] if stage['end_time'] else '-'
            duration = calculate_duration(stage['start_time'], stage['end_time'])
            time_data.append({
                'ë‹¨ê³„': f"{icon} {name}",
                'ì‹œìž‘': start,
                'ì¢…ë£Œ': end,
                'ì†Œìš” ì‹œê°„': duration or '-'
            })

    if time_data:
        import pandas as pd
        st.dataframe(pd.DataFrame(time_data), hide_index=True)
    else:
        st.info("ì‹¤í–‰ ê¸°ë¡ì´ ì—†ìŠµë‹ˆë‹¤.")

    # ì¶”ê°€ ì‹¤í–‰ í‘œì‹œ
    additional_runs = status.get('additional_runs', [])
    if additional_runs:
        st.markdown("---")
        st.markdown("**ðŸ“Œ ê°™ì€ ë‚  ì¶”ê°€ ì‹¤í–‰**")
        for run in additional_runs:
            st.caption(f"â€¢ {run['time']} - {run['mode']} ìž‘ì—… ì‹¤í–‰ë¨")

    # ê²½ê³ /ì—ëŸ¬
    if status['errors']:
        unique_errors = list(dict.fromkeys(status['errors']))
        with st.expander(f"âŒ **ì—ëŸ¬ ({len(unique_errors)}ê±´)** - í´ë¦­í•˜ì—¬ íŽ¼ì¹˜ê¸°", expanded=False):
            error_html = "".join([
                f"<div style='background:#f8d7da; color:#721c24; padding:4px 8px; margin:2px 0; border-radius:4px; font-size:13px;'>{err}</div>"
                for err in unique_errors
            ])
            st.markdown(error_html, unsafe_allow_html=True)

    if status['warnings']:
        unique_warnings = list(dict.fromkeys(status['warnings']))
        with st.expander(f"âš ï¸ **ê²½ê³  ({len(unique_warnings)}ê±´)** - í´ë¦­í•˜ì—¬ íŽ¼ì¹˜ê¸°", expanded=False):
            warning_html = "".join([
                f"<div style='background:#fff3cd; color:#856404; padding:4px 8px; margin:2px 0; border-radius:4px; font-size:13px;'>{warn}</div>"
                for warn in unique_warnings
            ])
            st.markdown(warning_html, unsafe_allow_html=True)


# ê¸°ì¡´ í•¨ìˆ˜ ìœ ì§€ (í•˜ìœ„ í˜¸í™˜ì„±)
def render_pipeline_status_toggle(store: str, selected_date: str = None):
    """[Deprecated] render_pipeline_status() ì‚¬ìš© ê¶Œìž¥"""
    render_pipeline_status(store, selected_date)