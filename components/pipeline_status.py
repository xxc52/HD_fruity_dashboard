"""
Pipeline Status Component
=========================
scheduler 로그를 파싱하여 파이프라인 상태 표시
"""

import streamlit as st
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List


# logs 폴더 경로
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


def get_log_file_by_date(date_str: str) -> Optional[Path]:
    """
    특정 날짜의 scheduler 로그 파일 반환

    Parameters
    ----------
    date_str : str
        날짜 (YYYY-MM-DD 또는 YYYYMMDD)

    Returns
    -------
    Optional[Path]
        로그 파일 경로
    """
    if not LOGS_DIR.exists():
        return None

    # YYYY-MM-DD → YYYYMMDD 변환
    date_key = date_str.replace('-', '')

    # scheduler_YYYYMMDD.log 형식 (scheduler_runner_*.log 제외)
    log_path = LOGS_DIR / f"scheduler_{date_key}.log"

    if log_path.exists():
        return log_path

    return None


def parse_log_file(log_path: Path) -> Dict:
    """
    로그 파일 파싱하여 파이프라인 상태 추출
    (같은 날 여러 번 실행 시 마지막 실행 기준)

    Returns
    -------
    Dict
        {
            'date': str,
            'mode': str,
            'extract': {'time': str, 'status': str},
            'preprocess': {'time': str, 'status': str},
            'tuning': {'time': str, 'status': str},
            'fitting': {'time': str, 'status': str},
            'predict': {'time': str, 'status': str},
            'completed': bool,
            'warnings': list,  # 경고 메시지 리스트
            'errors': list,    # 에러 메시지 리스트
            'last_run_start': str,
        }
    """
    result = {
        'date': None,
        'mode': None,
        'extract': {'time': None, 'status': 'pending'},
        'preprocess': {'time': None, 'status': 'pending'},
        'tuning': {'time': None, 'status': 'pending'},
        'fitting': {'time': None, 'status': 'pending'},
        'predict': {'time': None, 'status': 'pending'},
        'completed': False,
        'warnings': [],
        'errors': [],
        'last_run_start': None,
    }

    if not log_path or not log_path.exists():
        return result

    try:
        content = log_path.read_text(encoding='utf-8')
        lines = content.split('\n')

        # 마지막 "[scheduler] 파이프라인 시작" 위치 찾기 (mode= 포함된 메인 시작)
        last_start_idx = -1
        for i, line in enumerate(lines):
            if '[scheduler]' in line and '파이프라인 시작' in line and 'mode=' in line:
                last_start_idx = i

        # 마지막 실행 구간만 파싱
        if last_start_idx >= 0:
            lines = lines[last_start_idx:]

        for line in lines:
            # 시간 추출 패턴 (YYYY-MM-DD HH:MM:SS)
            time_match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
            timestamp = time_match.group(1) if time_match else None

            # 파이프라인 시작 (scheduler 메인)
            if '[scheduler]' in line and '파이프라인 시작' in line and 'mode=' in line:
                mode_match = re.search(r'mode=(\w+)', line)
                if mode_match:
                    result['mode'] = mode_match.group(1)
                if timestamp:
                    result['date'] = timestamp.split(' ')[0]
                    result['last_run_start'] = timestamp

            # 추출
            if '[scheduler]' in line and '[추출] 완료' in line:
                result['extract'] = {'time': timestamp, 'status': 'success'}
            elif '[추출] 실패' in line or ('[추출]' in line and '[ERROR]' in line):
                result['extract'] = {'time': timestamp, 'status': 'error'}

            # 전처리
            if '[scheduler]' in line and '[전처리] 완료' in line:
                result['preprocess'] = {'time': timestamp, 'status': 'success'}
            elif '[전처리] 실패' in line or ('[전처리]' in line and '[ERROR]' in line):
                result['preprocess'] = {'time': timestamp, 'status': 'error'}

            # 튜닝
            if '[scheduler]' in line and '[튜닝] 완료' in line:
                result['tuning'] = {'time': timestamp, 'status': 'success'}
            elif '[튜닝] 에러' in line or ('[튜닝]' in line and '[ERROR]' in line):
                result['tuning'] = {'time': timestamp, 'status': 'error'}

            # 학습
            if '[scheduler]' in line and '[학습] 완료' in line:
                result['fitting'] = {'time': timestamp, 'status': 'success'}
            elif '[학습] 에러' in line or ('[학습]' in line and '[ERROR]' in line):
                result['fitting'] = {'time': timestamp, 'status': 'error'}

            # 예측
            if '[scheduler]' in line and '[예측] 완료' in line:
                result['predict'] = {'time': timestamp, 'status': 'success'}
            elif '[예측] 에러' in line or ('[예측]' in line and '[ERROR]' in line):
                result['predict'] = {'time': timestamp, 'status': 'error'}

            # 파이프라인 완료
            if '[scheduler]' in line and '파이프라인 완료' in line:
                result['completed'] = True

            # 경고/에러 메시지 수집
            if '[WARNING]' in line:
                # 메시지 추출 (시간 제외)
                msg_match = re.search(r'\[WARNING\]\s*(?:\[[^\]]+\])?\s*(.+)$', line)
                if msg_match:
                    msg = msg_match.group(1).strip()
                    # 중복 방지 및 너무 긴 메시지 제한
                    if msg and len(msg) < 200:
                        result['warnings'].append(msg)
            if '[ERROR]' in line:
                msg_match = re.search(r'\[ERROR\]\s*(?:\[[^\]]+\])?\s*(.+)$', line)
                if msg_match:
                    msg = msg_match.group(1).strip()
                    if msg and len(msg) < 200:
                        result['errors'].append(msg)

    except Exception as e:
        print(f"[pipeline_status] Log parsing error: {e}")

    return result


def get_status_emoji(status: str) -> str:
    """상태에 따른 이모지 반환"""
    if status == 'success':
        return '✅'
    elif status == 'error':
        return '❌'
    elif status == 'pending':
        return '⏳'
    else:
        return '⚪'


def render_pipeline_status(date_t: str = None):
    """
    파이프라인 상태를 오른쪽 위에 표시

    Parameters
    ----------
    date_t : str
        기준 날짜 (YYYY-MM-DD), None이면 오늘
    """
    if date_t is None:
        date_t = datetime.now().strftime('%Y-%m-%d')

    log_path = get_log_file_by_date(date_t)
    status = parse_log_file(log_path)

    # 상태 요약
    if not log_path:
        overall = f'❓ {date_t} 로그 없음'
    elif status['completed']:
        if len(status['errors']) > 0:
            overall = '⚠️ 완료 (에러 있음)'
        elif len(status['warnings']) > 0:
            overall = '✅ 완료 (경고 있음)'
        else:
            overall = '✅ 정상 완료'
    else:
        overall = '🔄 진행 중'

    # Expander로 표시
    with st.expander(f"📊 파이프라인 상태: {overall}", expanded=False):
        # 1. 단계별 상태 (날짜 포함)
        st.markdown("**단계별 진행 상태**")
        stages = [
            ('데이터 추출', 'extract'),
            ('전처리', 'preprocess'),
            ('튜닝', 'tuning'),
            ('학습', 'fitting'),
            ('예측', 'predict'),
        ]

        for name, key in stages:
            stage = status[key]
            emoji = get_status_emoji(stage['status'])
            time_str = stage['time'] if stage['time'] else '-'
            st.markdown(f"{emoji} **{name}**: {time_str}")

        st.markdown("---")

        # 2. 요약 정보
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**기준일**: {date_t}")
            st.markdown(f"**실행 모드**: {status['mode'] or 'N/A'}")

        with col2:
            if status['last_run_start']:
                st.markdown(f"**마지막 실행**: {status['last_run_start']}")
            st.markdown(f"**경고**: {len(status['warnings'])}건 / **에러**: {len(status['errors'])}건")

        # 3. 특수 상황 안내 (전일 휴일로 인한 데이터 미추출 등)
        no_data_warning = any('추출된 데이터가 없습니다' in w for w in status['warnings'])
        if no_data_warning:
            st.markdown("---")
            st.info("ℹ️ **전일 데이터 미추출**: 전일이 휴일/휴무일이어서 추출된 판매 데이터가 없습니다. 정상적인 상황입니다.")

        # 4. 경고/에러 상세 (있을 경우)
        if status['warnings'] or status['errors']:
            st.markdown("---")

            if status['errors']:
                st.markdown("**에러 내용:**")
                for i, err in enumerate(status['errors'][:5], 1):  # 최대 5개
                    st.markdown(f"  {i}. ❌ {err}")
                if len(status['errors']) > 5:
                    st.markdown(f"  ... 외 {len(status['errors']) - 5}건")

            if status['warnings']:
                st.markdown("**경고 내용:**")
                # 중복 제거 및 요약
                unique_warnings = []
                seen = set()
                for w in status['warnings']:
                    # 핵심 내용만 추출 (예: "20260101 예보 데이터 없음" → "예보 데이터 없음")
                    key = w.split(':')[0] if ':' in w else w[:50]
                    if key not in seen:
                        seen.add(key)
                        unique_warnings.append(w)

                for i, warn in enumerate(unique_warnings[:5], 1):  # 최대 5개
                    st.markdown(f"  {i}. ⚠️ {warn}")
                if len(unique_warnings) > 5:
                    st.markdown(f"  ... 외 {len(unique_warnings) - 5}건")