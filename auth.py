"""
Authentication Module
=====================
streamlit-authenticator 기반 사용자 인증 관리
"""

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from pathlib import Path
from typing import Optional, List, Dict, Any


AUTH_CONFIG_PATH = Path(__file__).parent / "auth_config.yaml"


def load_auth_config() -> Dict[str, Any]:
    """인증 설정 로드"""
    with open(AUTH_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_auth_config(config: Dict[str, Any]) -> None:
    """인증 설정 저장"""
    with open(AUTH_CONFIG_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def get_authenticator() -> stauth.Authenticate:
    """Authenticator 인스턴스 반환"""
    config = load_auth_config()

    authenticator = stauth.Authenticate(
        credentials=config['credentials'],
        cookie_name=config['cookie']['name'],
        cookie_key=config['cookie']['key'],
        cookie_expiry_days=config['cookie']['expiry_days'],
    )

    return authenticator


def get_user_role(username: str) -> str:
    """사용자 역할 반환 (admin / store)"""
    config = load_auth_config()
    users = config['credentials']['usernames']

    if username in users:
        return users[username].get('role', 'store')
    return 'store'


def get_user_stores(username: str) -> List[str]:
    """사용자가 접근 가능한 점포 목록 반환"""
    config = load_auth_config()
    users = config['credentials']['usernames']

    if username in users:
        return users[username].get('stores', [])
    return []


def is_admin(username: str) -> bool:
    """관리자 여부 확인"""
    return get_user_role(username) == 'admin'


def check_page_access(username: str, page: str) -> bool:
    """
    페이지 접근 권한 확인

    Parameters
    ----------
    username : str
        사용자명
    page : str
        페이지명 ('order' / 'admin')

    Returns
    -------
    bool
        접근 가능 여부
    """
    role = get_user_role(username)

    if page == 'admin':
        return role == 'admin'
    elif page == 'order':
        return True  # 모든 로그인 사용자 접근 가능

    return False


def generate_password_hash(password: str) -> str:
    """
    비밀번호 해시 생성

    Usage:
        python -c "from auth import generate_password_hash; print(generate_password_hash('your_password'))"
    """
    from streamlit_authenticator.utilities.hasher import Hasher
    return Hasher.hash(password)


def init_session_state():
    """세션 상태 초기화"""
    if 'authentication_status' not in st.session_state:
        st.session_state['authentication_status'] = None
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    if 'name' not in st.session_state:
        st.session_state['name'] = None


def render_login() -> tuple[Optional[str], Optional[bool], Optional[str]]:
    """
    로그인 폼 렌더링

    Returns
    -------
    tuple
        (name, authentication_status, username)
    """
    init_session_state()
    authenticator = get_authenticator()

    name, authentication_status, username = authenticator.login(
        location='main',
        fields={
            'Form name': 'FRUITY 로그인',
            'Username': '사용자명',
            'Password': '비밀번호',
            'Login': '로그인'
        }
    )

    return name, authentication_status, username


def render_logout():
    """로그아웃 버튼 렌더링"""
    authenticator = get_authenticator()
    authenticator.logout('로그아웃', 'sidebar')


if __name__ == "__main__":
    # 비밀번호 해시 생성 테스트
    passwords = ['admin123', 'fruit210', 'fruit220', 'fruit480']
    for pwd in passwords:
        print(f"{pwd}: {generate_password_hash(pwd)}")