"""
Claude LLM Client
==================
Anthropic Claude API 클라이언트
"""

import streamlit as st
from typing import Optional
from .llm_base import LLMClient

try:
    import anthropic
    import httpx
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("[LLM] anthropic not installed. Run: pip install anthropic")


class ClaudeClient(LLMClient):
    """Claude API 클라이언트"""

    def __init__(self):
        self.client = None
        self.model = "claude-haiku-4-5-20251001"
        self._initialize()

    def _initialize(self):
        """클라이언트 초기화"""
        if not ANTHROPIC_AVAILABLE:
            return

        try:
            api_key = None
            if "claude" in st.secrets and "claude_api_key" in st.secrets["claude"]:
                api_key = st.secrets["claude"]["claude_api_key"]

            if api_key:
                # 내부망 SSL 프록시 우회
                http_client = httpx.Client(verify=False)
                self.client = anthropic.Anthropic(api_key=api_key, http_client=http_client)
                print("[LLM] Claude Client initialized OK")
            else:
                print("[LLM] Claude API key not found in secrets.toml [claude] section")
        except Exception as e:
            print(f"[LLM] Claude init failed: {e}")
            self.client = None

    def is_available(self) -> bool:
        """클라이언트 사용 가능 여부"""
        return self.client is not None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """텍스트 생성"""
        if not self.client:
            return "[Claude API 연결 실패]"

        try:
            messages = [{"role": "user", "content": prompt}]

            kwargs = {
                "model": self.model,
                "max_tokens": 2048,
                "messages": messages,
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            response = self.client.messages.create(**kwargs)
            return response.content[0].text

        except Exception as e:
            print(f"[LLM] Claude generate error: {e}")
            return f"[Claude API 에러: {e}]"