"""
Gemini LLM Client
==================
Google Gemini API 클라이언트
"""

import os
import ssl
import streamlit as st
from typing import Optional
from .llm_base import LLMClient

# 내부망 SSL 프록시 우회
os.environ['GRPC_SSL_CIPHER_SUITES'] = 'HIGH+ECDSA'
ssl._create_default_https_context = ssl._create_unverified_context

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[LLM] google-genai not installed. Run: pip install google-genai")


class GeminiClient(LLMClient):
    """Gemini API 클라이언트"""

    def __init__(self):
        self.client = None
        self.model = "gemini-2.5-flash"
        self._initialize()

    def _initialize(self):
        """클라이언트 초기화"""
        if not GEMINI_AVAILABLE:
            return

        try:
            api_key = None
            if "gemini" in st.secrets and "gemini_api_key" in st.secrets["gemini"]:
                api_key = st.secrets["gemini"]["gemini_api_key"]

            if api_key:
                self.client = genai.Client(api_key=api_key)
                print("[LLM] Gemini Client initialized OK")
            else:
                print("[LLM] Gemini API key not found in secrets.toml [gemini] section")
        except Exception as e:
            print(f"[LLM] Gemini init failed: {e}")
            self.client = None

    def is_available(self) -> bool:
        """클라이언트 사용 가능 여부"""
        return self.client is not None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """텍스트 생성"""
        if not self.client:
            return "[Gemini API 연결 실패]"

        try:
            # Gemini는 system prompt를 prompt 앞에 붙임
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt
            )
            return response.text

        except Exception as e:
            print(f"[LLM] Gemini generate error: {e}")
            return f"[Gemini API 에러: {e}]"