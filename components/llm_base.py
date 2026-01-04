"""
LLM Client Abstraction Layer
=============================
Claude / Gemini 전환 가능한 LLM 클라이언트 추상화
"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """LLM 클라이언트 베이스 클래스"""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        텍스트 생성

        Parameters
        ----------
        prompt : str
            사용자 프롬프트
        system_prompt : Optional[str]
            시스템 프롬프트 (역할 지정)

        Returns
        -------
        str
            생성된 텍스트
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """클라이언트 사용 가능 여부"""
        pass