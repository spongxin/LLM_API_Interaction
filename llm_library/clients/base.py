from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseClient(ABC):
    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """发送消息到LLM并返回字符串回复。"""
        pass 