from .base import BaseClient
from openai import OpenAI, AzureOpenAI
import requests
from typing import Any, Dict, List
import logging

CLIENT_REGISTRY = {}

def register_client(name):
    def decorator(cls):
        if isinstance(name, list):
            for n in name:
                CLIENT_REGISTRY[n.lower()] = cls
        else:
            CLIENT_REGISTRY[name.lower()] = cls
        return cls
    return decorator

@register_client("gpt-4o")
class AzureGPTClient(BaseClient):
    def __init__(self, endpoint, api_key, model_name="gpt-4o", **kwargs):
        self.model_name = model_name
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=kwargs.get('api_version', "2025-04-01-preview")
        )
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=kwargs.get('temperature', 0.7),
            timeout=kwargs.get('timeout', 90),
            max_tokens=kwargs.get('max_tokens', 1024*8)
        )
        if hasattr(response, 'usage') and response.usage:
            logging.info(f"[AzureGPTClient] Token usage: {response.usage}")
        return response.choices[0].message.content

@register_client(['deepseek-v3', 'deepseek-r1'])
class RequestClient(BaseClient):
    def __init__(self, endpoint: str, api_key: str, model_name: str, **kwargs):
        self.model_name = model_name
        self.base_url = endpoint
        self.headers = {
            "Content-Type": "application/json",
            'Authorization': 'Bearer '+ api_key
        }
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get('temperature', 0.7),
            "stream": "false"
        }
        response = requests.post(self.base_url, headers=self.headers, json=payload)
        response.raise_for_status()
        if hasattr(response, 'usage') and response.usage:
            logging.info(f"[RequestClient] Token usage: {response.usage}")
        return response.json()['choices'][0]['message']['content']

class OpenaiClient(BaseClient):
    def __init__(self, endpoint, api_key, model_name, **kwargs):
        self.model_name = model_name
        self.client = OpenAI(base_url=endpoint, api_key=api_key)
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        response = self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            temperature=kwargs.get('temperature', 0.7),
            timeout=kwargs.get('timeout', 90),
            max_tokens=kwargs.get('max_tokens', 1024*8)
        )
        if hasattr(response, 'usage') and response.usage:
            logging.info(f"[OpenaiClient] Token usage: {response.usage}")
        return response.choices[0].message.content

# How to register a custom client:
# @register_client("your-model-name")
# class YourCustomClient(BaseClient):
#     ...

def get_client(endpoint: str, api_key: str, model_name: str, **kwargs) -> BaseClient:
    client_class = CLIENT_REGISTRY.get(model_name.lower(), OpenaiClient)
    return client_class(endpoint, api_key, model_name, **kwargs) 