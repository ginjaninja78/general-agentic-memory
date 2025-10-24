from dataclasses import dataclass
from typing import Any

@dataclass
class OpenAIGeneratorConfig:
    """OpenAI生成器配置"""
    model_name: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 300
    thread_count: int | None = None
    system_prompt: str | None = None
    timeout: float = 60.0

@dataclass
class VLLMGeneratorConfig:
    """VLLM生成器配置"""
    generate_model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 300
    stop: list[str] | None = None
    repetition_penalty: float = 1.1
    lora_path: str | None = None
    n: int = 1
    system_prompt: str | None = None
