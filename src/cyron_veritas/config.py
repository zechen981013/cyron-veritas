"""路径、环境变量和大模型配置。"""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

CACHE_DIR = ROOT / "data" / "cache"
OUTPUT_DIR = ROOT / "output"
CONFIG_DIR = ROOT / "config"
PORTFOLIO_DIR = ROOT / "portfolio"

BASE_CCY = "CNY"


def chat_model(temperature: float = 0.2):
    """DeepSeek（OpenAI 兼容接口）。换 vLLM 或其他模型只需改 .env 里的 LLM_BASE_URL / LLM_MODEL。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("LLM_API_KEY"),
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        temperature=temperature,
        timeout=float(os.getenv("LLM_TIMEOUT", "120")),
        max_retries=2,
    )
