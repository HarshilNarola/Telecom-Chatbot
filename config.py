"""Application configuration and project paths."""
from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BASE_DIR / "chroma_store"
DATA_DIR = BASE_DIR / "data"

load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    groq_model: str
    hf_token: str | None


def get_settings() -> Settings:
    groq_api_key = _get_secret("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. Add it to .env locally or "
            "Streamlit Secrets when deployed."
        )

    hf_token = _get_secret("HF_TOKEN") or None
    groq_model = _get_secret("GROQ_MODEL") or "qwen/qwen3.8-27b"
    if groq_model == "qwen/qwen3-32b":
        groq_model = "qwen/qwen3.8-27b"
    return Settings(
        groq_api_key=groq_api_key,
        groq_model=groq_model,
        hf_token=hf_token,
    )


def _get_secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value

    try:
        from streamlit import secrets
    except ImportError:
        return ""

    try:
        return str(secrets.get(name, "")).strip()
    except FileNotFoundError:
        return ""
