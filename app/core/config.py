
# app/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional, Dict
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LLM Backend API"
    
    # OpenAI settings
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", "")
    
    # MongoDB settings
    MONGODB_URL: str = os.getenv("MONGODB_URL", "")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "")
    
    # LLM settings
    DEFAULT_LLM_PROVIDER: str = os.getenv("DEFAULT_LLM_PROVIDER", "openai")
    LLM_API_KEYS: Dict[str, str] = {
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
        # Add other LLM providers as needed
    }
    
    # Local LLM settings
    LOCAL_LLM_PATH: Optional[str] = os.getenv("LOCAL_LLM_PATH", None)
    LOCAL_LLM_TYPE: str = os.getenv("LOCAL_LLM_TYPE", "llama")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", "app.log")

     # Azure OpenAI settings
    AZURE_OPENAI_API_KEY: Optional[str] = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_ENDPOINT: Optional[str] = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_MODEL_NAME: str = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-40-mini")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2023-03-15-preview")
    
    # OpenAI settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    # OPENAI_API_BASE: Optional[str] = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    # OPENAI_API_TYPE: str = os.getenv("OPENAI_API_TYPE", "openai")
    # OPENAI_API_VERSION: str = os.getenv("OPENAI_API_VERSION", "2023-03-15-preview")
    OPENAI_API_MODEL: str = os.getenv("OPENAI_API_MODEL", "gpt-40-mini")
    
    #Firecrawl settings
    FIRECRAWL_API_KEY: Optional[str] = os.getenv("FIRECRAWL_API_KEY","")

    # Update LLM API keys with Azure OpenAI
    LLM_API_KEYS: Dict[str, str] = {
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
        "azure_openai": os.getenv("AZURE_OPENAI_API_KEY", ""),
    }
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()