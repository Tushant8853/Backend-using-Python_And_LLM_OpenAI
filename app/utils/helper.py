# app/utils/helpers.py
import uuid
from datetime import datetime
import json
from typing import Any, Dict, List, Optional
from ..core.logger import logger
from ..models.schemas import SupportedChannel


def generate_uuid():
    """Generate a unique UUID"""
    return str(uuid.uuid4())

def now_timestamp():
    """Get current timestamp in ISO format"""
    return datetime.utcnow().isoformat()

def safe_serialize(obj: Any) -> Dict[str, Any]:
    """Safely serialize an object to JSON-compatible dict"""
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize object: {e}")
        return {"error": "Unable to serialize object"}

def validate_environment():
    """Validate that all required environment variables are set"""
    from ..core.config import settings
    
    missing_vars = []
    
    # Check MongoDB settings
    if not settings.MONGODB_URL:
        missing_vars.append("MONGODB_URL")
    
    # Check LLM settings
    if not settings.DEFAULT_LLM_PROVIDER:
        missing_vars.append("DEFAULT_LLM_PROVIDER")
    
    if settings.DEFAULT_LLM_PROVIDER == "openai" and not settings.LLM_API_KEYS.get("openai"):
        missing_vars.append("OPENAI_API_KEY")
    
    if settings.DEFAULT_LLM_PROVIDER == "anthropic" and not settings.LLM_API_KEYS.get("anthropic"):
        missing_vars.append("ANTHROPIC_API_KEY")
    
    if settings.DEFAULT_LLM_PROVIDER == "local" and not settings.LOCAL_LLM_PATH:
        missing_vars.append("LOCAL_LLM_PATH")
    
    return missing_vars

def get_channel_character_limit(channel: SupportedChannel):
    if channel == SupportedChannel.LINKEDIN:
        return 2000
    elif channel == SupportedChannel.TWITTER:
        return 220
    return None