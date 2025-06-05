from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Type
from enum import Enum

class GlobalResponse(BaseModel):
    status: int
    message: Optional[str] = str
    data: Optional[dict] = None
    errors: Optional[dict] = None
    
class SupportedChannel(str, Enum):
    LINKEDIN = "LinkedIn"
    TWITTER = "Twitter"

class LlmProviderEnum(str, Enum):
    openAI = "OpenAI"
    gemini = "Gemini"
    anthropic = "Anthropic"

class GetVariantRequest(BaseModel):
    content: str
    active_current_channel: SupportedChannel
    destination_channel: SupportedChannel
    count: int = Field(..., ge=1, le=5, description="Number of variants (between 1 and 5)")
    instructions: Optional[str] = None
    target_language: Optional[str] = None


class GetVariantResponse(BaseModel):
    variants: List[str]

class CheckGrammarRequest(BaseModel):
    content: str

class LLMRequest(BaseModel):
    messages: list
    provider: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    additional_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    SchemaFormat: Optional[Type[BaseModel]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain quantum computing in simple terms",
                "provider": LlmProviderEnum.openAI,
                "model": "gpt-4",
                "max_tokens": 500,
                "temperature": 0.7
            }
        }

class LLMResponse(BaseModel):
    text : Any
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class ErrorResponse(BaseModel):
    detail: str

class GrammarCheckResponse(BaseModel):
    status_code: int
    corrected_text: str

class GrammarCheckRequest(BaseModel):
    content: str

class SummaryRequest(BaseModel):
    content: str

class SummaryResponse(BaseModel):
    summary: str

class ReduceRequest(BaseModel):
    content: str

class ReduceResponse(BaseModel):
    output: str

class ExpandRequest(BaseModel):
    content: str

class ExpandResponse(BaseModel):
    output: str

class PDFSummaryRequest(BaseModel):
    file_name: str
    current_channel: SupportedChannel
    count: int = Field(..., ge=1, le=5, description="Number of variants (between 1 and 5)")


class PDFSummaryResponse(BaseModel):
    status_code: int
    status_msg: Optional[str]
    data: dict

class ToneRequest(BaseModel):
    content: str
    tone: str

class ToneResponse(BaseModel):
    output: str

class StyleRequest(BaseModel):
    content: str
    style: str

class StyleResponse(BaseModel):
    output: str
    
class CommentRequest(BaseModel):
    content: str
    channel: str
    instruction: str
    department: str
    title: str

class CommentResponse(BaseModel):
    output: str

class ThoughtRequest(BaseModel):
    content: str
    channel: str
    instruction: str
    department: str
    title: str
    
class ThoughtResponse(BaseModel):
    output: str

class APIKeyCheckRequest(BaseModel):
    provideName: str
    provideKey: str
    
class APIKeyCheckResponse(BaseModel):
    status: int
    message: str
    provideName: str

class TranslateRequest(BaseModel):
    content: str
    destinationt_language: str

class TranslateResponse(BaseModel):
    output: str

class FirecrawlRequest(BaseModel):
     url: str