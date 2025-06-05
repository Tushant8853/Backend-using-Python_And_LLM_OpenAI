from fastapi import APIRouter, HTTPException, Header
from ...models.schemas import (
    ErrorResponse,
    GrammarCheckResponse,
    GrammarCheckRequest,
    SummaryRequest,
    SummaryResponse,
    ReduceRequest,
    ReduceResponse,
    ExpandRequest,
    ExpandResponse,
    ToneRequest,
    ToneResponse,
    StyleRequest,
    StyleResponse,
    CommentRequest,
    CommentResponse,
    ThoughtRequest,
    ThoughtResponse,
    GlobalResponse, 
    LlmProviderEnum,
    TranslateResponse,
    TranslateRequest
)
from pydantic import BaseModel
from ...services.annotation_service import Annotation
from ...core.logger import logger
from fastapi.responses import JSONResponse
import os
import traceback
from ...services.llm_service import LLMService

router = APIRouter()

async def set_openai_key(providerKey: str):
    if not providerKey:
        return JSONResponse(
            status_code=400,
            content={
                "status": 400,
                "message": "Missing OpenAI API key in header",
                "data": {}
            }
        )
    os.environ["OPENAI_API_KEY"] = providerKey

async def set_gemini_key(providerKey: str):
    if not providerKey:
        return JSONResponse(
            status_code=400,
            content={
                "status": 400,
                "message": "Missing OpenAI API key in header",
                "data": {}
            }
        )
    os.environ["GEMINI_API_KEY"] = providerKey

async def set_provider(provider_name: str, provider_key: str):
    if provider_name == LlmProviderEnum.openAI:
        await set_openai_key(provider_key)
    elif provider_name == LlmProviderEnum.gemini:
        await set_gemini_key(provider_key)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider_name}")

@router.post( "/checkGrammar", response_model=GrammarCheckResponse,responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def check_grammar(request: GrammarCheckRequest ,providerName: str = Header(...),providerKey: str = Header(...)):
    """
    Check grammar using the Annotation service.
    """
    class SchemaFormat(BaseModel):
        corrected_text: str

    try:
        await set_provider(providerName, providerKey)
    except Exception as e:
        logger.error(f"Error setting provider: {str(e)}")
        res = GlobalResponse(status=400, message=f"Unsupported provider: {str(e)}")
        return JSONResponse(
            status_code=401,
            content=res.model_dump()
        )
    try:
        llm_service = LLMService()
        annotation_service = Annotation(llm_service)
        corrected_text, status_code = await annotation_service.check_grammar(
            request,
            provider=providerName,
            SchemaFormat=SchemaFormat
            )
        return JSONResponse(
            content={"status_code": status_code, "corrected_text": corrected_text},
            status_code=status_code,
        )
    except ValueError as e:
        if "Authentication failed" in str(e):
            res = GlobalResponse(status=401, message=str(e))
            return JSONResponse(status_code=401, content=res.model_dump())
        else:
            res = GlobalResponse(status=400, message=str(e))
            return JSONResponse(status_code=400, content=res.model_dump())
    except Exception as e:
        logger.error(f"Error checking grammar: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/summary", response_model=SummaryResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def generate_summary(request: SummaryRequest,providerName: str = Header(...),providerKey: str = Header(...)):
    """
    Generate a summary of the input text using the Annotation service.
    """
    class SchemaFormat(BaseModel):
        summary: str
    try:
        await set_provider(providerName, providerKey)
    except Exception as e:
        logger.error(f"Error setting provider: {str(e)}")
        res = GlobalResponse(status=400, message=f"Unsupported provider: {str(e)}")
        return JSONResponse(
            status_code=401,
            content=res.model_dump()
        )
    try:
        logger.info(f"Received summary request: {request.content}")
        llm_service = LLMService()
        annotation_service = Annotation(llm_service)
        summary_response = await annotation_service.generate_summary(
            request.content,
            provider=providerName,
            SchemaFormat=SchemaFormat
            )
        return SummaryResponse(summary=summary_response)
    
    except ValueError as e:
        if "Authentication failed" in str(e):
            res = GlobalResponse(status=401, message=str(e))
            return JSONResponse(status_code=401, content=res.model_dump())
        else:
            res = GlobalResponse(status=400, message=str(e))
            return JSONResponse(status_code=400, content=res.model_dump())

    
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
@router.post("/reduce", response_model=ReduceResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def reduce_text(request: ReduceRequest,providerName: str = Header(...),providerKey: str = Header(...)):
    """
    Reduce the given text using the Annotation service.
    """
    class SchemaFormat(BaseModel):
        output: str
    try:
        await set_provider(providerName, providerKey)
    except Exception as e:
        logger.error(f"Error setting provider: {str(e)}")
        res = GlobalResponse(status=400, message=f"Unsupported provider: {str(e)}")
        return JSONResponse(
            status_code=401,
            content=res.model_dump()
        )
    try:
        logger.info(f"Received text reduction request: {request.content}")
        llm_service = LLMService()
        annotation_service = Annotation(llm_service)
        response = await annotation_service.reduce_text(
            request.content,
            provider=providerName,
            SchemaFormat=SchemaFormat
            )

        return ReduceResponse(output=response)

    except ValueError as e:
        if "Authentication failed" in str(e):
            res = GlobalResponse(status=401, message=str(e))
            return JSONResponse(status_code=401, content=res.model_dump())
        else:
            res = GlobalResponse(status=400, message=str(e))
            return JSONResponse(status_code=400, content=res.model_dump())
    except Exception as e:
        logger.error(f"Error reducing text: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/expand", response_model=ExpandResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def expand_text(request: ExpandRequest,providerName: str = Header(...),providerKey: str = Header(...)):
    """
    Expand the given text using the Annotation service.
    """
    class SchemaFormat(BaseModel):
        output: str
    try:
        await set_provider(providerName, providerKey)
    except Exception as e:
        logger.error(f"Error setting provider: {str(e)}")
        res = GlobalResponse(status=400, message=f"Unsupported provider: {str(e)}")
        return JSONResponse(
            status_code=401,
            content=res.model_dump()
        )
    try:
        logger.info(f"Received text expansion request: {request.content}")
        llm_service = LLMService()
        annotation_service = Annotation(llm_service)
        response = await annotation_service.expand_text(
            request.content,
            provider=providerName,
            SchemaFormat=SchemaFormat
            )

        return ExpandResponse(output=response)
    except ValueError as e:
            if "Authentication failed" in str(e):
                res = GlobalResponse(status=401, message=str(e))
                return JSONResponse(status_code=401, content=res.model_dump())
            else:
                res = GlobalResponse(status=400, message=str(e))
                return JSONResponse(status_code=400, content=res.model_dump())
    except Exception as e:
        logger.error(f"Error expanding text: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/tone", response_model=ToneResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def adjust_tone(request: ToneRequest,providerName: str = Header(...),providerKey: str = Header(...)):
    
    class SchemaFormat(BaseModel):
        output: str
    try:
        await set_provider(providerName, providerKey)
    except Exception as e:
        logger.error(f"Error setting provider: {str(e)}")
        res = GlobalResponse(status=400, message=f"Unsupported provider: {str(e)}")
        return JSONResponse(
            status_code=401,
            content=res.model_dump()
        )
    try:
        logger.info(f"Received tone adjustment request: {request.content}")
        llm_service = LLMService()
        annotation_service = Annotation(llm_service)
        response = await annotation_service.adjust_tone(
            request.content,
            request.tone,
            provider=providerName,
            SchemaFormat=SchemaFormat
            )
        return ToneResponse(output=response)
    except ValueError as e:
        if "Authentication failed" in str(e):
            res = GlobalResponse(status=401, message=str(e))
            return JSONResponse(status_code=401, content=res.model_dump())
        else:
            res = GlobalResponse(status=400, message=str(e))
            return JSONResponse(status_code=400, content=res.model_dump())
    except Exception as e:
        logger.error(f"Error adjusting tone: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
@router.post("/style", response_model=StyleResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def adjust_style(request: StyleRequest,providerName: str = Header(...),providerKey: str = Header(...)):
    class SchemaFormat(BaseModel):
        output: str
    try:
        await set_provider(providerName, providerKey)
    except Exception as e:
        logger.error(f"Error setting provider: {str(e)}")
        res = GlobalResponse(status=400, message=f"Unsupported provider: {str(e)}")
        return JSONResponse(
            status_code=401,
            content=res.model_dump()
        )
    try:
        logger.info(f"Received style adjustment request: {request.content}")
        llm_service = LLMService()
        annotation_service = Annotation(llm_service)
        response = await annotation_service.adjust_style(
            request.content,
            request.style,
            provider=providerName,
            SchemaFormat=SchemaFormat
            )
        return StyleResponse(output=response)
    except ValueError as e:
        if "Authentication failed" in str(e):
            res = GlobalResponse(status=401, message=str(e))
            return JSONResponse(status_code=401, content=res.model_dump())
        else:
            res = GlobalResponse(status=400, message=str(e))
            return JSONResponse(status_code=400, content=res.model_dump())
    except Exception as e:
        logger.error(f"Error adjusting style: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/getComment", response_model=CommentResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def get_comment(request: CommentRequest,providerName: str = Header(...),providerKey: str = Header(...)):
    class SchemaFormat(BaseModel):
        output: str
    try:
        await set_provider(providerName, providerKey)
    except Exception as e:
        logger.error(f"Error setting provider: {str(e)}")
        res = GlobalResponse(status=400, message=f"Unsupported provider: {str(e)}")
        return JSONResponse(
            status_code=401,
            content=res.model_dump()
        )
    try:
        logger.info(f"Received comment generation request: {request.content} for {request.channel}")
        llm_service = LLMService()
        annotation_service = Annotation(llm_service)
        comment = await annotation_service.generate_comment(
            request,
            provider=providerName,
            SchemaFormat=SchemaFormat
            )
        return CommentResponse(output=comment)
    except ValueError as e:
        if "Authentication failed" in str(e):
            res = GlobalResponse(status=401, message=str(e))
            return JSONResponse(status_code=401, content=res.model_dump())
        else:
            res = GlobalResponse(status=400, message=str(e))
            return JSONResponse(status_code=400, content=res.model_dump())
    except Exception as e:
        logger.error(f"Error generating comment: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/getThought", response_model=ThoughtResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def get_thought(request: ThoughtRequest,providerName: str = Header(...),providerKey: str = Header(...)):
    class SchemaFormat(BaseModel):
        output: str
    try:
        await set_provider(providerName, providerKey)
    except Exception as e:
        logger.error(f"Error setting provider: {str(e)}")
        res = GlobalResponse(status=400, message=f"Unsupported provider: {str(e)}")
        return JSONResponse(
            status_code=401,
            content=res.model_dump()
        )
    try:
        logger.info(f"Received thought generation request: {request.content} for {request.channel}")
        llm_service = LLMService()
        annotation_service = Annotation(llm_service)
        thought = await annotation_service.generate_thought(
            request,
            provider=providerName,
            SchemaFormat=SchemaFormat
            )
        return ThoughtResponse(output=thought)
    except ValueError as e:
        if "Authentication failed" in str(e):
            res = GlobalResponse(status=401, message=str(e))
            return JSONResponse(status_code=401, content=res.model_dump())
        else:
            res = GlobalResponse(status=400, message=str(e))
            return JSONResponse(status_code=400, content=res.model_dump())
    except Exception as e:
        logger.error(f"Error generating thought: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
@router.post("/translate", response_model=TranslateResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def translate(request: TranslateRequest,providerName: str = Header(...),providerKey: str = Header(...)):
    class SchemaFormat(BaseModel):
        output: str
    try:
        await set_provider(providerName, providerKey)
    except Exception as e:
        logger.error(f"Error setting provider: {str(e)}")
        res = GlobalResponse(status=400, message=f"Unsupported provider: {str(e)}")
        return JSONResponse(
            status_code=401,
            content=res.model_dump()
        )
    try:
        logger.info(f"Received translation request: {request.content} to  {request.destinationt_language}")
        llm_service = LLMService()
        annotation_service = Annotation(llm_service)
        translation = await annotation_service.generate_translation(
            request.content, 
            request.destinationt_language,
            provider=providerName,
            SchemaFormat=SchemaFormat
            )
        return TranslateResponse(output=translation)
    except ValueError as e:
        if "Authentication failed" in str(e):
            res = GlobalResponse(status=401, message=str(e))
            return JSONResponse(status_code=401, content=res.model_dump())
        else:
            res = GlobalResponse(status=400, message=str(e))
            return JSONResponse(status_code=400, content=res.model_dump())    
    except Exception as e:
        logger.error(f"Error generating translation: {str(e)}")        
        raise HTTPException(status_code=500, detail="Internal server error")    



