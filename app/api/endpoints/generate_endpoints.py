# app/api/endpoints/llm_endpoints.py
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Header
from typing import List, Optional
import json
import tempfile
import shutil
import os
from ...models.schemas import ErrorResponse, GetVariantRequest, GetVariantResponse, PDFSummaryResponse, GlobalResponse, LlmProviderEnum
from ...services.llm_service import LLMService
from ...core.logger import logger
from ...utils.helper import validate_environment
from app.services.agent_service import TextVariantAgent,ReadPDFAgent
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import traceback

router = APIRouter()
#llm_service = LLMService()
#text_variant_agent = TextVariantAgent(llm_service)
# read_pdf_agent = ReadPDFAgent(llm_service)

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

@router.post("/getVariants", response_model=GetVariantResponse,responses={400: {"model": ErrorResponse},401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def generate_text(request: GetVariantRequest, providerName: str = Header(...),providerKey: str = Header(...)):
    """
    Generate text using the configured LLM provider.
    """
    class SchemaFormat(BaseModel):
        variants: list[str]
    try:
        await set_provider(providerName, providerKey)
    except Exception as e:
        logger.error(f"Error setting provider: {str(e)}")
        res = GlobalResponse(status=400, message=f"Unsupported provider: {str(e)}")
        return JSONResponse(
            status_code=400,
            content=res.model_dump()
        )
    try:
        #logger.info(f"Received text generation request: {request.content or 'default'} with count: {request.count}")
        llm_service = LLMService()
        text_variant_agent = TextVariantAgent(llm_service)
        response = await text_variant_agent.create_variant(
            request.content,
            request.active_current_channel,
            request.destination_channel,
            instructions=request.instructions,
            count=request.count,
            provider=providerName,
            target_language=request.target_language,
            SchemaFormat=SchemaFormat
        )

        return response
    except ValueError as e:
        if "Authentication failed" in str(e):
            res = GlobalResponse(status=401, message=str(e))
            return JSONResponse(status_code=401, content=res.model_dump())
        else:
            res = GlobalResponse(status=400, message=str(e))
            return JSONResponse(status_code=400, content=res.model_dump())
    except Exception as e:
        logger.error(f"Error checking grammar: {str(e)}")
        res = GlobalResponse(status=500, message=str(e))
        return JSONResponse(status_code=500, content=res.model_dump())
        #raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/createPost", response_model=PDFSummaryResponse, responses={400: {"model": ErrorResponse},401: {"model": ErrorResponse} , 500: {"model": ErrorResponse}})
async def generate_pdf_variant(data: str = Form(...), files: list[UploadFile] = File(None), providerName: str = Header(...), providerKey: str = Header(...)):
    class SchemaFormat(BaseModel):
        variants: list[str]
    
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
        parsed_data = json.loads(data)
        if not files and not parsed_data.get("topicDescription"):
            return JSONResponse(
                status_code=400,
                content=GlobalResponse(
                    status=400,
                    message="At least one of 'files' or meaningful 'data' (topicDescription) must be provided."
                ).model_dump()
            )

        files_data = []
        if files:
            for file in files:
                filetype = file.filename.lower()
                if not (filetype.endswith(".pdf")
                    or filetype.endswith(".txt") 
                    or filetype.endswith(".srt") 
                    or filetype.endswith(".vtt") 
                    or filetype.endswith(".mp3")
                    or filetype.endswith(".mp4")
                    ):
                    return JSONResponse(
                        status_code=400,
                        content=GlobalResponse(
                            status=400,
                            message=f"File {file.filename} is not supported. Only PDF, TXT, VTT, SRT, MP3, or MP4 files are supported."
                        ).model_dump()
                    )
                
                try:
                    if filetype.endswith(".mp3"):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                            shutil.copyfileobj(file.file, temp_audio)
                            temp_audio_path = temp_audio.name
                        files_data.append((temp_audio_path, filetype))
                    else:
                        file_content = await file.read()
                        files_data.append((file_content, filetype))
                    
                except Exception as file_error:
                    error_msg = str(file_error)
                    if "Authentication failed" in error_msg or "invalid_api_key" in error_msg or "Incorrect API key" in error_msg:
                        return JSONResponse(
                            status_code=401,
                            content=GlobalResponse(
                                status=401,
                                message=f"Authentication failed: {error_msg}"
                            ).model_dump()
                        )
                    return JSONResponse(
                        status_code=500,
                        content=GlobalResponse(
                            status=500,
                            message=f"Error processing file {file.filename}: {error_msg}"
                        ).model_dump()
                    )

            # Process all files together
            try:
                llm_service = LLMService()
                read_pdf_agent = ReadPDFAgent(llm_service)
                
                response = await read_pdf_agent.create_ideas_and_posts(
                    files_data,
                    parsed_data,
                    provider=providerName,
                    SchemaFormat=SchemaFormat
                )
                
                # Clean up temporary files
                for file_content, filetype in files_data:
                    if isinstance(file_content, str) and os.path.exists(file_content):
                        try:
                            os.remove(file_content)
                        except:
                            pass
                
                return response

            except Exception as e:
                error_msg = str(e)
                if "Authentication failed" in error_msg or "invalid_api_key" in error_msg or "Incorrect API key" in error_msg:
                    return JSONResponse(
                        status_code=401,
                        content=GlobalResponse(
                            status=401,
                            message=f"Authentication failed: {error_msg}"
                        ).model_dump()
                    )
                return JSONResponse(
                    status_code=500,
                    content=GlobalResponse(
                        status=500,
                        message=f"Error processing files: {error_msg}"
                    ).model_dump()
                )

        # If no files, process with topic description
        try:
            llm_service = LLMService()
            read_pdf_agent = ReadPDFAgent(llm_service)
            response = await read_pdf_agent.create_ideas_and_posts(
                [],  # Empty list for no files
                parsed_data,
                provider=providerName,
                SchemaFormat=SchemaFormat
            )
            return response

        except Exception as e:
            error_msg = str(e)
            if "Authentication failed" in error_msg or "invalid_api_key" in error_msg or "Incorrect API key" in error_msg:
                return JSONResponse(
                    status_code=401,
                    content=GlobalResponse(
                        status=401,
                        message=f"Authentication failed: {error_msg}"
                    ).model_dump()
                )
            return JSONResponse(
                status_code=500,
                content=GlobalResponse(
                    status=500,
                    message=f"Error generating content: {error_msg}"
                ).model_dump()
            )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error: {error_msg}")
        return JSONResponse(
            status_code=500,
            content=GlobalResponse(
                status=500,
                message=f"Internal server error: {error_msg}"
            ).model_dump()
        )

@router.get("/providers", response_model=List[str])
async def list_providers(openai_llm_api_key: str = Header(None)):
    """
    List available LLM providers.
    """
    await set_openai_key(openai_llm_api_key)
    try:
        llm_service = LLMService()
        providers = list(llm_service.providers.keys())
        logger.info(f"Listed available providers: {providers}")
        return providers
    except Exception as e:
        logger.error(f"Error listing providers: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def health_check(openai_llm_api_key: str = Header(None)):

    """
    Check the health of the application and its dependencies.
    """
    await set_openai_key(openai_llm_api_key)
    try:
        from ...db.mongodb import mongodb
        
        # Check environment
        missing_vars = validate_environment()
        llm_service = LLMService()
        if missing_vars:
            return {
                "status": "warning",
                "message": f"Missing environment variables: {', '.join(missing_vars)}",
                "mongodb": "not_checked",
                "providers": list(llm_service.providers.keys())
            }
        
        # Check MongoDB connection
        try:
            await mongodb.connect()
            mongo_status = "connected"
            mongo_message = "MongoDB connection successful"
        except Exception as e:
            mongo_status = "error"
            mongo_message = f"MongoDB connection failed: {str(e)}"
        
        return {
            "status": "ok" if mongo_status == "connected" else "error",
            "mongodb": {
                "status": mongo_status,
                "message": mongo_message
            },
            "providers": list(llm_service.providers.keys()),
            "environment": "valid"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}"
        }

