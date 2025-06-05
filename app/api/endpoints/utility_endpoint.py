from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.models.schemas import GlobalResponse, LlmProviderEnum
from ...core.logger import logger
from openai import AsyncOpenAI
from pydantic import BaseModel
import traceback


router = APIRouter()

class APIKeyCheckRequest(BaseModel):
    provideName: str
    provideKey: str

@router.post("/validateKey")
async def check_openai_key(body: APIKeyCheckRequest):
    """
    Validate the API key using the given provider, Currently supports OpenAI only.
    """
    if body.provideName != LlmProviderEnum.openAI:
        res = GlobalResponse(status=400, message=f"Unsupported provider: {body.provideName}")
        return JSONResponse(
            content=res.model_dump(),
            status_code=400
        )

    try:
        client = AsyncOpenAI(api_key=body.provideKey)
        await client.models.list()
        logger.info("OpenAI API key is valid.")
        res = GlobalResponse(status=200, message="API key is valid.")
        return  JSONResponse(
            content=res.model_dump(),
            status_code=200
        )
    except Exception as e:
        logger.error(f"Error checking OpenAI API key: {str(e)}")
        logger.error(traceback.format_exc())
        status_code = 401
        res = GlobalResponse(status=status_code, message=f"Unauthorized key")
        return JSONResponse(
            content=res.model_dump(),
            status_code=status_code
        )