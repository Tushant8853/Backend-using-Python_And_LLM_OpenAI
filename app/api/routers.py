# app/api/routes.py
from fastapi import APIRouter
from .endpoints import generate_endpoints, ask_endpoints,utility_endpoint,webpage_endpoint

api_router = APIRouter()

api_router.include_router(generate_endpoints.router, prefix="/variant", tags=["llm"])
api_router.include_router(ask_endpoints.router, prefix="/ask", tags=["ask"])
api_router.include_router(utility_endpoint.router, prefix="/utility", tags=["utility"])
api_router.include_router(webpage_endpoint.router, prefix="/web", tags=["web"])