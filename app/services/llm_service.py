# app/services/llm_service.py
from abc import ABC, abstractmethod
import os
import openai
from app.core.config import settings
from app.core.logger import logger
from app.models.schemas import LLMRequest, LLMResponse, LlmProviderEnum
from google import genai

class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        pass

class OpenAIProvider(LLMProvider):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            # Lazily import to avoid loading all providers
            from openai import AsyncOpenAI as OpenAI
            
            # openai.api_key = settings.LLM_API_KEYS.get("openai", "")
            # if not openai.api_key:
                # raise ValueError("OpenAI API key is not set")
            
            model ="gpt-4o-mini"
            
            logger.info(f"Generating text with OpenAI model: {model}")
            
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
            ) 
            content=str(request.messages[0]["content"])   
            response = await client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": content},
                ],
                response_format=request.SchemaFormat,
                temperature=0.9
            )
            final_text=response.choices[0].message.parsed
            return LLMResponse(
                text=final_text,
                metadata={
                    "usage": response.usage.to_dict() if hasattr(response, "usage") else {},
                    "finish_reason": response.choices[0].finish_reason if hasattr(response.choices[0], "finish_reason") else None
                },
            )
        except ImportError:
            logger.error("OpenAI package is not installed")
            raise ValueError("OpenAI package is not installed. Install it with 'pip install openai'")
        except openai.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise ValueError("Authentication failed. Please check your API key.")
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            raise


class AnthropicProvider(LLMProvider):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            # Lazily import to avoid loading all providers
            import anthropic
            
            api_key = settings.LLM_API_KEYS.get("anthropic", "")
            if not api_key:
                raise ValueError("Anthropic API key is not set")
            
            model = request.model or "claude-3-opus-20240229"
            
            logger.info(f"Generating text with Anthropic model: {model}")
            
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=[{"role": "user", "content": request.prompt}],
                **request.additional_params
            )
            
            return LLMResponse(
                text=response.content[0].text,
                provider="anthropic",
                model=model,
                metadata={
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    } if hasattr(response, "usage") else {},
                    "stop_reason": response.stop_reason if hasattr(response, "stop_reason") else None
                }
            )
        except ImportError:
            logger.error("Anthropic package is not installed")
            raise ValueError("Anthropic package is not installed. Install it with 'pip install anthropic'")
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {e}")
            raise

class LocalLLMProvider(LLMProvider):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            logger.info(f"Generating text with local LLM: {settings.LOCAL_LLM_TYPE}")
            
            if not settings.LOCAL_LLM_PATH:
                raise ValueError("Local LLM path is not set")
            
            # This implementation depends on which local LLM framework you're using
            # Here's an example for using LangChain with a local LLM
            
            from langchain.llms import LlamaCpp
            
            model_path = settings.LOCAL_LLM_PATH
            llm = LlamaCpp(
                model_path=model_path,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                **request.additional_params
            )
            
            response = llm(request.prompt)
            
            return LLMResponse(
                text=response,
                provider="local",
                model=settings.LOCAL_LLM_TYPE,
                metadata={
                    "model_path": model_path,
                }
            )
        except ImportError:
            logger.error("Required packages for local LLM are not installed")
            raise ValueError("Required packages for local LLM are not installed")
        except Exception as e:
            logger.error(f"Error generating text with local LLM: {e}")
            raise

class AzureOpenAIProvider(LLMProvider):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            # Lazily import to avoid loading all providers
            import openai
            from openai import AzureOpenAI
            
            # Get Azure OpenAI settings
            api_key = settings.LLM_API_KEYS.get("azure_openai", "")
            endpoint = settings.AZURE_OPENAI_ENDPOINT
            api_version = settings.AZURE_OPENAI_API_VERSION
            deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME or request.model
            
            if not api_key or not endpoint:
                raise ValueError("Azure OpenAI API key or endpoint is not set")
            
            if not deployment_name:
                raise ValueError("Azure OpenAI deployment name is not specified")
            
            logger.info(f"Generating text with Azure OpenAI deployment: {deployment_name}")
            
            # Initialize Azure OpenAI client
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
            
            # Create the completion
            response = await client.chat.completions.create(
                model=deployment_name,  # This is the deployment name in Azure
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                **request.additional_params
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            return LLMResponse(
                text=response_text,
                provider="azure_openai",
                model=deployment_name,
                metadata={
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    } if hasattr(response, "usage") else {},
                    "finish_reason": response.choices[0].finish_reason if hasattr(response.choices[0], "finish_reason") else None
                }
            )
        except ImportError:
            logger.error("OpenAI package is not installed")
            raise ValueError("OpenAI package is not installed. Install it with 'pip install openai'")
        except Exception as e:
            logger.error(f"Error generating text with Azure OpenAI: {e}")
            raise

class AnthropicProvider(LLMProvider):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            # Lazily import to avoid loading all providers
            import anthropic
            
            api_key = settings.LLM_API_KEYS.get("anthropic", "")
            if not api_key:
                raise ValueError("Anthropic API key is not set")
            
            model = request.model or "claude-3-opus-20240229"
            
            logger.info(f"Generating text with Anthropic model: {model}")
            
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                messages=[{"role": "user", "content": request.prompt}],
                **request.additional_params
            )
            
            return LLMResponse(
                text=response.content[0].text,
                provider="anthropic",
                model=model,
                metadata={
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens
                    } if hasattr(response, "usage") else {},
                    "stop_reason": response.stop_reason if hasattr(response, "stop_reason") else None
                }
            )
        except ImportError:
            logger.error("Anthropic package is not installed")
            raise ValueError("Anthropic package is not installed. Install it with 'pip install anthropic'")
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {e}")
            raise

class LocalLLMProvider(LLMProvider):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            logger.info(f"Generating text with local LLM: {settings.LOCAL_LLM_TYPE}")
            
            if not settings.LOCAL_LLM_PATH:
                raise ValueError("Local LLM path is not set")
            
            # This implementation depends on which local LLM framework you're using
            # Here's an example for using LangChain with a local LLM
            
            from langchain.llms import LlamaCpp
            
            model_path = settings.LOCAL_LLM_PATH
            llm = LlamaCpp(
                model_path=model_path,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                **request.additional_params
            )
            
            response = llm(request.prompt)
            
            return LLMResponse(
                text=response,
                provider="local",
                model=settings.LOCAL_LLM_TYPE,
                metadata={
                    "model_path": model_path,
                }
            )
        except ImportError:
            logger.error("Required packages for local LLM are not installed")
            raise ValueError("Required packages for local LLM are not installed")
        except Exception as e:
            logger.error(f"Error generating text with local LLM: {e}")
            raise

class AzureOpenAIProvider(LLMProvider):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            # Lazily import to avoid loading all providers
            import openai
            from openai import AzureOpenAI
            
            # Get Azure OpenAI settings
            api_key = settings.LLM_API_KEYS.get("azure_openai", "")
            endpoint = settings.AZURE_OPENAI_ENDPOINT
            api_version = settings.AZURE_OPENAI_API_VERSION
            deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME or request.model
            
            if not api_key or not endpoint:
                raise ValueError("Azure OpenAI API key or endpoint is not set")
            
            if not deployment_name:
                raise ValueError("Azure OpenAI deployment name is not specified")
            
            logger.info(f"Generating text with Azure OpenAI deployment: {deployment_name}")
            
            # Initialize Azure OpenAI client
            client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
            
            # Create the completion
            response = await client.chat.completions.create(
                model=deployment_name,  # This is the deployment name in Azure
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                **request.additional_params
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            return LLMResponse(
                text=response_text,
                provider="azure_openai",
                model=deployment_name,
                metadata={
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    } if hasattr(response, "usage") else {},
                    "finish_reason": response.choices[0].finish_reason if hasattr(response.choices[0], "finish_reason") else None
                }
            )
        except ImportError:
            logger.error("OpenAI package is not installed")
            raise ValueError("OpenAI package is not installed. Install it with 'pip install openai'")
        except Exception as e:
            logger.error(f"Error generating text with Azure OpenAI: {e}")
            raise

class GeminiProvider(LLMProvider):
    async def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini API key is not set")
            input_prompt =  request.messages[0]["content"]
            model = "gemini-1.5-flash"
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=model, contents=input_prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': request.SchemaFormat
                },
            )

            return LLMResponse(
                text=response.text,
                provider=LlmProviderEnum.gemini,
                model=model,
                metadata={}
            )
        except Exception as e:
            err_msg = str(e)
            if "API key not valid" in err_msg or "API_KEY_INVALID" in err_msg:
                raise ValueError("Authentication failed. Please check your Gemini API key.")

            logger.error(f"Error generating text with Gemini: {err_msg}")
            raise

class LLMService:
    def __init__(self):
        self.providers = {
            LlmProviderEnum.openAI: OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "azure_openai": AzureOpenAIProvider(), 
            "local": LocalLLMProvider(),
            LlmProviderEnum.gemini: GeminiProvider()
        }
    
    async def generate_text(self, request: LLMRequest) -> LLMResponse:
        provider_name = request.provider or settings.DEFAULT_LLM_PROVIDER
        
        if provider_name not in self.providers:
            logger.error(f"LLM provider {provider_name} not supported")
            raise ValueError(f"LLM provider {provider_name} not supported")
        
        logger.info(f"Using LLM provider: {provider_name}")
        provider = self.providers[provider_name]
        
        try:
            response = await provider.generate(request)
            return response
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

if __name__ == "__main__":
    import asyncio
    from app.services.llm_service import LLMService
    from app.models.schemas import LLMRequest
    provider_name = LlmProviderEnum.gemini
    text = "Convert below twitter text to linkedin post: Swiggy raised 50Mn$ in series B round."
    messages = [{"role": "user", "content": text}]

    request = LLMRequest(
        messages=messages,
        provider=provider_name,
        model="gpt-4o-mini" if provider_name == LlmProviderEnum.openAI else "gemini-pro",
        temperature=0.7,
        max_tokens=100,
        additional_params={"stop": ["\n"]}
    )

    async def run():
        llm_service = LLMService()
        response = await llm_service.generate_text(request)
        print(f"\n📌 Provider: {provider_name}")
        print(f"🧠 Response:\n{response.text}\n")

    asyncio.run(run())

    