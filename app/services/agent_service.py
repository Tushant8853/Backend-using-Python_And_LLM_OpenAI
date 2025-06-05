# app/services/agent_service.py
from langchain.chains import LLMChain
from typing import Any
from pydantic import BaseModel
from app.constants import avoided_ai_cliches
from app.core.logger import logger
from app.models.schemas import LLMRequest, GetVariantResponse, SupportedChannel, PDFSummaryResponse, LlmProviderEnum
from app.services.llm_service import LLMService
import traceback
import json
from PyPDF2 import PdfReader
from io import BytesIO
import tempfile
import os
import subprocess
import uuid
import shutil
import openai
from app.utils.helper import get_channel_character_limit

# Model definitions
class Post(BaseModel):
    linkedin: str
    twitter: str

class IdeasResponse(BaseModel):
    ideas: list[str]

class PostsResponse(BaseModel):
    posts: list[Post]

class TextVariantAgent:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.variants = {
            "linkedin_to_twitter": self._setup_linkedin_to_twitter_chain(),
            "twitter_to_linkedin": self._setup_twitter_to_linkedin_chain(),
            "linkedin_to_linkedin": self._setup_linkedin_to_linkedin_chain(),
            "twitter_to_twitter": self._setup_twitter_to_twitter_chain(),
        }
        logger.info("TextVariantAgent initialized with variant templates")

    def _setup_linkedin_to_twitter_chain(self):
        return """
        Create a Twitter variation of the LinkedIn post provided at the end of the prompt below.

        Instructions:
        - Do not change the main message or the essence of the message.
        - Use this language {target_language} to create the variants.
        - Maintain the character limit of 250.
        - Space out the sentences for easier readability
        - Avoid hashtags unless absolutely necessary.
        - Use concise phrasing and line breaks (`\n` for single line, `\n\n` for paragraph separation) for better readability.
        - Do not break line and paragraph in this only add single statement in array element
        
        Linkedin Post:
        {input_text}
        """

    def _setup_twitter_to_linkedin_chain(self):
        return """
        Create a LinkedIn variation of the Twitter post provided at the end of the prompt below.

        Instructions:
        - Do not change the main message or the essence of the message.
        - Use this language {target_language} to create the variants.
        - Maintain the character limit of 2000.
        - Space out the sentences for easier readability
        - Avoid hashtags unless absolutely necessary.
        - Use concise phrasing and line breaks (`\n` for single line, `\n\n` for paragraph separation) for better readability.
        - Do not break line and paragraph in this only add single statement in array element
        
        Twitter Post:
        {input_text}
        """

    def _setup_linkedin_to_linkedin_chain(self):
        return """
        Create a new variation of the LinkedIn post provided at the end of the prompt below.

        Instructions:
        - Do not change the main message or the essence of the message. 
        - Use this language {target_language} to create the variants.
        - Change the hook or the intro in the first two lines
        - Vary tone and style
        - Emphasize a different angle or message
        - Reorder the structure of the post
        - Add a new hashtag and only use one hashtag from the original post.
        - Space out the sentences for easier readability.
        - If there is a signature or a call to action, then use the same in the variant.
        - Maintain the character limit of 2000.
        - Use concise phrasing and line breaks (`\n` for single line, `\n\n` for paragraph separation) for better readability. 
        - Do not break line and paragraph in this only add single statement in array element
        
        Linkedin Post:
        {input_text}
        """

    def _setup_twitter_to_twitter_chain(self):
        return """
        Create a new variation of the Twitter post provided at the end of the prompt below.

        Instructions:
        - Do not change the main message or the essence of the message. 
        - Use this language {target_language} to create the variants.
        - Change the hook or the intro in the first two lines
        - Vary tone and style
        - Emphasize a different angle or message
        - Reorder the structure of the post
        - Add a new hashtag and only use one hashtag from the original post.
        - Space out the sentences for easier readability.
        - If there is a signature or a call to action, then use the same in the variant.
        - Maintain the character limit of 250.
        - Use concise phrasing and line breaks (`\n` for single line, `\n\n` for paragraph separation) for better readability. 
        - Do not break line and paragraph in this only add single statement in array element
        
        Twitter Post:
        {input_text}
        """

    async def create_variant(self, text: str, active_current_channel: str, destination_channel: str, instructions: str , count: int = 1,target_language: str = None, provider: str = LlmProviderEnum.openAI, SchemaFormat: Any = None) -> GetVariantResponse:
        if active_current_channel == SupportedChannel.LINKEDIN and destination_channel == SupportedChannel.TWITTER:
            variant_type = "linkedin_to_twitter"
        elif active_current_channel == SupportedChannel.TWITTER and destination_channel == SupportedChannel.LINKEDIN:
            variant_type = "twitter_to_linkedin"
        elif active_current_channel == SupportedChannel.LINKEDIN and destination_channel == SupportedChannel.LINKEDIN:
            variant_type = "linkedin_to_linkedin"
        elif active_current_channel == SupportedChannel.TWITTER and destination_channel == SupportedChannel.TWITTER:
            variant_type = "twitter_to_twitter"
        else:
            raise ValueError("Invalid variant type")
        try:
            template = self.variants[variant_type]
            if instructions:
                template += f"\nAdditional Instructions:\n{instructions}"

            formatted_prompt = template.format(input_text=text, count=count, target_language=target_language)
            llm_chat_msgs = [{"role": "user", "content": formatted_prompt}]

            request = LLMRequest(
                messages=llm_chat_msgs,
                provider=provider, 
                max_tokens=1000,
                temperature=0.7,
                SchemaFormat=SchemaFormat
            )

            response = await self.llm_service.generate_text(request)
            try:
                response_text = response.text
                if provider == LlmProviderEnum.openAI:
                    logger.info(f"Parsed LLM Response: {response_text}")
                    if isinstance(response_text, SchemaFormat) and hasattr(response_text, "variants"):
                        if isinstance(response_text.variants, list) and all(isinstance(v, str) for v in response_text.variants):
                            return GetVariantResponse(**response_text.__dict__)

                    logger.error("Invalid JSON structure: Missing 'variants' key or incorrect format.")
                    raise ValueError("Invalid structure")
                else:
                    parsed_response = json.loads(response_text, strict=False)
                    if isinstance(parsed_response, dict) and "variants" in parsed_response:
                        if isinstance(parsed_response["variants"], list) and all(isinstance(v, str) for v in parsed_response["variants"]):
                            return GetVariantResponse(**parsed_response)
                    logger.error("Invalid JSON structure: Missing 'variants' key or incorrect format.")
                    raise ValueError("Invalid structure")
            except Exception as e:
                logger.error(traceback.format_exc())
                raise ValueError(e)
        except Exception as e:
            logger.error(f"Error creating variant: {e}")
            logger.error(traceback.format_exc())
            raise


class ReadPDFAgent:
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    def extract_text_from_pdf(self, pdf_file: bytes) -> str:
        try:
            pdf_reader = PdfReader(BytesIO(pdf_file))
            text = "".join([page.extract_text() for page in pdf_reader.pages])
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            logger.info(traceback.format_exc())
            raise ValueError(f"Failed to process PDF file: {str(e)}")

    def extract_text_from_txt(self, txt_file: bytes) -> str:
        try:
            return txt_file.decode('utf-8')
        except Exception as e:
            logger.error(f"Error reading .txt file: {e}")
            logger.info(traceback.format_exc())
            raise ValueError(f"Failed to process text file: {str(e)}")

    def extract_text_from_srt(self, srt_file: bytes) -> str:
        try:
            decoded = srt_file.decode('utf-8')
            lines = decoded.splitlines()
            text_lines = [line.strip() for line in lines if line.strip() and not line.strip().isdigit() and "-->" not in line]
            return " ".join(text_lines)
        except Exception as e:
            logger.error(f"Error reading .srt file: {e}")
            logger.info(traceback.format_exc())
            raise ValueError(f"Failed to process SRT file: {str(e)}")

    def extract_text_from_vtt(self, vtt_file: bytes) -> str:
        try:
            decoded = vtt_file.decode('utf-8')
            lines = decoded.splitlines()
            text_lines = [line.strip() for line in lines if line.strip() and not line.startswith("WEBVTT") and "-->" not in line]
            return " ".join(text_lines)
        except Exception as e:
            logger.error(f"Error reading .vtt file: {e}")
            logger.info(traceback.format_exc())
            raise ValueError(f"Failed to process VTT file: {str(e)}")

    async def extract_text_from_audio(self, audio_path: str) -> str:
        try:
            print("Audio Path :::::::::::", audio_path, ":::::")
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            with open(audio_path, "rb") as audio_file:
                response = await client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=audio_file,
                    response_format="text"
                )
            print("Response of the Audio File ::::::::", response, "::::")
            return response
        except Exception as e:
            logger.error(f"Error extracting text from audio: {e}")
            logger.info(traceback.format_exc())
            return None
    
    def extract_audio_from_mp4(self, video_bytes: bytes) -> str:
        try:
            # Use local ffmpeg path only if not already available
            if not shutil.which("ffmpeg"):
                os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\ffmpeg-2025-03-31-git-35c091f4b7-full_build\bin"

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as video_file:
                video_file.write(video_bytes)
                video_temp_path = video_file.name

            audio_temp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")

            subprocess.run(
                ["ffmpeg", "-i", video_temp_path, "-vn", "-acodec", "libmp3lame", audio_temp_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            os.remove(video_temp_path)
            return audio_temp_path

        except Exception as e:
            logger.error(f"Error extracting audio from MP4: {e}")
            logger.info(traceback.format_exc())
            raise ValueError(f"Failed to process video file: {str(e)}")

    def generate_idea_prompt(self, extracted_text: str, count: int, topicDescription: str = None) -> str:
        base_prompt = f"""
        Generate {count} unique and innovative ideas based on the following content.
        Each idea should:
        - Begin with a detailed summary of the source content
        - Summarize the content in a concise way in 1000 words
        - Have a different angle or perspective
        - Be clear and actionable
        - Include potential benefits or impact
        - Idea will be used to generate posts for LinkedIn and Twitter
        - Text should contain the idea and the summary of the source content in the same text
        - Do not break line and paragraph in this only add single statement in array element

        Return the response in this exact format:
        {{
            "ideas": [
                "Complete idea 1 text here",
                "Complete idea 2 text here",
                "Complete idea 3 text here"
            ]
        }}
        """

        if extracted_text:
            base_prompt += f"""
            Source Content:
            {extracted_text}
            """

        if topicDescription:
            base_prompt += f"""
            Topic Focus:
            {topicDescription}
            """

        base_prompt += """
        Format Instructions:
        - Each idea must include a detailed summary of the source content before diving into the idea
        - Do not break line and paragraph in this only add single statement in array element
        - Text should contain the idea and the summary of the source content in the same text
        - Text should be the 1200 words
        - Keep ideas concise and well-structured
        - Use natural and engaging language
        - Avoid cliches and generic statements
        - {avoided_ai_cliches}
        """

        return base_prompt

    def generate_posts_from_ideas_prompt(self, ideas: list, count: int, content_type: str, post_type: str, tone: str, style: str, emojis: str, language: str, signatureToAppend: str = None, promptInstructions: str = None, hook: str = None) -> str:
        base_prompt = f"""
        Based on the following Text which contains the summary of the source content and the idea, generate both LinkedIn and Twitter posts.
        Text contains the summary of the source content and the idea
        So use the text to generate the posts for LinkedIn and Twitter
        For each Text in the list, create one LinkedIn post and one Twitter post that effectively communicates that idea.

        Ideas to convert into posts:
        {json.dumps(ideas, indent=2)}

        Important Instructions:
        - Create one LinkedIn and one Twitter post for EACH Text (which contains the summary of the source content + the idea) in the list
        - LinkedIn posts should be professional and detailed {get_channel_character_limit("LinkedIn")}
        - Twitter posts should be concise and impactful {get_channel_character_limit("Twitter")}
        - The response should be in {content_type} format
        - Post type should be: {post_type}

        [Writing Style Instructions]:
        - Generate content using a natural, engaging style suitable for social media
        - Ensure varied sentence structures and inject personality where appropriate
        - Keep content clear, concise, and authentic
        - Space out sentences for easier readability
        - Use this tone: {tone}
        - Use this style: {style}
        - Language should be: {language}
        - Limit emojis to: {emojis} per post
        - Count characters carefully ( {get_channel_character_limit("LinkedIn")} max for LinkedIn, {get_channel_character_limit("Twitter")} max for Twitter)
        - Use concise phrasing and appropriate line breaks
        - {avoided_ai_cliches}
        """

        if promptInstructions:
            base_prompt += f"""
            Additional Instructions:
            {promptInstructions}
            """

        if hook:
            base_prompt += f"""
            Hook Instructions:
            Use this hook at the beginning of each post (you may adapt it to fit the platform and content):
            {hook}
            """

        if signatureToAppend:
            base_prompt += f"""
            Signature:
            Add this exact signature at the end of each post:
            {signatureToAppend}
            """

        base_prompt += """
        Return the response in this exact JSON format:
        {
            "posts": [
                {
                    "linkedin": "LinkedIn post content here",
                    "twitter": "Twitter post content here"
                },
                // One object for each idea in the input list
            ]
        }
        """

        return base_prompt

    async def extract_text_from_multiple_files(self, files_data: list) -> str:
        """
        Extract text from multiple files and combine them.
        
        Args:
            files_data: List of tuples containing (file_content: bytes, filetype: str)
            
        Returns:
            Combined extracted text from all files
        """
        try:
            all_extracted_text = []
            
            for file_content, filetype in files_data:
                try:
                    extracted_text = None
                    if filetype.endswith(".pdf"):
                        extracted_text = self.extract_text_from_pdf(file_content)
                    elif filetype.endswith(".txt"):
                        extracted_text = self.extract_text_from_txt(file_content)
                    elif filetype.endswith(".srt"):
                        extracted_text = self.extract_text_from_srt(file_content)
                    elif filetype.endswith(".vtt"):
                        extracted_text = self.extract_text_from_vtt(file_content)
                    elif filetype.endswith(".mp3"):
                        extracted_text = await self.extract_text_from_audio(file_content)
                        try:
                            os.remove(file_content)
                        except:
                            pass
                    elif filetype.endswith(".mp4"):
                        audio_path = self.extract_audio_from_mp4(file_content)
                        if audio_path:
                            extracted_text = await self.extract_text_from_audio(audio_path)
                            try:
                                os.remove(audio_path)
                            except:
                                pass
                    
                    if extracted_text:
                        all_extracted_text.append(f"\nContent from {filetype} file:\n{extracted_text}")
                        
                except Exception as e:
                    logger.error(f"Error processing file {filetype}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue
            
            return "\n".join(all_extracted_text) if all_extracted_text else None
            
        except Exception as e:
            logger.error(f"Error in extract_text_from_multiple_files: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def create_ideas_and_posts(self, files_data: list, parsed_data: dict, provider: str, SchemaFormat: any = None) -> PDFSummaryResponse:
        try:
            # Get the maximum count from channels
            max_count = max(
                details.get("count", 1) 
                for channels, details in parsed_data.get("channels", {}).items()
            )

            # Step 1: Extract text from all files
            extracted_text = await self.extract_text_from_multiple_files(files_data)

            if not extracted_text and files_data:
                return PDFSummaryResponse(
                    status_code=500,
                    status_msg="Failed to extract text from any of the files",
                    data={}
                )

            # Step 2: Generate Ideas using max count from channels
            try:
                idea_prompt = self.generate_idea_prompt(
                    extracted_text=extracted_text if extracted_text else parsed_data.get("topicDescription", ""),
                    count=max_count,
                    topicDescription=parsed_data.get("topicDescription")
                )

                idea_request = LLMRequest(
                    messages=[{"role": "user", "content": idea_prompt}],
                    provider=provider,
                    temperature=0.7,
                    max_tokens=1000,
                    SchemaFormat=IdeasResponse if provider == LlmProviderEnum.openAI else None
                )

                idea_response = await self.llm_service.generate_text(idea_request)
                
                if not idea_response or not idea_response.text:
                    return PDFSummaryResponse(
                        status_code=500,
                        status_msg="Failed to generate ideas: Empty response from service",
                        data={}
                    )

                # Parse ideas response
                if provider == LlmProviderEnum.openAI:
                    ideas = idea_response.text.ideas if hasattr(idea_response.text, "ideas") else []
                else:
                    try:
                        response_dict = json.loads(idea_response.text)
                        ideas = response_dict.get("ideas", [])
                    except json.JSONDecodeError:
                        return PDFSummaryResponse(
                            status_code=500,
                            status_msg="Invalid ideas response format",
                            data={}
                        )

                if not ideas:
                    return PDFSummaryResponse(
                        status_code=500,
                        status_msg="No ideas generated",
                        data={}
                    )

                # Step 3: Generate Posts from Ideas
                posts_prompt = self.generate_posts_from_ideas_prompt(
                    ideas=ideas,
                    count=max_count,
                    content_type=parsed_data.get("contentType", "text"),
                    post_type=parsed_data.get("postType", "general"),
                    tone=parsed_data.get("tone", "professional"),
                    style=parsed_data.get("style", "engaging"),
                    emojis=parsed_data.get("emojis", "2"),
                    language=parsed_data.get("language", "English"),
                    signatureToAppend=parsed_data.get("signatureToAppend"),
                    promptInstructions=parsed_data.get("promptInstructions"),
                    hook=parsed_data.get("hook")
                )

                posts_request = LLMRequest(
                    messages=[{"role": "user", "content": posts_prompt}],
                    provider=provider,
                    temperature=0.7,
                    max_tokens=2000,
                    SchemaFormat=PostsResponse if provider == LlmProviderEnum.openAI else None
                )

                posts_response = await self.llm_service.generate_text(posts_request)

                if not posts_response or not posts_response.text:
                    logger.error("Empty response from LLM service")
                    return PDFSummaryResponse(
                        status_code=500,
                        status_msg="Failed to generate posts: Empty response from service",
                        data={}
                    )

                logger.info(f"Raw LLM Response: {posts_response.text}")
                logger.info(f"Response type: {type(posts_response.text)}")

                # Parse posts response
                if provider == LlmProviderEnum.openAI:
                    posts = []
                    if hasattr(posts_response.text, 'posts') and isinstance(posts_response.text.posts, list):
                        logger.info(f"Processing OpenAI response with {len(posts_response.text.posts)} posts")
                        posts = [
                            {
                                'linkedin': post.linkedin,
                                'twitter': post.twitter
                            }
                            for post in posts_response.text.posts
                        ]
                    elif isinstance(posts_response.text, dict) and 'posts' in posts_response.text:
                        posts = posts_response.text['posts']
                else:
                    try:
                        response_dict = json.loads(posts_response.text)
                        posts = response_dict.get("posts", [])
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse posts response: {posts_response.text}")
                        return PDFSummaryResponse(
                            status_code=500,
                            status_msg="Invalid posts response format",
                            data={}
                        )

                if not posts:
                    logger.error(f"No posts generated. Response: {posts_response.text}")
                    return PDFSummaryResponse(
                        status_code=500,
                        status_msg="No posts generated",
                        data={}
                    )

                # Format response to match endpoint expectations
                channel_variants = {
                    "linkedin": [],
                    "twitter": []
                }

                # Separate posts by channel
                for post in posts:
                    logger.info(f"Processing post: {post}")
                    if isinstance(post, dict):
                        if 'linkedin' in post:
                            channel_variants["linkedin"].append(post['linkedin'])
                        if 'twitter' in post:
                            channel_variants["twitter"].append(post['twitter'])

                logger.info(f"Processed variants: {channel_variants}")

                # Create final response format matching the endpoint's expected structure
                unique_posts = []
                max_posts = max(
                    details.get("count", 1) 
                    for channel, details in parsed_data.get("channels", {}).items()
                )

                # Create post objects with all channels
                for i in range(max_posts):
                    post_obj = {}
                    for channel, posts_list in channel_variants.items():
                        if i < len(posts_list):
                            post_obj[channel.lower()] = posts_list[i]
                    if post_obj:
                        unique_posts.append(post_obj)

                logger.info(f"Final unique posts: {unique_posts}")

                # Create and return the response
                response = PDFSummaryResponse(
                    status_code=200,
                    status_msg="Success",
                    data={
                        "variants": unique_posts
                    }
                )
                
                logger.info(f"Final response: {response.dict()}")
                return response

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in idea/post generation: {error_msg}")
                logger.error(traceback.format_exc())
                
                if "Authentication failed" in error_msg or "invalid_api_key" in error_msg or "Incorrect API key" in error_msg:
                    return PDFSummaryResponse(
                        status_code=401,
                        status_msg=f"Authentication failed: {error_msg}",
                        data={}
                    )
                return PDFSummaryResponse(
                    status_code=500,
                    status_msg=f"Error generating content: {error_msg}",
                    data={}
                )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Unexpected error in create_ideas_and_posts: {error_msg}")
            logger.error(traceback.format_exc())
            return PDFSummaryResponse(
                status_code=500,
                status_msg=f"Internal server error: {error_msg}",
                data={}
            )
