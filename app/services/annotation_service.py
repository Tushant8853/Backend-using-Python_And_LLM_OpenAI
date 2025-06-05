import json

from app.constants import avoided_ai_cliches
from app.models.schemas import GrammarCheckRequest, GrammarCheckResponse, LLMRequest, LlmProviderEnum, CommentRequest, \
    ThoughtRequest
from app.services.llm_service import LLMService

class Annotation:
    def __init__(self ,llm_service: LLMService):
        self.llm_service = llm_service

    async def check_grammar(self, request: GrammarCheckRequest, provider: str, SchemaFormat: any = None) -> GrammarCheckResponse:
        prompt = f"Check the grammar of the following text. If incorrect, correct it:\n\n{request.content}\n\nCorrected text:"
        llm_request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            max_tokens=1000,
            temperature=0.7,
            SchemaFormat=SchemaFormat
        )
        try:
            response = await self.llm_service.generate_text(llm_request)
            print(response)

            if provider == LlmProviderEnum.gemini:
                corrected_text = response.text.strip()
                try:
                    corrected_data = json.loads(corrected_text)
                    corrected_text = corrected_data.get("corrected_text", corrected_text)
                except json.JSONDecodeError:
                    pass
            else:
                if isinstance(response.text, SchemaFormat):
                    corrected_text = response.text.corrected_text
                elif isinstance(response.text, str):
                    corrected_text = response.text.strip()
                else:
                    raise ValueError("Unsupported response format.")

            status_code = 200 if corrected_text != request.content else 201
            return corrected_text, status_code
        except Exception as e:
            raise ValueError(e)

    async def generate_summary(self, text: str, provider: str, SchemaFormat: any = None) -> str:
        prompt = f"Summarize the following text in a concise manner:\n\n{text}\n\nSummary:"
        llm_request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            max_tokens=1000,
            temperature=0.7,
            SchemaFormat=SchemaFormat
        )
        try:
            response = await self.llm_service.generate_text(llm_request)
            if provider == LlmProviderEnum.gemini:
                summary = response.text.strip()
                try:
                    summary_data = json.loads(summary)
                    summary = summary_data.get("summary", summary)
                except json.JSONDecodeError:
                    pass

                return summary
            else:
                if isinstance(response.text, SchemaFormat):
                    summary = response.text.summary
                elif isinstance(response.text, str):
                    summary = response.text.strip()
                else:
                    raise ValueError("Unsupported response format.")

                return summary
        except Exception as e:
            raise ValueError(e)

    async def adjust_tone(self, text: str, tone: str, provider: str, SchemaFormat: any = None ):
        prompt = f"""
        Rewrite the following text in a {tone} tone:
        **Text:**:{text}
        **Instructions:**
        - Stick to the original meaning of the text, but change the tone to {tone}.
        - Do not add any additional information or context.
        - Strictly follow the representation of the Output Format.
        - Nothing should be added beyond the Output Format.
        - Do **not** add, remove, or modify any contextual details.  
        **Output Format:** 
        {{"output": <str>}}
        """
        llm_request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            max_tokens=1000,
            temperature=0.7,
            SchemaFormat=SchemaFormat
        )
        try:
            response = await self.llm_service.generate_text(llm_request)
            if provider == LlmProviderEnum.gemini:
                text = response.text.strip()
                try:
                    text_data = json.loads(text)
                    text = text_data.get("output", text)
                except json.JSONDecodeError:
                    pass

                return text
            else:
                if isinstance(response.text, SchemaFormat):
                    text = response.text.output
                elif isinstance(response.text, str):
                    text = response.text.strip()
                else:
                    raise ValueError("Unsupported response format.")

                return text
        except Exception as e:
            raise ValueError(e)

    async def adjust_style(self, text: str, style: str, provider: str,SchemaFormat: any = None):
        prompt = f"""
        Rewrite the following text in a {style} style while preserving its original meaning:
        **Text:** {text}
        **Instructions:**  
        - Maintain the original meaning of the text.  
        - Change only the writing style to {style}.  
        - Do **not** add, remove, or modify any contextual details.  
        - **Strictly** follow the output format below.  


        """
        llm_request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            max_tokens=1000,
            temperature=0.7,
            SchemaFormat=SchemaFormat
        )
        try:
            response = await self.llm_service.generate_text(llm_request)
            if provider == LlmProviderEnum.gemini:
                text = response.text.strip()
                try:
                    text_data = json.loads(text)
                    text = text_data.get("output", text)
                except json.JSONDecodeError:
                    pass

                return text
            else:
                if isinstance(response.text, SchemaFormat):
                    text = response.text.output
                elif isinstance(response.text, str):
                    text = response.text.strip()
                else:
                    raise ValueError("Unsupported response format.")

                return text
        except Exception as e:
            raise ValueError(e)

    async def reduce_text(self, text: str, provider: str,SchemaFormat: any = None) -> str:
        prompt = f"Reduce the following text while keeping its meaning:\n\n{text}\n\nReduced version:"
        llm_request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            max_tokens=1000,
            temperature=0.7,
            SchemaFormat=SchemaFormat
        )
        try:
            response = await self.llm_service.generate_text(llm_request)
            if provider == LlmProviderEnum.gemini:
                text = response.text.strip()
                try:
                    text_data = json.loads(text)
                    text = text_data.get("output", text)
                except json.JSONDecodeError:
                    pass

                return text
            else:
                if isinstance(response.text, SchemaFormat):
                    text = response.text.output
                elif isinstance(response.text, str):
                    text = response.text.strip()
                else:
                    raise ValueError("Unsupported response format.")

                return text
        except Exception as e:
            raise ValueError(e)

    async def expand_text(self, text: str, provider: str, SchemaFormat: any = None) -> str:
        prompt = f"Expand and elaborate on the following text:\n\n{text}\n\nExpanded version:"
        llm_request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            max_tokens=1000,
            temperature=0.7,
            SchemaFormat=SchemaFormat
        )
        try:
            response = await self.llm_service.generate_text(llm_request)
            if provider == LlmProviderEnum.gemini:
                text = response.text.strip()
                try:
                    text_data = json.loads(text)
                    text = text_data.get("output", text)
                except json.JSONDecodeError:
                    pass

                return text
            else:
                if isinstance(response.text, SchemaFormat):
                    text = response.text.output
                elif isinstance(response.text, str):
                    text = response.text.strip()
                else:
                    raise ValueError("Unsupported response format.")

                return text
        except Exception as e:
            raise ValueError(e)

    async def generate_comment(self, request: CommentRequest, provider: str, SchemaFormat: any = None):
        prompt = f"""
        Write a comment to a post on {request.channel}.
        The post is given in the end as Source.
        Use it to generate a thoughtful comment that reads as if written by a human.
        """

        if request.instruction:
            prompt += f"""
            Use the text as instruction for the comment
            - {request.instruction}
            """

        if request.department:
            prompt += f"""
            The person who is commenting is in the {request.department} department.
            """

        if request.title:
            prompt += f"""
            The title of the person who is commenting is {request.title}.
            """

        prompt += f"""
        Start the comment by acknowledging the value of the post but make it simple and then add some additional value.

        [Writing Style Instructions]: 
        - Generate the response using a natural, engaging suitable for social media.
        - Ensure varied sentence structures and inject personality where appropriate. Keep it clear, concise, and authentic.
        - Avoid overly formal language, excessive jargon, and generic concluding phrases (like "in conclusion").
        - {avoided_ai_cliches}
        - Do not use hashtags.
        
        Post to be commented on: {request.content}"""

        llm_request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            max_tokens=1000,
            temperature=0.7,
            SchemaFormat=SchemaFormat
        )
        try:
            response = await self.llm_service.generate_text(llm_request)
            if provider == LlmProviderEnum.gemini:
                text = response.text.strip()
                try:
                    text_data = json.loads(text)
                    text = text_data.get("output", text)
                except json.JSONDecodeError:
                    pass

                return text
            else:
                if isinstance(response.text, SchemaFormat):
                    text = response.text.output
                elif isinstance(response.text, str):
                    text = response.text.strip()
                else:
                    raise ValueError("Unsupported response format.")

                return text
        except Exception as e:
            raise ValueError(e)

    async def generate_thought(self, request: ThoughtRequest, provider: str, SchemaFormat: any = None):
        prompt = f"""
                Write a repost thought to a post on {request.channel}.
                The post is given in the end as Source.
                """

        if request.instruction:
            prompt += f"""
                    Use the text as instruction for the thought
                    - {request.instruction}
                    """

        if request.department:
            prompt += f"""
            The person who is commenting is in the {request.department} department.
            """

        if request.title:
            prompt += f"""
            The title of the person who is commenting is {request.title}.
            """

        prompt += f"""
        Start the thought by acknowledging the value of the post but make it simple and then add some additional value.

        [Writing Style Instructions]:
        - Generate the response using a natural, engaging suitable for social media.
        - Ensure varied sentence structures and inject personality where appropriate. Keep it clear, concise, and authentic.
        - Avoid overly formal language, excessive jargon, and generic concluding phrases (like "in conclusion").
        - {avoided_ai_cliches}
        - Do not use hashtags.
        
        Post to be commented on: {request.content}"""
        llm_request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            max_tokens=1000,
            temperature=0.7,
            SchemaFormat=SchemaFormat
        )
        try:
            response = await self.llm_service.generate_text(llm_request)
            if provider == LlmProviderEnum.gemini:
                text = response.text.strip()
                try:
                    text_data = json.loads(text)
                    text = text_data.get("output", text)
                except json.JSONDecodeError:
                    pass

                return text
            else:
                if isinstance(response.text, SchemaFormat):
                    text = response.text.output
                elif isinstance(response.text, str):
                    text = response.text.strip()
                else:
                    raise ValueError("Unsupported response format.")

                return text
        except Exception as e:
            raise ValueError(e)
 
    async def generate_translation(self, content: str, destination_language: str, provider: str, SchemaFormat: any = None):
        prompt = f"""
        Translate the following content into {destination_language}:

        **Original Content:**
        {content}

        **Instructions:**
        - Provide a natural and accurate translation.
        - Use formal or neutral tone appropriate for general audiences.
        - Only return the translated text.
        """
        llm_request = LLMRequest(
            messages=[{"role": "user", "content": prompt}],
            provider=provider,
            max_tokens=1000,
            temperature=0.7,
            SchemaFormat=SchemaFormat
        )
        try:
            response = await self.llm_service.generate_text(llm_request)
            if provider == LlmProviderEnum.gemini:
                text = response.text.strip()
                try:
                    text_data = json.loads(text)
                    text = text_data.get("output", text)
                except json.JSONDecodeError:
                    pass

                return text
            else:
                if isinstance(response.text, SchemaFormat):
                    text = response.text.output
                elif isinstance(response.text, str):
                    text = response.text.strip()
                else:
                    raise ValueError("Unsupported response format.")

                return text
        except Exception as e:
            raise ValueError(e)

