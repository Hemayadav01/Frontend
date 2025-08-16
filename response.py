# response_feature.py
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, BaseSettings
from enum import Enum
from typing import Optional, AsyncGenerator, Any
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Secure API key management using Pydantic
class Settings(BaseSettings):
    openai_api_key: str

    class Config:
        env_file = ".env"  # Optional: load from .env file

settings = Settings()

# Enums for tone and length
class ToneEnum(str, Enum):
    formal = "formal"
    casual = "casual"
    friendly = "friendly"
    professional = "professional"

class LengthEnum(str, Enum):
    concise = "concise"
    detailed = "detailed"
    medium = "medium"

# Request model
class EmailRequest(BaseModel):
    email_text: str
    tone: Optional[ToneEnum] = ToneEnum.friendly
    length: Optional[LengthEnum] = LengthEnum.concise

# Helper to stream response chunks
async def stream_response_chunks(chain: LLMChain, inputs: dict) -> AsyncGenerator[str, Any]:
    """
    Asynchronously streams response chunks from LangChain LLM.
    """
    async for chunk in chain.astream(inputs):
        # Yield each chunk with a newline for easier frontend handling
        yield chunk["text"] + "\n"

@router.post("/generate_response")
async def generate_response(request: EmailRequest):
    """
    Generate a polite and professional email reply using LangChain GPT with streaming.
    """
    if not request.email_text.strip():
        raise HTTPException(status_code=400, detail="Email text cannot be empty.")

    try:
        # Initialize ChatOpenAI with secure API key and streaming
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=settings.openai_api_key,
            streaming=True,
        )

        # Dynamic prompt with tone and length
        prompt_template = (
            "You are a polite and professional assistant. "
            "Write a {tone} and {length} email reply to this message:\n\n"
            "{email_text}\n\nReply:"
        )
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)

        # Prepare inputs
        inputs = {
            "email_text": request.email_text,
            "tone": request.tone.value,
            "length": request.length.value
        }

        # Return streaming response (frontend-friendly)
        return StreamingResponse(
            stream_response_chunks(chain, inputs),
            media_type="text/event-stream"
        )

    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the response."
        )
