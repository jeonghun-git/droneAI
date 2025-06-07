from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Iterator

# OpenAI API 형식 모델 정의
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    conversationId: Optional[str] = None
    parentMessageId: Optional[str] = None

class Delta(BaseModel):
    content: Optional[str] = None

class Choice(BaseModel):
    index: int
    delta: Optional[Delta] = None
    message: Optional[Message] = None
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage 