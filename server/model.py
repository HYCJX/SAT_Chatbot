from typing import List, Optional
from pydantic import BaseModel


class UserResponse(BaseModel):
    utterance: str


class BotResponse(BaseModel):
    utterance: str
    options: Optional[List[str]]
