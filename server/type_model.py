from pydantic import BaseModel
from typing import List, Optional


class UserResponse(BaseModel):
    utterance: str


class BotResponse(BaseModel):
    message: str
    options: Optional[List[str]]
