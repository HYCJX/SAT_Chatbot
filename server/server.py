from fastapi import FastAPI

from model import BotResponse, UserResponse

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/user_response", response_model=BotResponse)
async def handle_user_response(userResponse: UserResponse):
    user_utterance = userResponse.utterance
