from email import message
import logging
import uvicorn

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from chatbot import SATChatBot
from nlp_modules.emotion_classifier import EmotionClassifier
from nlp_modules.response_generator import ResponseGenerator
from type_model import BotResponse, UserResponse

from nlp_modules.emotion_classifier import SentimentClassifer

logging.basicConfig(filename="server.log",
                    filemode="a",
                    level=logging.INFO,
                    format="%(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")

sentiment_classifier = SentimentClassifer(model_type="facebook/muppet-roberta-base",
                                          model_checkpoint="results/sentiment_analysis_outputs/classification/facebookmuppet-roberta-base")
response_generator = ResponseGenerator(
    model_path="results/response_generation_outputs/daily_on_em/daily_on_em_epoch=5"
)
chatbot = SATChatBot(sentiment_classifier=sentiment_classifier,
                     response_generator=response_generator)
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/user_response", response_model=BotResponse)
async def handle_user_response(userResponse: UserResponse):
    user_utterance = userResponse.utterance
    response = chatbot.process_and_respond(user_utterance)
    bot_response = BotResponse(message=response)
    return bot_response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
