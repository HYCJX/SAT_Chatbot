from typing import Optional

from nlp_modules.emotion_classifier import EmotionClassifier, SentimentClassifer
from nlp_modules.response_generator import ResponseGenerator


EMOTION_CAUSES = {}


class SATChatBot():
    def __init__(self,

                 sentiment_classifier: SentimentClassifer,
                 response_generator: ResponseGenerator,
                 emotion_classifier: Optional[EmotionClassifier] = None) -> None:
        # NLP Modules:
        self.emotion_classifier = emotion_classifier
        self.sentiment_classifier = sentiment_classifier
        self.response_generator = response_generator
        # States:
        self.emotion_history = []
        self.sentiment_history = []
        self.emotion_cause_dictionary = dict.fromkeys(EMOTION_CAUSES, False)

    def process_and_respond(self, utterance: str) -> str:
        # current_emotion = self.emotion_classifier.classify_utterance(utterance)
        # self.emotion_history.append(current_emotion)
        current_sentiment = self.sentiment_classifier.classify_utterance(
            utterance
        )
        self.sentiment_history.append(current_sentiment)
        response = self.response_generator.infer(utterance)
        return response
