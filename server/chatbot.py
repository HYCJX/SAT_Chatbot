from logging import getLogger

from nlp_modules.emotion_classifier import EmotionClassifier, SentimentClassifer
from nlp_modules.response_generator import ResponseGenerator
from nlp_modules.emotion_cause_recogniser import EmotionCauseRecogniser


EMOTION_CAUSES = [
    "abuse",
    "assessment_pressure",
    "break-up",
    "change_in_life",
    "disloyalty",
    "embarrassed",
    "family_death",
    "family_health",
    "financial_pressure",
    "friend_death",
    "health",
    "injustice",
    "irritated",
    "jealousy",
    "loneliness",
    "missed_expectation",
    "owe",
    "pet_loss",
    "societal_relationship",
    "stress",
    "trauma",
    "work_pressure"
]


class SATChatBot():
    def __init__(self,
                 sentiment_classifier: SentimentClassifer,
                 emotion_classifier: EmotionClassifier,
                 emotion_cause_recogniser: EmotionCauseRecogniser,
                 response_generator: ResponseGenerator) -> None:
        self.logger = getLogger("SAT Chatbot")
        self.turn = 0
        self.sentiment_classifier = sentiment_classifier
        self.emotion_classifier = emotion_classifier
        self.emotion_cause_recogniser = emotion_cause_recogniser
        self.response_generator = response_generator
        self.sentiment_label_list = []
        self.emotion_label_list = []
        self.emocause_label_list = []
        self.emotion_cause_dictionary = dict.fromkeys(EMOTION_CAUSES, False)
        self.protocal_list = []

    def process_and_respond(self, utterance: str) -> str:
        self._process_sentiment(utterance)
        self._process_emotion(utterance)
        self._process_emotion_cause(utterance)
        response = self.response_generator.infer(utterance)
        self.turn += 1
        return response

    def _process_sentiment(self, utterance: str) -> None:
        sentiment_result = self.sentiment_classifier.classify_utterance(
            utterance
        )
        score_pos = sentiment_result[0][0]["score"]
        score_neg = sentiment_result[0][0]["score"]
        if score_pos > score_neg:
            self.sentiment_label_list.append("positive")
        else:
            self.sentiment_label_list.append("negative")
        self.logger.info(
            f"Current sentiment labels: {self.sentiment_label_list}")

    def _process_emotion(self, utterance: str) -> None:
        emotion_result = self.emotion_classifier.classify_utterance(utterance)
        highest_score = 0
        emotion_label = ""
        for r in emotion_result[0]:
            if r["score"] > highest_score:
                highest_score = r["score"]
                emotion_label = r["label"]
        self.emotion_label_list.append(emotion_label)
        self.logger.info(f"Current emotion labels: {self.emotion_label_list}")

    def _process_emotion_cause(self, utterance: str) -> None:
        if self.emotion_label_list[-1] != "neutral" and self.emotion_label_list[-1] != "joy":
            emotion_cause_result = self.emotion_cause_recogniser.classify_utterance(
                utterance
            )
            highest_score = 0
            emotion_cause_label = ""
            for r in emotion_cause_result[0]:
                if r["score"] > highest_score:
                    highest_score = r["score"]
                    emotion_cause_label = r["label"]
            if highest_score > 0.4:
                self.emocause_label_list.append(emotion_cause_label)
                self.logger.info(
                    f"Current emocause labels: {self.emocause_label_list}"
                )
                if emotion_cause_label in self.emotion_cause_dictionary:
                    self.emotion_cause_dictionary[emotion_cause_label] = True
        else:
            self.emocause_label_list.append("")
            self.logger.info(
                f"Current emocause labels: {self.emocause_label_list}"
            )

# "abuse",
#     "assessment_pressure",
#     "break-up",
#     "change_in_life",
#     "disloyalty",
#     "embarrassed",
#     "family_death",
#     "family_health",
#     "financial_pressure",
#     "friend_death",
#     "health",
#     "injustice",
#     "irritated",
#     "jealousy",
#     "loneliness",
#     "missed_expectation",
#     "owe",
#     "pet_loss",
#     "societal_relationship",
#     "stress",
#     "trauma",
#     "work_pressure"
