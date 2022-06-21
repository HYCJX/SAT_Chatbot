from asyncio import protocols
from logging import getLogger

from nlp_modules.emotion_classifier import EmotionClassifier, SentimentClassifer
from nlp_modules.response_generator import ResponseGenerator
from nlp_modules.intent_classifier import IntentClassifier


INTENTS = [
    'yes',
    'no',
    'maybe',
    'greeting',
    'goodbye',
    'are_you_a_bot',
    'how_old_are_you',
    'what_are_your_hobbies',
    'what_is_your_name',
    'abuse',
    'assessment',
    'death',
    'financial_pressure',
    'health',
    'injustice',
    'jealousy',
    'loneliness',
    'owe',
    'partner',
    'trauma',
    'work',
]

QUESTIONS = [
    "Was this a recent event?",
    "Have you strongly felt or expressed any of the following emotions towards someone: envy, jealousy, greed, hatred, mistrust, malevolence, or revengefulness?",
    "Do you believe that you should be the saviour of someone else?",
    "Do you see yourself as the victim, blaming someone else for how negative you feel?",
    "Do you feel that you are trying to control someone?",
    "Are you always blaming and accusing yourself for when something goes wrong?",
    "Is it possible that in previous conversations you may not have always considered other viewpoints presented?",
    "Are you undergoing a personal crisis(experiencing difficulties with loved ones e.g. falling out with friends)?",
    "Thank you for your conversations. May I ask you a few more questions?"
]


class SATChatBot():
    def __init__(self,
                 intent_classifier: IntentClassifier,
                 sentiment_classifier: SentimentClassifer,
                 emotion_classifier: EmotionClassifier,
                 response_generator: ResponseGenerator) -> None:
        self.logger = getLogger("SAT Chatbot")
        self.turn = 0
        self.sentiment_classifier = sentiment_classifier
        self.emotion_classifier = emotion_classifier
        self.intent_classifier = intent_classifier
        self.response_generator = response_generator
        self.sentiment_label_list = []
        self.emotion_label_list = []
        self.intent_label_list = []
        self.emotion_cause_answers = [0] * 9
        self.question_answered = [False] * 9
        self.protocal_list = []
        self.current_question_id = -1
        self.can_ask_questions = 0

    def process_and_respond(self, utterance: str) -> str:
        self._process_sentiment(utterance)
        self._process_emotion(utterance)
        intent = self._process_intent(utterance)
        response = self._generate_response(utterance, intent)
        self.turn += 1
        return response

    def _process_sentiment(self, utterance: str) -> None:
        sentiment_result = self.sentiment_classifier.classify_utterance(
            utterance
        )
        score_pos = sentiment_result[0][0]["score"]
        score_neg = sentiment_result[0][1]["score"]
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

    def _process_intent(self, utterance: str) -> str:
        intent_result = self.intent_classifier.classify_utterance(
            utterance
        )
        highest_score = 0
        intent_label = ""
        for r in intent_result[0]:
            if r["score"] > highest_score:
                highest_score = r["score"]
                intent_label = r["label"]
        self.intent_label_list.append(intent_label)
        self.logger.info(
            f"Current intent labels: {self.intent_label_list}")
        return intent_label

    def _generate_response(self, utterance: str, intent: str) -> str:
        if self.turn < 7:
            if intent == "greeting":
                response = self.response_generator.infer(utterance)
                response += " Do you have any concerns?"
                return response
            if intent == "yes":
                if self.current_question_id != -1:
                    self.emotion_cause_answers[self.current_question_id] = 1
                    self.question_answered[self.current_question_id] = True
                    self.current_question_id = -1
                    response = self.response_generator.infer("")
                    return "Thanks. " + response
                response = self.response_generator.infer(utterance)
                return response
            if intent == "no":
                if self.current_question_id != -1:
                    self.emotion_cause_answers[self.current_question_id] = -1
                    self.current_question_id = -1
                    self.question_answered[self.current_question_id] = True
                    response = self.response_generator.infer("")
                    return "Thanks. " + response
                response = self.response_generator.infer(utterance)
                return response
            if intent == "maybe":
                response = self.response_generator.infer(utterance)
                return response
            response = ""
            if intent == 'abuse' or intent == "assessment" or intent == "death" or intent == "financial_pressure" or intent == "health" or intent == "injustice" or intent == "jealousy" or intent == 'loneliness' or intent == 'owe' or intent == 'partner' or intent == 'trauma' or intent == 'work':
                if self.sentiment_label_list[-1] != "positive" and self.emotion_label_list[-1] != "joy":
                    if intent == "abuse":
                        self.emotion_cause_answers[3] = 1
                        self.question_answered[3] = True
                    elif intent == "injustice":
                        self.emotion_cause_answers[3] = 1
                        self.question_answered[3] = True
                    elif intent == "owe":
                        self.emotion_cause_answers[5] = 1
                        self.question_answered[5] = True
                    elif intent == "partner":
                        if self.emotion_label_list[-1] != "neutral":
                            self.emotion_cause_answers[7] = 1
                            self.question_answered[7] = True
                    elif intent == "jealousy" or intent == "loneliness" or intent == "trauma":
                        if self.sentiment_label_list[-1] != "positive" and self.emotion_label_list[-1] != "joy":
                            self.emotion_cause_answers[1] = 1
                            self.question_answered[1] = True
            response_gen = self.response_generator.infer(utterance)
            response_gen += response
            return response_gen
        else:
            if intent == "yes":
                if self.current_question_id != -1:
                    self.emotion_cause_answers[self.current_question_id] = 1
                    self.question_answered[self.current_question_id] = True
                    self.current_question_id = -1
            elif intent == "no":
                if self.current_question_id != -1:
                    self.emotion_cause_answers[self.current_question_id] = -1
                    self.question_answered[self.current_question_id] = True
                    self.current_question_id = -1
            else:
                if self.current_question_id != -1:
                    self.question_answered[self.current_question_id] = True
                    self.current_question_id = -1
            if self.question_answered[8] == False:
                self.current_question_id = 8
                return QUESTIONS[8]
            if self.emotion_cause_answers[8] == -1:
                return self.generate_protocol_list()
            for id in range(1, 8):
                answered = self.question_answered[id]
                if not answered:
                    self.current_question_id = id
                    return QUESTIONS[id]
            response = self.generate_protocol_list()
            return response

    def generate_protocol_list(self) -> str:
        response = "We recommend protocols "
        protocols = set([])
        if self.emotion_cause_answers[1] == 1:
            protocols.add(13)
            protocols.add(14)
        if self.emotion_cause_answers[1] == -1:
            protocols.add(13)
        if self.emotion_cause_answers[2] == 1:
            protocols.add(8)
            protocols.add(15)
            protocols.add(16)
            protocols.add(19)
        if self.emotion_cause_answers[2] == -1:
            protocols.add(13)
        if self.emotion_cause_answers[3] == 1:
            protocols.add(8)
            protocols.add(15)
            protocols.add(16)
            protocols.add(19)
        if self.emotion_cause_answers[3] == -1:
            protocols.add(13)
        if self.emotion_cause_answers[4] == 1:
            protocols.add(8)
            protocols.add(15)
            protocols.add(16)
            protocols.add(19)
        if self.emotion_cause_answers[4] == -1:
            protocols.add(13)
        if self.emotion_cause_answers[5] == -1:
            protocols.add(8)
            protocols.add(15)
            protocols.add(16)
            protocols.add(19)
        if self.emotion_cause_answers[5] == -1:
            protocols.add(13)
        if self.emotion_cause_answers[6] == 1:
            protocols.add(13)
            protocols.add(19)
        if self.emotion_cause_answers[6] == -1:
            protocols.add(13)
        if self.emotion_cause_answers[7] == 1:
            protocols.add(13)
            protocols.add(19)
        if self.emotion_cause_answers[7] == -1:
            protocols.add(13)
        response += str(protocols)
        return response
