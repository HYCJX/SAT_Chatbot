from logging import getLogger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

EMOTION_CAUSES = [
    'greeting', 'disloyalty', 'injustice', 'what_are_your_hobbies', 'jealousy', 'no', 'pet_loss', 'owe', 'goodbye', 'break-up', 'assessment_pressure', 'loneliness', 'yes', 'abuse', 'trauma', 'what_is_your_name', 'how_old_are_you', 'family_health', 'maybe', 'are_you_a_bot', 'financial_pressure', 'health', 'family_death', 'work_pressure', 'others'
]
EMOCAUSE2ID = {emotion_cause: idx
               for idx, emotion_cause
               in enumerate(EMOTION_CAUSES)}
ID2EMOCAUSE = {val: key for key, val in EMOCAUSE2ID.items()}

NUM_LABELS = len(EMOTION_CAUSES)


class EmotionCauseRecogniser():
    def __init__(self,
                 model_type: str,
                 model_checkpoint: str) -> None:
        self.logger = getLogger(f"Emotion Classifier")
        self.logger.info("Setting up the emotion cause recogniser...")
        tokenizer = AutoTokenizer.from_pretrained(model_type,
                                                  use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                                   num_labels=NUM_LABELS,
                                                                   id2label=ID2EMOCAUSE,
                                                                   label2id=EMOCAUSE2ID)
        self.pipe = TextClassificationPipeline(tokenizer=tokenizer,
                                               model=model,
                                               return_all_scores=True)
        self.logger.info("The emotion cause recogniser initialised.")

    def classify_utterance(self, utterance: str) -> list:
        result = self.pipe(utterance)
        self.logger.info(f"Emotion cause recogniser result: {result}.")
        return result


if __name__ == "__main__":
    e = EmotionCauseRecogniser("roberta-base", "results/roberta-base")
    res = e.classify_utterance("My wife loves me.")
    highest_score = 0
    emotion_cause_label = ""
    for r in res[0]:
        if r["score"] > highest_score:
            highest_score = r["score"]
            emotion_cause_label = r["label"]

    print(highest_score)
    print(emotion_cause_label)
