from logging import getLogger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

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
EMOCAUSE2ID = {emotion: idx for idx, emotion in enumerate(EMOTION_CAUSES)}
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
    res = e.classify_utterance("My girlfriend break up with me.")
    print(res)
