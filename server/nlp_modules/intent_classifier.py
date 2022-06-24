from logging import getLogger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

INTENTS = [
    "maybe",
    "no",
    "death",
    "self-blame",
    "injustice",
    "abuse",
    "missing",
    "work",
    "jealousy",
    "partner",
    "loneliness",
    "trauma",
    "goodbye",
    "health",
    "assessment",
    "greeting",
    "yes"
]
INTENT2ID = {emotion_cause: idx
             for idx, emotion_cause
             in enumerate(INTENTS)}
ID2INTENT = {val: key for key, val in INTENT2ID.items()}

NUM_LABELS = len(INTENTS)


class IntentClassifier():
    def __init__(self,
                 model_type: str,
                 model_checkpoint: str) -> None:
        self.logger = getLogger(f"Intent Classifier")
        self.logger.info("Setting up the intent recogniser...")
        tokenizer = AutoTokenizer.from_pretrained(model_type,
                                                  use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                                   num_labels=NUM_LABELS,
                                                                   id2label=ID2INTENT,
                                                                   label2id=INTENT2ID)
        self.pipe = TextClassificationPipeline(tokenizer=tokenizer,
                                               model=model,
                                               return_all_scores=True)
        self.logger.info("The emotion cause recogniser initialised.")

    def classify_utterance(self, utterance: str) -> list:
        result = self.pipe(utterance)
        self.logger.info(f"Emotion cause recogniser result: {result}.")
        return result


if __name__ == "__main__":
    e = IntentClassifier(model_type="facebook/muppet-roberta-base",
                         model_checkpoint="results/intent_classification_outputs/facebookmuppet-roberta-base")
    res = e.classify_utterance("I regret that I have't done better.")
    highest_score = 0
    emotion_cause_label = ""
    for r in res[0]:
        if r["score"] > highest_score:
            highest_score = r["score"]
            emotion_cause_label = r["label"]
    print(res)
    print(highest_score)
    print(emotion_cause_label)
