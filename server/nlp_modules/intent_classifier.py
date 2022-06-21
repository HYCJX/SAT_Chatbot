from logging import getLogger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

INTENTS = [
    'loneliness', 'what_are_your_hobbies', 'are_you_a_bot', 'goodbye', 'how_old_are_you', 'owe', 'injustice', 'what_is_your_name', 'work', 'trauma', 'maybe', 'health', 'assessment', 'no', 'death', 'financial_pressure', 'yes', 'partner', 'abuse', 'jealousy', 'greeting'
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
    e = IntentClassifier("roberta-base", "results/roberta-base")
    res = e.classify_utterance("I regret that I didn't do well.")
    highest_score = 0
    emotion_cause_label = ""
    for r in res[0]:
        if r["score"] > highest_score:
            highest_score = r["score"]
            emotion_cause_label = r["label"]

    print(highest_score)
    print(emotion_cause_label)
