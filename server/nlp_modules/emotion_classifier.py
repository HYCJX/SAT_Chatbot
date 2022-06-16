from logging import getLogger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
from typing import Tuple


EMOTION_NUM_LABELS = 7
SENTIMENT_NUM_LABELS = 3


def get_emotion2id() -> Tuple[dict, dict]:
    """
    Get a dict that converts string class to numbers.
    """
    emotions = [
        "neutral",
        "joy",
        "surprise",
        "anger",
        "sadness",
        "disgust",
        "fear",
    ]
    emotion2id = {emotion: idx for idx, emotion in enumerate(emotions)}
    id2emotion = {val: key for key, val in emotion2id.items()}
    return emotion2id, id2emotion


def get_sentiment2id() -> Tuple[dict, dict]:
    """
    Get a dict that converts string class to numbers.
    """
    sentiments = [
        "positive",
        "negative"
    ]
    sentiment2id = {emotion: idx for idx, emotion in enumerate(sentiments)}
    id2sentiment = {val: key for key, val in sentiment2id.items()}
    return sentiment2id, id2sentiment


class EmotionClassifier():
    def __init__(self,
                 model_type: str,
                 model_checkpoint: str) -> None:
        self.logger = getLogger(f"Emotion Classifier")
        tokenizer = AutoTokenizer.from_pretrained(model_type,
                                                  use_fast=True)
        label2id, id2label = get_emotion2id()
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                                   num_labels=EMOTION_NUM_LABELS,
                                                                   id2label=id2label,
                                                                   label2id=label2id)
        self.pipe = TextClassificationPipeline(tokenizer=tokenizer,
                                               model=model,
                                               return_all_scores=True)

    def classify_utterance(self, utterance: str) -> int:
        return self.pipe(utterance)


class SentimentClassifer():
    def __init__(self,
                 model_type: str,
                 model_checkpoint: str) -> None:
        self.logger = getLogger(f"Sentiment Classifier")
        self.logger.info("Setting up the sentiment classifier...")
        tokenizer = AutoTokenizer.from_pretrained(model_type,
                                                  use_fast=True)
        label2id, id2label = get_sentiment2id()
        model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,
                                                                   num_labels=SENTIMENT_NUM_LABELS,
                                                                   id2label=id2label,
                                                                   label2id=label2id)
        self.pipe = TextClassificationPipeline(tokenizer=tokenizer,
                                               model=model,
                                               return_all_scores=True)
        self.label_list = []
        self.score_list = []
        self.logger.info("Sentiment classifier initialised.")

    def classify_utterance(self, utterance: str) -> Tuple[str, float]:
        result = self.pipe(utterance)
        self.label_list.append(result["label"])
        self.score_list.append(result["score"])
        self.logger.info(f"Sentiment analysis results: {result}.")
        return result["label"], result["score"]


if __name__ == "__main__":
    s = SentimentClassifer(model_type="facebook/muppet-roberta-base",
                           model_checkpoint="results/sentiment_analysis_outputs/classification/facebookmuppet-roberta-base")
    utterances = ["I haven't been home for a long time."]
    print(s.classify_utterance(utterances))
