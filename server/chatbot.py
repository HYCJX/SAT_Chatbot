class SATChatBot():
    def __init__(self):
        # Emotion Classification Models; the Name Implicates the Number of Emotions Recognisable:
        self.emotion_model_3 = None
        self.emotion_model_7 = None
        self.emotion_model_32 = None
        # Emotion Cause Extraction Models; Span or Classification:
        self.emocause_model_cl = None
        self.emocause_model_span = None
        # Response Generation Models:
        self.resGen_model = None
        self.resGen_model_emo = None
        self.resGen_model_all = None

    def analyse_user_utterance(self):
        return ""
