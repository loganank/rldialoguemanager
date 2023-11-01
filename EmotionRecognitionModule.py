from retico_core import abstract
from retico_core.text import TextIU
from iu import SimpleTextIU

from transformers import pipeline


class EmotionRecognitionModule(abstract.AbstractModule):

    @staticmethod
    def name():
        return "EmotionRecognitionModule"

    @staticmethod
    def description():
        return "A Module that produces determines emotions represented in text"

    @staticmethod
    def input_ius():
        return [TextIU]

    @staticmethod
    def output_iu():
        return SimpleTextIU

    def __init__(self, **kwargs):
        """Initializes the Emotion Recognition Module."""
        super().__init__(**kwargs)
        self.emotions = []
        self.classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    def process_update(self,update_message):
        for iu, um in update_message:
            print(um)
            if um == abstract.UpdateType.ADD:
                self.process_iu(iu)
            elif um == abstract.UpdateType.REVOKE:
                self.process_revoke(iu)

    def process_iu(self, input_iu):
        # run text through model to find emotions
        model_output = self.classifier([input_iu.get_text()])[0]
        # Turn the output into a dictionary

        emotion_scores = {}

        for emotion in model_output:
            emotion_scores[emotion['label']] = emotion['score']

        # create iu from those emotion scores
        self.emotions.append(emotion_scores)
        new_iu = self.create_iu(input_iu)
        new_iu.payload = emotion_scores
        print(new_iu.payload)

        update_iu = abstract.UpdateMessage.from_iu(new_iu, abstract.UpdateType.ADD)
        self.append(update_iu)  # pass iu to next module

    def revoke(self, input_iu):
        self.emotions.pop()

    def start(self):
        print('EmotionRecognitionModule started!')
