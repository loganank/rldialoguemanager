from retico_core import abstract
from retico_core.text import TextIU
from iu import EmotionsIU
import torch

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
        return EmotionsIU

    def __init__(self, **kwargs):
        """Initializes the Emotion Recognition Module."""
        super().__init__(**kwargs)
        self.emotions = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    def process_update(self, update_message):
        for iu, um in update_message:
            if um == abstract.UpdateType.ADD:
                self.process_iu(iu)
            elif um == abstract.UpdateType.REVOKE:
                self.process_revoke(iu)

    def process_iu(self, input_iu):
        # run text through model to find emotions
        model_output = self.classifier([input_iu.get_text()])[0]

        # Only keep values
        emotion_values = torch.tensor([emotion['score'] for emotion in model_output], device=self.device)

        # create iu
        self.emotions.append(emotion_values)
        new_iu = self.create_iu(input_iu)
        new_iu.payload = emotion_values

        update_iu = abstract.UpdateMessage.from_iu(new_iu, abstract.UpdateType.ADD)
        self.append(update_iu)  # pass iu to next module

    def process_revoke(self, input_iu):
        self.emotions.pop()

    def start(self):
        print('EmotionRecognitionModule started!')
