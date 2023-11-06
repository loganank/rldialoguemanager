from retico_core import abstract

from RLModel import RLModel
from iu import SimpleTextIU, BERTEmbeddingIU, EmotionsIU


class RLDialogueManagerModule(abstract.AbstractModule):

    @staticmethod
    def name():
        return "RLDialogueManagerModule"

    @staticmethod
    def description():
        return "A Module that produces determines dialogue actions with a reinforcement learning model"

    @staticmethod
    def input_ius():
        return [BERTEmbeddingIU, EmotionsIU]

    @staticmethod
    def output_iu():
        return SimpleTextIU

    def __init__(self, **kwargs):
        """Initializes the Reinforcement Learning Dialogue Manager Module."""
        super().__init__(**kwargs)
        self.rl_model = RLModel()
        self.storedIUs = {}

    def process_update(self, update_message):
        for iu, um in update_message:
            print(um)
            if um == abstract.UpdateType.ADD:
                self.process_iu(iu)
            elif um == abstract.UpdateType.REVOKE:
                self.process_revoke(iu)

    def process_iu(self, input_iu):
        # TODO do model stuff
        # self.rl_model.process_message(new_sentence, new_emotions)
        dm_decision = None
        if input_iu.grounded_in.iuid in self.storedIUs:
            # TODO both ius are present, pass to model
            storedIU = self.storedIUs[input_iu.grounded_in.iuid]
            if isinstance(storedIU, BERTEmbeddingIU):
                bert_embedding = storedIU.get_embeddings()  # get bert embedding
                emotions_embedding = input_iu.get_emotions()
            else:
                emotions_embedding = storedIU.get_emotions()  # get emotions
                bert_embedding = input_iu.get_embeddings()
            del self.storedIUs[input_iu.grounded_in.iuid] # remove so dictionary doesn't get excessively large
            dm_decision = self.rl_model.process_message(bert_embedding, emotions_embedding)
        else:
            self.storedIUs[input_iu.grounded_in.iuid] = input_iu
        if dm_decision is not None:
            print(dm_decision)
        new_iu = self.create_iu(input_iu)
        new_iu.payload = input_iu.get_embeddings()
        print(new_iu.payload)
        update_iu = abstract.UpdateMessage.from_iu(new_iu, abstract.UpdateType.ADD)
        self.append(update_iu)  # pass iu to next module

    def process_revoke(self, input_iu):
        self.words.pop()

    def start(self):
        print('RLDialogueManagerModule started!')
        # create reinforcement learning model