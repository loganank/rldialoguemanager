from retico_core import abstract
from retico_core.text import TextIU


class RLDialogueManagerModule(abstract.AbstractModule):

    @staticmethod
    def name():
        return "RLDialogueManagerModule"

    @staticmethod
    def description():
        return "A Module that produces determines dialogue actions with a reinforcement learning model"

    @staticmethod
    def input_ius():
        return [TextIU]

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(self, **kwargs):
        """Initializes the Reinforcement Learning Dialogue Manager Module."""
        super().__init__(**kwargs)
        # self.env = DialogueManagerEnv()

    def process_update(self,update_message):
        for iu,um in update_message:
            print(um)
            if um == abstract.UpdateType.ADD:
                self.process_iu(iu)
            elif um == abstract.UpdateType.REVOKE:
                self.process_revoke(iu)

    def process_iu(self, input_iu):
        # do model stuff
        new_iu = self.create_iu(input_iu)
        new_iu.payload = input_iu.get_text()
        print(new_iu.payload)
        update_iu = abstract.UpdateMessage.from_iu(new_iu, abstract.UpdateType.ADD)
        self.append(update_iu)  # pass iu to next module

    def revoke(self, input_iu):
        self.words.pop()

    def start(self):
        print('RLDialogueManagerModule started!')
        # create reinforcement learning model