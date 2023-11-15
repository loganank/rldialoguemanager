from retico_core import IncrementalQueue, abstract

from iu import SimpleTextIU
from BERTEmbeddingModule import BERTEmbeddingModule
from EmotionRecognitionModule import EmotionRecognitionModule
from RLDialogueManagerModule import RLDialogueManagerModule


class Runner:
    def __init__(self):
        self.decision_queue = IncrementalQueue(RLDialogueManagerModule, None)
        self.be = BERTEmbeddingModule()
        self.er = EmotionRecognitionModule()
        self.dm = RLDialogueManagerModule()

        self.be.subscribe(self.dm)
        self.er.subscribe(self.dm)
        self.dm.subscribe(module=None, q=self.decision_queue)

        self.be.run()
        self.er.run()
        self.dm.run()

        # A variable to store the response
        self.module_response = None

        # variable to make iuids unique
        self.next_iuid = 1

    # This function will be used to pass messages to the module
    def process_message(self, message):
        """ passes user message to retico """

        iu = SimpleTextIU(previous_iu=None, iuid=self.next_iuid, text=message)
        print("iuid: ", self.next_iuid)
        self.next_iuid += 1
        self.be.process_iu(iu)
        self.er.process_iu(iu)

    def process_correct_decision(self, correct_decision):
        """ passes correct decision to retico """

        iu = SimpleTextIU(previous_iu=None, iuid=self.next_iuid, text=correct_decision)
        print("iuid: ", self.next_iuid)
        self.next_iuid += 1
        self.dm.process_iu(iu)

    def get_dm_decision(self, message):
        """ retrieve the response from the RL Dialogue Manager
        after sending user's message """

        # Process the user's message
        self.process_message(message)

        decision = self.decision_queue.get()  # This might block until response is available
        for iu, um in decision:
            if um == abstract.UpdateType.ADD:
                print('add')
            elif um == abstract.UpdateType.REVOKE:
                print('revoke')
        return iu.payload

    def get_dm_response(self, correct_decision):
        """ retrieve the response from the RL Dialogue Manager
        after sending the correct decision """

        # Process the user's message
        self.process_correct_decision(correct_decision)

        response = self.decision_queue.get()  # This might block until response is available
        for iu, um in response:
            if um == abstract.UpdateType.ADD:
                print('add')
            elif um == abstract.UpdateType.REVOKE:
                print('revoke')
        return iu.payload

# runner = Runner()
# print(runner.get_dm_decision("this is a test"))