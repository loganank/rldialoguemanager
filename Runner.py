from retico_core import IncrementalQueue

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
        iu = SimpleTextIU(previous_iu=None, iuid=self.next_iuid, text=message)
        print("iuid: ", self.next_iuid)
        self.next_iuid += 1
        self.be.process_iu(iu)
        self.er.process_iu(iu)

    # Function to retrieve the response from the RL Dialogue Manager
    def get_dm_decision(self, message):
        # Process the user's message
        self.process_message(message)

        decision = self.decision_queue.get()  # This might block until response is available
        return decision.get_text()
