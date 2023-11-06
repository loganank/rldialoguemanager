from iu import SimpleTextIU

from BERTEmbeddingModule import BERTEmbeddingModule
from EmotionRecognitionModule import EmotionRecognitionModule
from RLDialogueManagerModule import RLDialogueManagerModule

from retico_core.debug import DebugModule

be = BERTEmbeddingModule()
er = EmotionRecognitionModule()
dm = RLDialogueManagerModule()
# debug = DebugModule(print_payload_only=True)
debug = DebugModule()

be.subscribe(dm)
er.subscribe(dm)
dm.subscribe(debug)
#dm.subscribe(debug)

be.run()
er.run()
dm.run()
debug.run()

user_input = 'This will be interpreted by the reinforcement learning model.'

# increment iuid every time so it's unique
iu = SimpleTextIU(previous_iu=None, iuid=1, text=user_input)
be.process_iu(iu)
er.process_iu(iu)

input()

be.stop()
er.stop()
dm.stop()
debug.stop()
