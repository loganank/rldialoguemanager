import os, sys

from iu import SimpleTextIU

from BERTEmbeddingModule import BERTEmbeddingModule
from EmotionRecognitionModule import EmotionRecognitionModule
from RLDialogueManagerModule import RLDialogueManagerModule

from retico_core.debug import DebugModule

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/logan/Projects/CS497/googlespeechtotextkey.json'

be = BERTEmbeddingModule()
er = EmotionRecognitionModule()
#dm = RLDialogueManagerModule()
# debug = DebugModule(print_payload_only=True)
debug = DebugModule()

be.subscribe(debug)
er.subscribe(debug)
#dm.subscribe(debug)

#be.run()
er.run()
#dm.run()
debug.run()

input = 'This will be interpreted by the reinforcement learning model.'

iu = SimpleTextIU(previous_iu=None, iuid=1, text=input)
#be.process_iu(iu)
er.process_iu(iu)
#dm.process_iu(iu)

#input()

#be.stop()
er.stop()
#dm.stop()
debug.stop()
