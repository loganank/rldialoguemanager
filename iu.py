from retico_core import abstract


class SimpleTextIU(abstract.IncrementalUnit):
    """ Simple IU for any amount of text"""

    @staticmethod
    def type():
        return "SimpleTextIU"

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None, text=None,
                 **kwargs):
        super().__init__(creator=creator, iuid=iuid, previous_iu=previous_iu,
                         grounded_in=grounded_in, payload=text)
        self.text = text

    def get_text(self):
        return self.text

    def __repr__(self):
        return "%s - (%s): %s" % (
            self.type(),
            self.creator,
            str(self.payload),
        )


class EmotionsIU(abstract.IncrementalUnit):
    """ An IU That contains a list of emotions following the format of this example:
        [{'label': 'neutral', 'score': 0.94}, {'label': 'approval', 'score': 0.04}, ...etc.]
    """

    @staticmethod
    def type():
        return "EmotionsIU"

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None, emotions=None,
                 **kwargs):
        super().__init__(creator=creator, iuid=iuid, previous_iu=previous_iu,
                         grounded_in=grounded_in, payload=emotions)
        self.emotions = emotions

    def get_emotions(self):
        return self.emotions


class BERTEmbeddingIU(abstract.IncrementalUnit):
    """ An IU That contains a list of embeddings created by BERT from words:"""

    @staticmethod
    def type():
        return "BERTEmbeddingIU"

    def __init__(self, creator=None, iuid=0, previous_iu=None, grounded_in=None, embeddings=None,
                 **kwargs):
        super().__init__(creator=creator, iuid=iuid, previous_iu=previous_iu,
                         grounded_in=grounded_in, payload=embeddings)
        self.embeddings = embeddings

    def get_embeddings(self):
        return self.embeddings
