from retico_core import abstract
from iu import SimpleTextIU
from iu import BERTEmbeddingIU

#from transformers import BertTokenizer, BertModel
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import pipeline

class BERTEmbeddingModule(abstract.AbstractModule):

    @staticmethod
    def name():
        return "BERTEmbeddingModule"

    @staticmethod
    def description():
        return "A Module that creates a list of embeddings of size 768 by passing words through BERT"

    @staticmethod
    def input_ius():
        return [SimpleTextIU]

    @staticmethod
    def output_iu():
        return BERTEmbeddingIU

    def __init__(self, **kwargs):
        """Initializes the BERT Embedding Module."""
        super().__init__(**kwargs)
        self.embeddings = []
        #model_string = "bert-base-uncased"
        model_string = "distilbert-base-uncased"
        #self.tokenizer = BertTokenizer.from_pretrained(model_string)
        #self.model = BertModel.from_pretrained(model_string)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_string)
        self.model = DistilBertModel.from_pretrained(model_string)

    def process_update(self, update_message):
        for iu, um in update_message:
            if um == abstract.UpdateType.ADD:
                self.process_iu(iu)
            elif um == abstract.UpdateType.REVOKE:
                self.process_revoke(iu)

    def process_iu(self, input_iu):
        encoded_input = self.tokenizer(input_iu.get_text(), return_tensors='pt')
        output = self.model(**encoded_input)
        embedding = output['last_hidden_state'].squeeze(dim=0).mean(dim=0)

        # create iu from embedding
        self.embeddings.append(embedding)
        new_iu = self.create_iu(input_iu)
        new_iu.creator = BERTEmbeddingModule
        new_iu.payload = embedding

        update_iu = abstract.UpdateMessage.from_iu(new_iu, abstract.UpdateType.ADD)
        self.append(update_iu)  # pass iu to next module

    def process_revoke(self, input_iu):
        self.embeddings.pop()

    def start(self):
        print('BERTEmbeddingModule started!')
