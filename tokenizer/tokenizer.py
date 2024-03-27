import os
import sentencepiece as sp

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'sp_tokenizer_model.model')

class MyTokenizer():
    def __init__(self):
        """
            Given a pretrained tokenizer, import and use in workflow
        """
        self.model = sp.SentencePieceProcessor()
        self.model.load(MODEL_PATH)

    def encode_as_ids(self, text):
        """ Returns the tokenized text as a list of token IDs """
        return self.model.encode_as_ids(text)
    

    def encode_as_pieces(self, text):
        """ Returns the tokenized text as a list of what the tokens are (as text) """
        return self.model.encode_as_pieces(text)

    def decode(self, tokens):
        """ Provide list of tokens, returns decoded text """
        return self.model.decode(tokens)