
import nltk.tokenize as tokenizer


class Example:
    def __init__(self, sentence, label, metadata = None):
        self.sentence = sentence
        self.label = label
        
        self.p_sentence = tokenizer.word_tokenize(sentence)
        
        self.metadata = metadata
    
    def get_label(self):
        return self.label
    
    def get_sentence(self):
        return self.p_sentence
    
    def get_aux_labels(self):
        return self.metadata
    
    def get_training_example(self):
        return self.p_sentence, self.label
    
    def get_aux_training(self):
        return self.p_sentence, self.metadata

    def __str__(self) -> str:
        example = dict(
            sentence=self.sentence,
            label=self.label,
            metadata=self.metadata
        )
        return str(example)





