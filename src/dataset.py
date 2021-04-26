import torch
from torch.utils.data import Dataset
from typing import List
from .example import Example
from .vocabulary import Vocabulary
from sklearn.utils import shuffle


# class PrDataset:
#     def __init__(self, examples: List[Example], voc: Vocabulary) -> None:
#         super().__init__()
#         self.vocabulary = voc
#         self.dataset = examples

#     def _get_input(self, sentence):
#         return self.vocabulary.code_sentence_cw(sentence)

#     def __getitem__(self, index):
#         example = self.dataset[index]
#         input_vec = self._get_input(example.get_sentence())
#         target = example.get_label()
#         return input_vec, target
    
#     def __len__(self):
#         return len(self.dataset)

#     def shuffle(self):
#         self.dataset = shuffle(self.dataset)

class PrDataset(Dataset):
    def __init__(self, examples: List[Example], voc: Vocabulary, args) -> None:
        super().__init__()
        self.vocabulary = voc
        self.dataset = examples
        self.use_char_lstm = args.use_char_lstm
        if args.use_char_lstm:
            self.seq_len = args.char_seq_len 
        else:
            self.seq_len = args.seq_len
#         print('self.seq_len',self.seq_len)
    def _get_input(self, sentence):
        return self.vocabulary.code_sentence_cw(sentence)

    def __getitem__(self, index):
        # TODO: padding the training sequences
        example = self.dataset[index]
        sentence = example.get_sentence()
        if self.use_char_lstm:
            input_vec = self.vocabulary.code_sentence_c(sentence)
        else:
            input_vec = self.vocabulary.code_sentence_w(sentence)
        input_vec = input_vec[:self.seq_len]
        input_vec = input_vec + [0 for _ in range(0, self.seq_len-len(input_vec))]
        target = example.get_label()
        return torch.tensor(input_vec), torch.tensor([target])
    
    def __len__(self):
        return len(self.dataset)

    def shuffle(self):
        self.dataset = shuffle(self.dataset)

class AttackDataset(Dataset):
    def __init__(self, examples: List[Example], voc: Vocabulary, args, output_size) -> None:
        super().__init__()
        self.vocabulary = voc
        self.dataset = examples
        self.use_char_lstm = args.use_char_lstm
        if args.use_char_lstm:
            self.seq_len = args.char_seq_len 
        else:
            self.seq_len = args.seq_len
            
        self.output_size = output_size
    def _get_input(self, sentence):
        return self.vocabulary.code_sentence_cw(sentence)

    def __getitem__(self, index):
        # TODO: padding the training sequences
        example = self.dataset[index]
        sentence = example.get_sentence()
        if self.use_char_lstm:
            input_vec = self.vocabulary.code_sentence_c(sentence)
        else:
            input_vec = self.vocabulary.code_sentence_w(sentence)
        input_vec = input_vec[:self.seq_len]
        input_vec = input_vec + [0 for _ in range(0, self.seq_len-len(input_vec))]
        
        target = list(example.get_aux_labels())
#         print('target',target)
        target = [1 if i in target else 0 for i in range(self.output_size)]
        return torch.tensor(input_vec), torch.tensor(target)
    
    def __len__(self):
        return len(self.dataset)

    def shuffle(self):
        self.dataset = shuffle(self.dataset)

# class PrDataLoader:
#     def __init__(self, dataset: PrDataset, batch_size=1, shuffle=True):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.max = len(dataset) / batch_size

#     def __iter__(self):
#         if self.shuffle:
#             self.dataset.shuffle()
#         self.n = 0
#         return self

#     def __next__(self):
#         if self.n <= self.max:
#             batch = self.dataset[self.batch_size * self.n: self.batch_size * (self.n+1)]
#             self.n += 1
#             return batch
#         else:
#             raise StopIteration

    
