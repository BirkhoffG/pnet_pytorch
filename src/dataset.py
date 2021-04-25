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
        self.seq_len = args.seq_len

    def _get_input(self, sentence):
        return self.vocabulary.code_sentence_cw(sentence)
    
    def __getitem__(self, index):
        # TODO: padding the training sequences
        example = self.dataset[index]
        sentence = example.get_sentence()
        sentence = sentence[:self.seq_len]
        sentence = sentence + ['PAD' for _ in range(len(sentence), self.seq_len)]
        input_vec = self.vocabulary.code_sentence_w(sentence)
        target = example.get_label()
        return torch.tensor(input_vec), torch.tensor([target])
    
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

    
