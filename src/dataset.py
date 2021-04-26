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
    def __init__(self, examples: List[Example], voc: Vocabulary, seq_len: int, aux_size: int = 2, return_aux: bool = True) -> None:
        super().__init__()
        self.vocabulary = voc
        self.dataset = examples
        self.seq_len = seq_len
        self.aux_size = aux_size
        self.return_aux = return_aux

    def _get_input(self, sentence):
        return self.vocabulary.code_sentence_cw(sentence)

    def __getitem__(self, index):
        # TODO: padding the training sequences
        example = self.dataset[index]
        sentence = example.get_sentence()
        sentence = sentence[:self.seq_len]
        sentence = sentence + ['<PAD>' for _ in range(0, self.seq_len-len(sentence))]
        input_vec = self.vocabulary.code_sentence_w(sentence)
        # label
        target = example.get_label()
        if self.return_aux:
            # aux 
            aux = list(example.get_aux_labels())
            aux = [1 if i in aux else 0 for i in range(self.aux_size)]
            return torch.tensor(input_vec), torch.tensor(aux), torch.tensor([target])
        else:
            return torch.tensor(input_vec), torch.tensor([target])
    
    def __len__(self):
        return len(self.dataset)

    def shuffle(self):
        self.dataset = shuffle(self.dataset)

class AttackDataset(Dataset):
    def __init__(self, examples: List[Example], voc: Vocabulary, seq_len, output_size) -> None:
        super().__init__()
        self.vocabulary = voc
        self.dataset = examples
        self.seq_len = seq_len
        self.output_size = output_size
    def _get_input(self, sentence):
        return self.vocabulary.code_sentence_cw(sentence)

    def __getitem__(self, index):
        # TODO: padding the training sequences
        example = self.dataset[index]
        sentence = example.get_sentence()
        sentence = sentence[:self.seq_len]
        sentence = sentence + ['<PAD>' for _ in range(0, self.seq_len-len(sentence))]
        input_vec = self.vocabulary.code_sentence_w(sentence)
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

    
