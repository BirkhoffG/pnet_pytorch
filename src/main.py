from .tp_data_reader import get_dataset
from .vocabulary import Vocabulary
from .example import Example
from .models.attacker import *
from .dataset import PrDataset

from collections import defaultdict
import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import numpy as np


def extract_vocabulary(dataset, add_symbols=None):
    freqs = defaultdict(int)
    for example in dataset:
        s = example.get_sentence()
        for token in s:
            freqs[token] += 1
    if add_symbols is not None:
        for s in add_symbols:
            freqs[s] += 1000
    return Vocabulary(freqs)


def get_classifier_labels(dataset):
    return set([data.get_label() for data in dataset])


def get_aux_labels(examples):
    labels = set()
    for ex in examples:
        for l in ex.get_aux_labels():
            labels.add(l)
    return labels


class PrModel:
    def __init__(self, args, vocabulary: Vocabulary, classifier_output_size: int, adversary_output_size: int) -> None:
        self.args = args

        self.vocabulary = vocabulary

        # classifier
        self.main_classifier = MainClassifier(
            alphabet_size=vocabulary.size_chars(), vocab_size=vocabulary.size_words(), 
            output_size=classifier_output_size, args=args
            )
        self.adversary_classifier = AdversaryClassifier(
            self.main_classifier.hidden_size, 
            output_size=adversary_output_size, args=args
            )
    
    def get_input(self, example: Example, adversarial=False):
        return self.vocabulary.code_sentence_cw(example.get_sentence(), adversarial=adversarial)

    def evaluate_main(self, dataset):
        self.main_classifier.eval()

        loss = 0
        acc = 0
        tot = len(dataset)
        data_loader = DataLoader(dataset, batch_size=args.batch_size)

        for i, (input_vec, target) in enumerate(data_loader):
            l, p = self.main_classifier.get_loss_prediction(input_vec, target)
            loss += l.item()
            acc += sum(p == target).item()
        return loss / tot, acc / tot * 100


    def train_main(self, train, dev):
        lr = self.args.learning_rate
        batch_size = self.args.batch_size

        train_dataset = PrDataset(train, self.vocabulary, args)
        val_dataset = PrDataset(dev, self.vocabulary, args)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.AdamW(self.main_classifier.parameters(), lr=lr)

        # epoch 0
        l, acc = self.evaluate_main(val_dataset)
        print(f"[epoch=0] loss: {l}, acc: {acc}%")

        for i in range(self.args.iterations):    

            # train_dataset.shuffle()
            self.main_classifier.train()

            for _i, (input_vec, target) in enumerate(tqdm(train_loader)):
                # sys.stderr.write("\r{}%".format(_i / len(train_dataset) * 100))
                
                loss = self.main_classifier.get_loss(input_vec, target)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()

                # if self.args.ptraining:
                #     self.privacy_train(example, train)
                
                # if self.args.atraining:
                #     discriminator_loss += self.discriminator_train(example)
                
                # if self.args.generator:
                #     generator_loss += self.generator_train(example)
            
            l, acc = self.evaluate_main(val_dataset)
            print(f"[epoch={i+1}] loss: {l}, acc: {acc}%")
            
        
    def train_adversarial(self, train, dev):
        pass

    def evaluate_influence_sample(self, train):
        pass


def main(args):

    torch.manual_seed(0)
    get_data = {"tp_fr": lambda : get_dataset("fr"),
                "tp_de": lambda : get_dataset("de"),
                "tp_dk": lambda : get_dataset("dk"),
                "tp_us": lambda : get_dataset("us"),
                "tp_uk": lambda : get_dataset("uk")
                }

    print("loading data...")
    train, dev, test = get_data["tp_fr"]()

    print("building vocabulary...")
    symbols = ["<g={}>".format(i) for i in ["F", "M"]] + ["<a={}>".format(i) for i in ["U", "O"]]
    vocabulary = extract_vocabulary(train, add_symbols=symbols)

    # output size
    classifier_output_size: int = len(get_classifier_labels(train))
    adversary_output_size: int = len(get_aux_labels(train))

    mod = PrModel(args, vocabulary, classifier_output_size, adversary_output_size)
    
    mod.train_main(train, dev)


if __name__ == "__main__":
    import argparse
    import random
    import numpy as np
    import os
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    
    usage = """Implements the privacy evaluation protocol described in the article.

(i) Trains a classifier to predict text labels (topic, sentiment)
(ii) Generate a dataset with the hidden
  representations of each text {r(x), z} with:
    * z: binary private variables
    * x: text
    * r(x): vector representation of text
(iii) Trains the attacker to predict z from x and evaluates privacy
"""
    
    parser = argparse.ArgumentParser(description = usage, formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.1)
    parser.add_argument("--batch-size", "-b", type=int, default=256)
    parser.add_argument("--iterations", "-i", type=int, default=20, help="Number of training iterations")
    

    # define model parameters
    parser.add_argument("--char-embed-dim","-c", type=int, default=50, help="Dimension of char embeddings")
    parser.add_argument("--char-hidden-dim","-C", type=int, default=50, help="Dimension of char lstm")
    parser.add_argument("--word-embed-dim","-w", type=int, default=50, help="Dimension of word embeddings")
    parser.add_argument("--word-hidden-dim","-W", type=int, default=50, help="Dimension of word lstm")

    parser.add_argument("--fc-dim","-l", type=int, default=50, help="Dimension of hidden layers")

    parser.add_argument("--seq_len", default=80, help="Sequence length")
    
    args = parser.parse_args()

    main(args)

