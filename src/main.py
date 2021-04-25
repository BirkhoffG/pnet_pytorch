from .tp_data_reader import get_dataset
from .vocabulary import Vocabulary
from .example import Example
from .models.attacker import *
from .dataset import PrDataset, AttackDataset

from collections import defaultdict
import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score


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
        self.device = args.device

        self.vocabulary = vocabulary

        # classifier
        self.main_classifier = MainClassifier(
            alphabet_size=vocabulary.size_chars(), vocab_size=vocabulary.size_words(), 
            output_size=classifier_output_size, args=args
            ).to(self.device)
        self.adversary_classifier = AdversaryClassifier(
            self.main_classifier.hidden_size, 
            output_size=adversary_output_size, args=args
            ).to(self.device)
    
    def get_input(self, example: Example, adversarial=False):
        return self.vocabulary.code_sentence_cw(example.get_sentence(), adversarial=adversarial)

    def evaluate_main(self, dataset):
        self.main_classifier.eval()
        device = self.device
        
        loss = 0
        acc = 0
        tot = 0#len(dataset)
        with torch.no_grad():
            for i, (input_vec, target) in enumerate(dataset):
                input_vec = input_vec.to(device)
                target = target.to(device)
                l, predicts = self.main_classifier.get_loss_prediction(input_vec, target)
                loss += l.item()
                for p, t in zip(predicts, target):
                    tot += 1
                    if p.item() == t.item():
                        acc += 1
#                 print(acc, tot)
        return (loss / tot), round(acc / tot  * 100, 3)


    def train_main(self, train, dev):
        lr = self.args.learning_rate
        batch_size = self.args.batch_size
        device = self.device
        
        train_dataset = PrDataset(train, self.vocabulary, self.args.seq_len)
        val_dataset = PrDataset(dev, self.vocabulary, self.args.seq_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        optimizer = optim.AdamW(self.main_classifier.parameters(), lr=lr)
        
        # epoch 0
        l, acc = self.evaluate_main(val_loader)
        print(f"[epoch=0] loss: {l}, acc: {acc}%")

        for i in range(self.args.iterations):
            self.main_classifier.train()
            
            train_acc = 0
            train_tot = 0
            train_loss = 0
            for _i, (input_vec, target) in enumerate(tqdm(train_loader)):
#                 print('input_vec',input_vec.shape)
                # sys.stderr.write("\r{}%".format(_i / len(train_dataset) * 100))
                loss = 0
                input_vec = input_vec.to(device)
                target = target.to(device)
                loss, predicts = self.main_classifier.get_loss_prediction(input_vec, target)
                
                for p, t in zip(predicts, target):
                    train_tot += 1
                    if p.item() == t.item():
                        train_acc += 1
                        
                loss.backward()
                train_loss += loss.item()
                # mimic batchingc
#                 if (_i + 1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

                # if self.args.ptraining:
                #     self.privacy_train(example, train)
                
                # if self.args.atraining:
                #     discriminator_loss += self.discriminator_train(example)
                
                # if self.args.generator:
                #     generator_loss += self.generator_train(example)
            train_acc = round(train_acc / train_tot * 100, 3)
            print(f"[epoch={i+1}] train loss: {train_loss}, train acc: {train_acc}")
            l, acc = self.evaluate_main(val_loader)
            print(f"[epoch={i+1}] val loss: {l}, val acc: {acc}")
            
 
    def evaluate_adversarial(self, dataset):
        self.adversary_classifier.eval()
#         self.main_classifier.eval()
        device = self.device
        loss = 0
        gender_acc = 0
        age_acc = 0
        tot = 0#len(dataset)
        with torch.no_grad():
            for i, (input_vec, target) in enumerate(dataset):
                input_vec = input_vec.to(device)
                target = target.to(device)
                hidden_state = self.main_classifier.get_lstm_embed(input_vec)
                l, predicts = self.adversary_classifier.get_loss_prediction(hidden_state, target)
                loss += l.item()
                for p, t in zip(predicts, target):
                    tot += 1
                    if p[0].item() == t[0].item():
                        gender_acc += 1
                    if p[1].item() == t[1].item():
                        age_acc += 1
        return loss / tot, round(gender_acc / tot * 100, 3) , round(age_acc / tot * 100, 3) 

    def train_adversarial(self, train, dev):
        lr = self.args.learning_rate
        batch_size = self.args.batch_size
        device = self.device
        output_size =  self.adversary_classifier.output_size
        seq_len = self.args.seq_len
        
        train_dataset = AttackDataset(train, self.vocabulary, seq_len, output_size)
        val_dataset = AttackDataset(dev, self.vocabulary, seq_len, output_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        optimizer = optim.AdamW(self.adversary_classifier.parameters(), lr=lr)

        # epoch 0
        l, gender_acc, age_acc = self.evaluate_adversarial(val_loader)
        print(f"[epoch=0] loss: {l}, gender acc: {gender_acc}%, age acc: {age_acc}%")
    
        for i in range(self.args.iterations):
            self.main_classifier.train()
            self.adversary_classifier.train()
            #freeze the model
            self.main_classifier.freeze_parameters()

            for _i, (input_vec, target) in enumerate(tqdm(train_loader)):
                input_vec = input_vec.to(device)
                target = target.to(device)
                hidden_state = self.main_classifier.get_lstm_embed(input_vec)
                
                loss = self.adversary_classifier.get_loss(hidden_state, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # if self.args.ptraining:
                #     self.privacy_train(example, train)
                
                # if self.args.atraining:
                #     discriminator_loss += self.discriminator_train(example)
                
                # if self.args.generator:
                #     generator_loss += self.generator_train(example)
            l, gender_acc, age_acc = self.evaluate_adversarial(val_loader)
            print(f"[epoch={i+1}] loss: {l}, gender acc: {gender_acc}%, age acc: {age_acc}%")

    def evaluate_influence_sample(self, train):
        pass


def main(args):

    device = torch.device(f'cuda:{args.device}' if args.device != 'cpu' else 'cpu')
    args.device = device
    torch.manual_seed(0)
    get_data = {"tp_fr": lambda : get_dataset("fr"),
                "tp_de": lambda : get_dataset("de"),
                "tp_dk": lambda : get_dataset("dk"),
                "tp_us": lambda : get_dataset("us"),
                "tp_uk": lambda : get_dataset("uk")
                }

    print("loading data...")
    train, dev, test = get_data[args.dataset]()

    print("building vocabulary...")
    symbols = ["<g={}>".format(i) for i in ["F", "M"]] + ["<a={}>".format(i) for i in ["U", "O"]]
    vocabulary = extract_vocabulary(train, add_symbols=symbols)

    # output size
    classifier_output_size: int = len(get_classifier_labels(train))
    adversary_output_size: int = len(get_aux_labels(train))

    mod = PrModel(args, vocabulary, classifier_output_size, adversary_output_size)
    
    mod.train_main(train, dev)
    mod.train_adversarial(train, dev)
    


if __name__ == "__main__":
    import argparse
    import random
    import numpy as np
    import os
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)
    
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
    
    parser.add_argument("--learning-rate", "-b", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--iterations", "-i", type=int, default=10, help="Number of training iterations")
    
    parser.add_argument("--seq_len", "-sl", type=int, default=75, help="Length of sequence")

    # define model parameters
    parser.add_argument("--char-embed-dim","-c", type=int, default=50, help="Dimension of char embeddings")
    parser.add_argument("--char-hidden-dim","-C", type=int, default=50, help="Dimension of char lstm")
    parser.add_argument("--word-embed-dim","-w", type=int, default=50, help="Dimension of word embeddings")
    parser.add_argument("--word-hidden-dim","-W", type=int, default=50, help="Dimension of word lstm")

    parser.add_argument("--fc-dim","-l", type=int, default=50, help="Dimension of hidden layers")
    
    parser.add_argument("--device", "-device", type=str, default='cpu', help="Length of sequence")
    parser.add_argument("--dataset", '-d', choices=["ag", "dw", "tp_fr", "tp_de", "tp_dk", "tp_us", "tp_uk", "bl"], help="Dataset. tp=trustpilot, bl=blog", required=True)
    
    args = parser.parse_args()

    main(args)

