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

import pytorch_influence_functions as pif
from pyvacy import optim, analysis, sampling
from torch.utils.data import DataLoader, TensorDataset

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
    labels = []
    for data in dataset:
        labels.append(data.get_label())
    for label in list(set(labels)):
        print(f'label ({label}) ratio: ',round(labels.count(label)/len(labels), 3))
    return set(labels)


def get_aux_labels(examples):
    label_1s = []#set()
    label_2s = []
    for ex in examples:
        target = list(ex.get_aux_labels())
        target = [1 if i in target else 0 for i in range(2)]
        label_1s.append(target[0])
        label_2s.append(target[1])
    for label in list(set(label_1s)):
        print(f'gender label ({label}) ratio: ',round(label_1s.count(label)/len(label_1s), 3))
    for label in list(set(label_2s)):
        print(f'age label ({label}) ratio: ',round(label_2s.count(label)/len(label_2s), 3))
    all_labels = label_1s + label_2s
    return set(all_labels)


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
    
    def make_weights_for_balanced_classes(self, datasets):
        labels = []
        for _, target in datasets:
            labels.append(target.item())     
        counts = []
        for label in list(set(labels)):
            counts.append(labels.count(label))
        weight = []
        total = sum(counts)
        for count in counts:
            weight.append(1/count)  
        return weight
    
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
#                 print('predicts',predicts)
                loss += l.item()
                for p, t in zip(predicts, target):
                    tot += 1
                    if p.item() == t.item():
                        acc += 1
#                 print(acc, tot)
        return (loss / tot), round(acc / tot  * 100, 3)

    def train_main(self, train, dev):
        
        l2_norm_clip = 1.0
        noise_multiplier = 1.1
        minibatch_size = self.args.batch_size
        microbatch_size = 1
        iterations = 100#int(len(train_dataset)/self.args.batch_size)
        delta = 1e-5
        
        
        lr = self.args.learning_rate
        batch_size = self.args.batch_size
        device = self.device
        
        train_dataset = PrDataset(train, self.vocabulary, self.args)
        val_dataset = PrDataset(dev, self.vocabulary, self.args)

        weights = self.make_weights_for_balanced_classes(train_dataset)
        weights = torch.DoubleTensor(weights)   
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_dataset))  
        
        if self.args.is_add_gradient_noise:
            optimizer = optim.DPAdam(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                minibatch_size=minibatch_size,
                microbatch_size=microbatch_size,
                params=self.main_classifier.parameters(),
                lr=lr)
            minibatch_loader, microbatch_loader = sampling.get_data_loaders(minibatch_size, microbatch_size, iterations)
            print('Achieves ({}, {})-DP'.format(analysis.epsilon(len(train_dataset), minibatch_size, noise_multiplier,
                    iterations, delta,), delta))
            train_loader = minibatch_loader(train_dataset)
#             val_loader = minibatch_loader(val_dataset)
        else:
            optimizer = optim.Adam(self.main_classifier.parameters(), lr=lr)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, sampler = sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
        

        # epoch 0
        l, acc = self.evaluate_main(val_loader)
        print(f"[epoch=0] loss: {l}, acc: {acc}%")

        best_val_loss = 1000
        best_model = None
        for i in range(self.args.iterations):
            self.main_classifier.train()
            train_loss = 0
            train_acc = 0
            train_tot = 0
            loss = 0
            for _i, (input_vec, target) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                
                if self.args.is_add_gradient_noise:
                    for X_microbatch, y_microbatch  in microbatch_loader(TensorDataset(input_vec, target)):
                        optimizer.zero_microbatch_grad()
                        X_microbatch = X_microbatch.to(device)
                        y_microbatch = y_microbatch.to(device)
                        loss, predicts = self.main_classifier.get_loss_prediction(X_microbatch, y_microbatch)
                        loss.backward()
                        optimizer.microbatch_step()
                        train_loss += loss.item()
                        train_tot += 1
                        if predicts[0].item() == y_microbatch[0].item():
                            train_acc += 1
                    optimizer.step()
                else:
                    
                    input_vec = input_vec.to(device)
                    target = target.to(device)
                    loss, predicts = self.main_classifier.get_loss_prediction(input_vec, target)
#                     loss = loss * 0 + 1
                    train_loss += loss.item()
                    
                    if self.args.is_add_loss_noise:  
                        loss = loss + self.add_loss_noise() #add noise before backward
                    loss.backward()  
                    for p, t in zip(predicts, target):
                        train_tot += 1
                        if predicts[0].item() == target[0].item():
                            train_acc += 1
                    optimizer.step()
            
            # if self.args.ptraining:
            #     self.privacy_train(example, train)

            # if self.args.atraining:
            #     discriminator_loss += self.discriminator_train(example)

            # if self.args.generator:
            #     generator_loss += self.generator_train(example)

            train_acc = round(train_acc / train_tot  * 100, 3)
            train_loss = train_loss / train_tot
            print(f"[train epoch={i+1}] loss: {train_loss}, acc: {train_acc}%")
            l, acc = self.evaluate_main(val_loader)
            print(f"[val epoch={i+1}] loss: {l}, acc: {acc}%")
            
            if l < best_val_loss:
                best_val_loss = l
                best_model = self.main_classifier
                print('[best_model updated]')
        self.main_classifier = best_model
        l, acc = self.evaluate_main(val_loader)
        print(f"[val epoch=final] loss: {l}, acc: {acc}%")
        return best_model

            
 
    def evaluate_adversarial(self, dataset):
        self.adversary_classifier.eval()
        self.main_classifier.eval()
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
#                 print('predicts',predicts)
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
        
        train_dataset = AttackDataset(train, self.vocabulary, self.args, output_size)
        val_dataset = AttackDataset(dev, self.vocabulary, self.args, output_size)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        optimizer = optim.Adam(self.adversary_classifier.parameters(), lr=lr)

        # epoch 0
        l, gender_acc, age_acc = self.evaluate_adversarial(val_loader)
        print(f"[epoch=0] loss: {l}, gender acc: {gender_acc}%, age acc: {age_acc}%")

        best_val_loss = 1000
        best_model = None
        self.main_classifier.eval()
        
        
        for i in range(self.args.iterations):
            self.adversary_classifier.train()
            
            train_loss = 0
            train_gender_acc = 0
            train_age_acc = 0
            train_tot = 0
            loss = 0
            for _i, (input_vec, target) in enumerate(tqdm(train_loader)):
                input_vec = input_vec.to(device)
                target = target.to(device)
                hidden_state = self.main_classifier.get_lstm_embed(input_vec)
                hidden_state = hidden_state.detach()
                loss, predicts = self.adversary_classifier.get_loss_prediction(hidden_state, target)
#                 loss = loss * 0 + 1
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()
                for p, t in zip(predicts, target):
                    train_tot += 1
                    if p[0].item() == t[0].item():
                        train_gender_acc += 1
                    if p[1].item() == t[1].item():
                        train_age_acc += 1
                # if self.args.ptraining:
                #     self.privacy_train(example, train)
                
                # if self.args.atraining:
                #     discriminator_loss += self.discriminator_train(example)
                
                # if self.args.generator:
                #     generator_loss += self.generator_train(example)
            train_gender_acc = round(train_gender_acc / train_tot  * 100, 3)
            train_age_acc = round(train_age_acc / train_tot  * 100, 3)
            train_loss = train_loss / train_tot
            print(f"[train epoch={i+1}] loss: {train_loss}, gender acc: {train_gender_acc}%, age acc: {train_age_acc}%")
            l, gender_acc, age_acc = self.evaluate_adversarial(val_loader)
            print(f"[val epoch={i+1}] loss: {l}, gender acc: {gender_acc}%, age acc: {age_acc}%")
            
            if l < best_val_loss:
                best_val_loss = l
                best_model = self.adversary_classifier
                print('[best_model updated]')
                
        self.adversary_classifier = best_model
        l, gender_acc, age_acc = self.evaluate_adversarial(val_loader)
        print(f"[val epoch=final] loss: {l}, gender acc: {gender_acc}%, age acc: {age_acc}%")
        return best_model

    def evaluate_influence_sample(self, train, test):
        train_dataset = PrDataset(train, self.vocabulary, self.args)
        test_dataset = PrDataset(test, self.vocabulary, self.args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

        config = pif.get_default_config()
        
        config['gpu'] = -1
        config['damp'] = 0.01
        config['scale'] = 1
        self.main_classifier = self.main_classifier.cpu()
        print("config: ", config)
        pif.calc_all_grad_then_test(config, self.main_classifier, train_dataloader, test_dataloader)
        
    def add_gradient_noise(self):
        for p in self.main_classifier.parameters():
            noise = np.random.laplace(loc=0, scale=1, size = p.grad.shape)
            p.grad += torch.from_numpy(noise).to(self.device)
        
    def add_loss_noise(self):
        noise = np.random.laplace(loc=0, scale=1, size = 1)[0]
#         loss += noise
        return noise


def main(args):
    args.device_num = args.device
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
    
    best_main_model = mod.train_main(train, dev)
    best_adv_model = mod.train_adversarial(train, dev)
    if args.is_influence_sample:
        mod.evaluate_influence_sample(train, test)
    


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

    parser.add_argument("dataset", default="tp_fr", choices=["tp_fr", "tp_de", "tp_dk", "tp_us", "tp_uk", "bl"], help="Dataset. tp=trustpilot, bl=blog")
    
    parser.add_argument("--learning-rate", "-b", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--iterations", "-i", type=int, default=15, help="Number of training iterations")
    
    parser.add_argument("--seq_len", "-sl", type=int, default=75, help="Length of word sequence")
    parser.add_argument("--char_seq_len", "-csl", type=int, default=500, help="Length of character sequence")

    # define model parameters
    parser.add_argument("--char-embed-dim","-c", type=int, default=50, help="Dimension of char embeddings")
    parser.add_argument("--char-hidden-dim","-C", type=int, default=50, help="Dimension of char lstm")
    parser.add_argument("--word-embed-dim","-w", type=int, default=50, help="Dimension of word embeddings")
    parser.add_argument("--word-hidden-dim","-W", type=int, default=50, help="Dimension of word lstm")

    parser.add_argument("--fc-dim","-l", type=int, default=50, help="Dimension of hidden layers")
    
    parser.add_argument("--device", '-device', type=str, default='cpu', help="Training device")
    
    parser.add_argument("--is-add-loss-noise", action="store_true", help="Add noise to loss, [default=false]")
    parser.add_argument("--is-add-gradient-noise", action="store_true", help="Add noise to gradient, [default=false]")
    parser.add_argument("--is-influence-sample", action="store_true", help="Evaluate influence, [default=false]")
    parser.add_argument("--use-char-lstm", action="store_true", help="Use a character LSTM, [default=false]")
    
    args = parser.parse_args()

    main(args)

#=====dead kitten======#
#         else:
#             
# #             optimizer = optim.AdamW(self.main_classifier.parameters(), lr=lr)
            
#             # epoch 0
#             l, acc = self.evaluate_main(val_loader)
#             print(f"[epoch=0] loss: {l}, acc: {acc}%")

#             for i in range(self.args.iterations):
#                 self.main_classifier.train()
#                 train_loss = 0
#                 train_acc = 0
#                 train_tot = 0
#                 for _i, (input_vec, target) in enumerate(tqdm(train_loader)):            
#                     input_vec = input_vec.to(device)
#                     target = target.to(device)
#                     loss, predicts = self.main_classifier.get_loss_prediction(input_vec, target)
#                     train_loss += loss.item()

#                     if self.args.is_add_loss_noise:  
#                         loss = loss + self.add_loss_noise() #add noise before backward
                        
#                     loss.backward()  
# #                     if self.args.is_add_gradient_noise:  
# #                         self.add_gradient_noise() #add noise after gradient is calculated. 
#                     for p, t in zip(predicts, target):
#                         train_tot += 1
#                         if predicts[0].item() == target[0].item():
#                             train_acc += 1
#                     optimizer.step()
#                     optimizer.zero_grad()

#                 # if self.args.ptraining:
#                 #     self.privacy_train(example, train)
                
#                 # if self.args.atraining:
#                 #     discriminator_loss += self.discriminator_train(example)
                
#                 # if self.args.generator:
#                 #     generator_loss += self.generator_train(example)
#                 train_acc = round(train_acc / train_tot  * 100, 3)
#                 print(f"[train epoch={i+1}] loss: {train_loss}, acc: {train_acc}%")

#                 l, acc = self.evaluate_main(val_loader)
#                 print(f"[val epoch={i+1}] loss: {l}, acc: {acc}%")
