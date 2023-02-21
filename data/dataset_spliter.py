'''
Copyright (c) 2023 by Repr. Lab, Westlake University, All Rights Reserved. 
Author: Yuyang Zhou
Date: 2023-02-19 14:29:18
LastEditTime: 2023-02-19 21:34:47
'''
import esm
import random
import pandas as pd
from .visualization import *
from distfit import distfit
import numpy as np

class DatasetSpliter():
    def __init__(self, positive_path = '', negative_path = '', cut_negative_seq = True, random_sequence_len = 10000):
        if not positive_path or not negative_path:
            return # empty spliter
        
        
        positive_seq = self.filter_seq(self.read_csv_str(positive_path))
        negative_seq = self.filter_seq(self.read_excel_str(negative_path))
        
        if random_sequence_len is int:
            str_list = [random.choice('ABCDEFGHIKLMNOPQRSTUVWYZ') for i in range(random_sequence_len)]
            negative_seq.append(str_list)
                    
        if cut_negative_seq is True:
            negative_seq = self.cut_neg(positive_seq, negative_seq)
        
        pos_len = len(positive_seq)
        neg_len = len(negative_seq)

        test_pos_len = int(pos_len*0.3)
        test_neg_len = int(neg_len*0.3)

        val_pos_len = int(pos_len*0.05)
        val_neg_len = int(neg_len*0.05)
        
        _, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.converter = alphabet.get_batch_converter()
        
        self.train_labels, self.train_strs, self.train_tokens = self.embed(positive_seq[:-test_pos_len], negative_seq[:-test_neg_len])
        self.train_lens = (self.train_tokens != alphabet.padding_idx).sum(1)
        
        self.test_labels, self.test_strs, self.test_tokens = self.embed(positive_seq[-test_pos_len:-val_pos_len], negative_seq[-test_neg_len:-val_neg_len])
        self.test_lens = (self.test_tokens != alphabet.padding_idx).sum(1)
        
        self.validation_labels, self.validation_strs, self.validation_tokens = self.embed(positive_seq[-val_pos_len:], negative_seq[-val_neg_len:])
        self.validation_lens = (self.validation_tokens != alphabet.padding_idx).sum(1)
        
    def read_csv_str(self, path):
        return pd.read_csv(path)['sequence'].to_list()
    
    def read_excel_str(self, path):
        return pd.read_excel(path)['Sequence'].to_list()
    
    def filter_seq(self, seqs):
        seqs = list(set(seqs))
        seqs = [seq for seq in seqs if type(seq) is str]
        seqs = [seq.replace('(', '').replace(')', '') for seq in seqs]
        return seqs
    
    def embed(self, positive_seq, negative_seq):
        positive_data = [(1., seq) for seq in positive_seq]
        negative_data = [(0., seq) for seq in negative_seq]
        
        all_data = positive_data + negative_data
        random.shuffle(all_data)
    
        return self.converter(all_data)
    
    def cut_neg(self, positive_seq, negative_seq, visualize=True):
        positive_lens = [len(seq) for seq in positive_seq if type(seq) is str]
        dist = distfit(todf=True)
        dist.fit_transform(np.array(positive_lens))
        
        if visualize is True:
            distribution(positive_lens)
            dist.plot()
        
        params = list(dist.model['params'])
        stats = getattr(__import__("scipy"), "stats")
        dist_func = getattr(stats, dist.model['name'])
        rvs = getattr(dist_func, "rvs")
        
        return self.cut(negative_seq, rvs, params)
    
    def cut(self, long, rvs, params, gap=3):
        short = []
        for seq in long:
            pos = 0
            seq_len = len(seq)
            while pos < seq_len:
                new_len = int(rvs(*params))
                while new_len <=0:
                    new_len = int(rvs(*params))
                if new_len > seq_len - pos:
                    new_len = seq_len - pos
                short.append(seq[pos:pos+new_len])
                pos += gap
        return short