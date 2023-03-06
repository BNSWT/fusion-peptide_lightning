import esm
import random
import pandas as pd
from distfit import distfit
import numpy as np
from data.visualization import *

class FastaProcessor():
    def __init__(self, positive_path='', negative_path='', group = False, random_sequence_len=50000, cut_negative_seq = True,save_negative=False, save_fasta=True):
        if not positive_path or not negative_path:
            return # empty spliter
        
        if not group:
            self.positive_seq = self.filter_seq(self.read_csv_str(positive_path))
            self.negative_seq = self.filter_seq(self.read_excel_str(negative_path))
        
            if random_sequence_len:
                print("add random")
                random.seed(0)
                str_list = [random.choice('ABCDEFGHIKLMNOPQRSTUVWYZ') for i in range(random_sequence_len)]
                str = ''.join(str_list)
                self.negative_seq.append(str)
                        
            if cut_negative_seq is True:
                self.negative_seq = self.cut_neg(self.positive_seq, self.negative_seq)

            if save_negative:
                self.save_seq(self.negative_seq, "/zhouyuyang/fusion-peptide_lightning/sequence/negative.txt")
            
            if save_fasta:
                self.save_fasta(self.positive_seq, "/zhouyuyang/fusion-peptide_lightning/sequence/positive.fasta")
                self.save_fasta(self.negative_seq, "/zhouyuyang/fusion-peptide_lightning/sequence/negative.fasta")
        else:
            self.positive_seq = self.get_group(positive_path)
            self.negative_seq = self.get_group(negative_path)
    def get_seq(self, path):
        seqs = []
        # Only for those fasta without newline inside sequence part
        with open(path, "r") as fp:
            for line in fp.readlines():
                if '>' in line:
                    continue
                else:
                    seqs.append(line.strip('\n'))
        return seqs
    def get_group(self, path):
        group = []
        with open(path, "r") as fp:
            all_content = fp.read()
            content = []
            for sequence in all_content.split('>'):
                items = sequence.split('\n')
                if len(items)==2: # start of a group
                    if len(content):
                        group.append(content)
                    content = []
                else:
                    sequence = ''
                    for i, item in enumerate(items):
                        if i != len(items)-1 and i != 0:
                            sequence += item
                    if len(sequence):
                        content.append(sequence)
            if len(content):
                group.append(content)
        return group
    
    def read_csv_str(self, path):
        return pd.read_csv(path)['sequence'].to_list()
    
    def read_excel_str(self, path):
        return pd.read_excel(path)['Sequence'].to_list()
    
    def save_seq(self, seqs, path):
        with open(path, "w") as fp:
            for seq in seqs:
                fp.write(seq+'\n')
    
    def load_seq(self, path):
        seqs = []
        with open(path, "r") as fp:
            for line in fp.readlines():
                seqs.append(line.strip('\n'))
        return seqs
                
    def save_fasta(self, seqs, path):
        with open(path, "w") as fp:
            for i, seq in enumerate(seqs):
                fp.write(">sequence"+str(i)+'\n')
                fp.write(seq+'\n')

    def filter_seq(self, seqs):
        seqs = list(set(seqs))
        seqs = [seq for seq in seqs if type(seq) is str]
        seqs = [seq.replace('(', '').replace(')', '') for seq in seqs]
        return seqs
    
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
