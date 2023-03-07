'''
Copyright (c) 2023 by Repr. Lab, Westlake University, All Rights Reserved. 
Author: Yuyang Zhou
Date: 2023-02-19 14:29:18
LastEditTime: 2023-02-19 21:34:47
'''
import esm
import pandas as pd
from .visualization import *


class DatasetSpliter():
    def __init__(self, positive_seq, negative_seq, pretrained_model, group = False):
        if positive_seq is None or negative_seq is None:
            return
        if group:
            pos_group_lens = [len(seqs) for seqs in positive_seq]
            neg_group_lens = [len(seqs) for seqs in negative_seq]
            positive_seq = [seq for seqs in positive_seq for seq in seqs]
            negative_seq = [seq for seqs in negative_seq for seq in seqs]

        pos_len = len(positive_seq)
        neg_len = len(negative_seq)

        if not group:
            test_pos_len = int(pos_len*0.3)
            test_neg_len = int(neg_len*0.3)

            val_pos_len = int(pos_len*0.05)
            val_neg_len = int(neg_len*0.05)
        else:
            total = 0
            val_pos_len = 0
            for group_len in reversed(pos_group_lens):
                if total > pos_len*0.3 and val_pos_len:
                    test_pos_len = total
                    break
                elif total > pos_len*0.05 and not val_pos_len:
                    val_pos_len = total 
                total += group_len
            total = 0
            val_neg_len =0
            for group_len in reversed(neg_group_lens):
                if total > neg_len*0.3:
                    test_neg_len = total
                    break
                elif total > neg_len*0.05 and not val_neg_len:
                    val_neg_len = total
                total += group_len
        

        pretrained = getattr(__import__("esm"), "pretrained")
        model = getattr(pretrained, pretrained_model)
        _, alphabet = model()
        self.converter = alphabet.get_batch_converter()
        
        # head : test_pos_len : val_pos_len : tail
        self.train_pos_len=pos_len-test_pos_len
        self.train_labels, self.train_strs, self.train_tokens = self.embed(positive_seq[:-test_pos_len], negative_seq[:-test_neg_len])
    
        self.test_pos_len=test_pos_len-val_pos_len
        self.test_labels, self.test_strs, self.test_tokens = self.embed(positive_seq[-test_pos_len:-val_pos_len], negative_seq[-test_neg_len:-val_neg_len])
        
        self.validation_pos_len=val_pos_len
        self.validation_labels, self.validation_strs, self.validation_tokens = self.embed(positive_seq[-val_pos_len:], negative_seq[-val_neg_len:])

    def embed(self, positive_seq, negative_seq):
        positive_data = [(1., seq) for seq in positive_seq]
        negative_data = [(0., seq) for seq in negative_seq]
        
        all_data = positive_data + negative_data
    
        return self.converter(all_data)