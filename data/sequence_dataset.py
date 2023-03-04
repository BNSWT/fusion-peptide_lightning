'''
Copyright (c) 2023 by Repr. Lab, Westlake University, All Rights Reserved. 
Author: Yuyang Zhou
Date: 2023-02-19 11:49:53
LastEditTime: 2023-02-19 21:50:27
'''
import torch.utils.data as data

class SequenceDataset(data.Dataset):
    def __init__(self, token, label, pos_len):
        self.token = token
        self.label = label
        self.pos_len = pos_len
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.token[idx], self.label[idx]