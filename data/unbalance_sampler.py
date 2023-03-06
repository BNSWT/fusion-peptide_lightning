import random
from data.sequence_dataset import SequenceDataset

class UnbalanceSampler():
    def __init__(self, dataset:SequenceDataset):
        self.pos_len = dataset.pos_len
        self.dataset_len = len(dataset)
        self.neg_len = self.dataset_len-self.pos_len
        self.pos_indices = list(range(self.pos_len))
        self.neg_indices = list(range(self.pos_len, self.dataset_len))
        random.shuffle(self.pos_indices)
        random.shuffle(self.neg_indices)
    def __iter__(self):
        indices = [self.pos_indices[i%self.pos_len] if i%2==0 else self.neg_indices[i] for i in range(self.neg_len)]
        return iter(indices)
    def __len__(self):
        return len(self.pos_indices)+len(self.neg_indices)
