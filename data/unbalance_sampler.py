import random
from torch.utils.data.sampler import Sampler
from data.sequence_dataset import SequenceDataset

class UnbalanceSampler(Sampler):
    def __init__(self, dataset:SequenceDataset):
        self.pos_len = dataset.pos_len
        self.dataset_len = len(dataset)
        self.pos_indices = list(range(self.pos_len))
        self.neg_indices = list(range(self.pos_len, self.dataset_len))
        random.shuffle(self.pos_indices)
        random.shuffle(self.neg_indices)
        self.__n = -1
        self.is_pos = True
    def __next__(self):
        if self.__n+1 < self.dataset_len - self.pos_len:
            if self.is_pos:
                self.__n += 1
                self.is_pos = False
                return self.pos_indices[self.__n % self.pos_len]
            else:
                self.is_pos = True
                return self.neg_indices[self.__n]
        else:
            raise StopIteration
    def __iter__(self):
        return self
    def __len__(self):
        return len(self.pos_indices+self.neg_indices)
