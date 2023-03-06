'''
Copyright (c) 2023 by Repr. Lab, Westlake University, All Rights Reserved. 
Author: Yuyang Zhou
Date: 2023-02-19 11:28:10
LastEditTime: 2023-02-19 21:49:11
'''
import inspect
import importlib
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.dataset_spliter import DatasetSpliter
from data.unbalance_sampler import UnbalanceSampler

class DInterface(pl.LightningDataModule):
    def __init__(self, dataset='', 
                 dataset_spliter=DatasetSpliter(None, None),
                 **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.dataset = dataset
        self.dataset_spliter = dataset_spliter
        self.load_data_module()
        
    def setup(self):
        self.trainset = self.instancialize(token=self.dataset_spliter.train_tokens, label=self.dataset_spliter.train_labels, pos_len=self.dataset_spliter.train_pos_len)
        self.validationset = self.instancialize(token=self.dataset_spliter.validation_tokens, label=self.dataset_spliter.validation_labels, pos_len=self.dataset_spliter.validation_pos_len)
        self.testset = self.instancialize(token=self.dataset_spliter.test_tokens, label=self.dataset_spliter.test_labels, pos_len=self.dataset_spliter.test_pos_len)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, sampler=UnbalanceSampler(self.trainset))

    def val_dataloader(self):
        return DataLoader(self.validationset, batch_size=self.batch_size, sampler=UnbalanceSampler(self.validationset))

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, sampler=UnbalanceSampler(self.testset))

    def load_data_module(self):
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        name = self.dataset
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            self.data_module = getattr(importlib.import_module(
                '.'+name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Dataset File Name or Invalid Class Name data.{name}')

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)