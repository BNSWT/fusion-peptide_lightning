'''
Author: Yuyang Zhou @ Westlake University
Date: 2023-02-19 11:25:29
LastEditTime: 2023-02-20 11:54:15
LastEditors: Please set LastEditors
Description: 
'''

import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


from data import *
from module import *

if __name__ == "__main__":

    datasetSpliter = DatasetSpliter(positive_path="/root/fusion-peptide_lightning/sequence/positive.csv", negative_path='/root/fusion-peptide_lightning/sequence/negative.xlsx')

    interface = DInterface('sequence_dataset', datasetSpliter, batch_size=10)
    interface.setup()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    fusionModel = FusionLearning(datasetSpliter.train_lens, datasetSpliter.test_lens, batch_size=10)
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1000, auto_lr_find=True, logger=wandb_logger)

    trainer.fit(model=fusionModel,train_dataloaders=interface.train_dataloader(), val_dataloaders=interface.val_dataloader())
    trainer.test(model=fusionModel, dataloaders=interface.test_dataloader())