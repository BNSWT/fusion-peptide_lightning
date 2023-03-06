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

    fastaProcessor = FastaProcessor(positive_path="/zhouyuyang/fusion-peptide_lightning/sequence/positive_group.fasta", negative_path='/zhouyuyang/fusion-peptide_lightning/sequence/negative_group.fasta', group=True)
    datasetSpliter = DatasetSpliter(fastaProcessor.positive_seq, fastaProcessor.negative_seq, group=True)

    interface = DInterface('sequence_dataset', datasetSpliter, batch_size=64)
    interface.setup()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    fusionModel = FusionLearning(batch_size=64)
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=1000, logger=wandb_logger)

    trainer.fit(model=fusionModel,train_dataloaders=interface.train_dataloader(), val_dataloaders=interface.val_dataloader())
    trainer.test(model=fusionModel, dataloaders=interface.test_dataloader())
