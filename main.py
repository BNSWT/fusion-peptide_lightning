'''
Author: Yuyang Zhou @ Westlake University
Date: 2023-02-19 11:25:29
LastEditTime: 2023-02-20 11:54:15
LastEditors: Please set LastEditors
Description: 
'''

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser

from data import *
from module import *

def main(args):
    pl.seed_everything(args.seed)
    fastaProcessor = FastaProcessor(positive_path=args.positive_path, negative_path=args.negative_path, group=args.group)
    datasetSpliter = DatasetSpliter(fastaProcessor.positive_seq, fastaProcessor.negative_seq, pretrained_model=args.pretrained_model,group=args.group)

    interface = DInterface('sequence_dataset', datasetSpliter, batch_size=args.batch_size)
    interface.setup()

    fusionModel = FusionLearning(batch_size=args.batch_size, pretrained_model=args.pretrained_model)
    wandb_logger = WandbLogger()
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, max_epochs=args.max_epochs, logger=wandb_logger)

    trainer.fit(model=fusionModel,train_dataloaders=interface.train_dataloader(), val_dataloaders=interface.val_dataloader())
    trainer.test(model=fusionModel, dataloaders=interface.test_dataloader())


if __name__ == "__main__":
    parser = ArgumentParser()
    # Hardware parameters
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--devices", default=1, type=int)

    # Pretrained model name
    parser.add_argument("--pretrained_model", default="esm2_t30_150M_UR50D", type=str)
    
    # Dataset path
    parser.add_argument("--positive_path", default="/zhouyuyang/fusion-peptide_lightning/sequence/positive_group.fasta", type=str)
    parser.add_argument("--negative_path", default="/zhouyuyang/fusion-peptide_lightning/sequence/negative_group.fasta", type=str)
    parser.add_argument("--group", default=True, type=bool)

    # Trainning hyper parameters
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_epochs", default=1000, type=int)
    parser.add_argument("--seed", default=44, type=int)

    args = parser.parse_args()
    main(args)