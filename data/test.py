'''
Copyright (c) 2023 by Repr. Lab, Westlake University, All Rights Reserved. 
Author: Yuyang Zhou
Date: 2023-02-19 15:07:56
LastEditTime: 2023-02-19 21:45:27
'''
import sys 
sys.path.append("..") 
from data import *

# Split test
if 1:
    fastaProcessor = FastaProcessor(positive_path="/zhouyuyang/fusion-peptide_lightning/sequence/positive_group.fasta", negative_path='/zhouyuyang/fusion-peptide_lightning/sequence/negative_group.fasta', group=True)
    datasetSpliter = DatasetSpliter(fastaProcessor.positive_seq, fastaProcessor.negative_seq, group=True)

    print("train len:", len(datasetSpliter.train_labels))
    print("test len:", len(datasetSpliter.test_labels))
    print("validation len:", len(datasetSpliter.validation_labels))
    print("train_pos_len:", datasetSpliter.train_pos_len)
    print("test_pos_len:", datasetSpliter.test_pos_len)
    print("validation_pos_len:", datasetSpliter.validation_pos_len)

# Interface test
if 1:
    interface = DInterface('sequence_dataset', datasetSpliter, batch_size=4)
    interface.setup()
    print("====train data====")
    for i, data in enumerate(interface.train_dataloader()):
        if i == 1:
            break
        print(data)
    print("====val data====")
    for i, data in enumerate(interface.val_dataloader()):
        if i == 1:
            break
        print(data)
    print("====test data====")
    for i, data in enumerate(interface.test_dataloader()):
        if i == 1:
            break
        print(data)

# get group test
if 0:
    fastaProcessor = FastaProcessor()
    group = fastaProcessor.get_group('/zhouyuyang/fusion-peptide_lightning/sequence/positive_group.fasta')