'''
Copyright (c) 2023 by Repr. Lab, Westlake University, All Rights Reserved. 
Author: Yuyang Zhou
Date: 2023-02-19 15:07:56
LastEditTime: 2023-02-19 21:45:27
'''
import sys 
sys.path.append("..") 
from data import *

fastaProcessor = FastaProcessor(positive_path="/zhouyuyang/fusion-peptide_lightning/sequence/positive.csv", negative_path='/zhouyuyang/fusion-peptide_lightning/sequence/negative.xlsx')
datasetSpliter = DatasetSpliter(fastaProcessor.positive_seq, fastaProcessor.negative_seq)

print(len(datasetSpliter.train_labels))
print(len(datasetSpliter.test_labels))
print(len(datasetSpliter.validation_labels))


# interface = DInterface('sequence_dataset', datasetSpliter, batch_size=10)
# interface.setup()
# interface.train_dataloader()
# interface.val_dataloader()
# interface.test_dataloader()

# fastaProcessor = FastaProcessor()
# group = fastaProcessor.get_group('/zhouyuyang/fusion-peptide_lightning/sequence/positive_group.fasta')
# print("end")