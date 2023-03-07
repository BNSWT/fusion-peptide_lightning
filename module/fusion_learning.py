'''
Copyright (c) 2023 by Repr. Lab, Westlake University, All Rights Reserved. 
Author: Yuyang Zhou
Date: 2023-02-19 22:02:23
LastEditTime: 2023-02-19 23:58:58
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import Accuracy
from torchmetrics import Precision
from torchmetrics import Recall
from torchmetrics import ROC
from torchmetrics import AUROC
import pytorch_lightning as pl
import esm

from data import *

class FusionLearning(pl.LightningModule):
    def __init__(self, batch_size, pretrained_model):
        super().__init__()
        pretrained = getattr(__import__("esm"), "pretrained")
        model = getattr(pretrained, pretrained_model)
        self.backbone, self.alphabet = model()
        repr_dim = self.backbone.embed_dim
        num_target_classes = 1
        self.linear = nn.Linear(repr_dim, num_target_classes)
        self.sigmoid = nn.Sigmoid()
        self.batch_size = batch_size
    
    def predict(self, output):
        ans = []
        for t in output:
            if t < 0.9:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        torch.cuda.empty_cache()
        batch_tokens, batch_labels = batch
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        results = self.backbone(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        
        sequence_representations = []
        for i, _ in enumerate(token_representations):
            tokens_len = batch_lens[i]
            sequence_representations.append(torch.tensor(token_representations[i, 1:tokens_len-1].mean(0)))
        
        sequence_representations = torch.cat(sequence_representations, dim=-1).reshape(-1, token_representations.shape[-1])
         
        output = self.linear(sequence_representations)
        result = self.sigmoid(output)
        result = result.squeeze(dim=-1)
        
        loss_func = torch.nn.BCELoss()
        result = result.float()
        batch_labels = batch_labels.float()
        loss = loss_func(result, batch_labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        preds = self.predict(result).to(self.device)
        
        return {"loss":loss,"preds":preds.detach(),"labels":batch_labels.detach(),"results":result.detach()}
    
    def training_step_end(self,outputs):
        acc = Accuracy('binary').to(self.device)
        precision = Precision('binary').to(self.device)
        recall = Recall('binary').to(self.device)
        
        train_acc = acc(outputs['preds'], outputs['labels']).item()    
        self.log("train_acc",train_acc,prog_bar=True, on_step=True, on_epoch=True)
        train_precision = precision(outputs['preds'], outputs['labels']).item()    
        self.log("train_precision",train_precision,prog_bar=True, on_step=True, on_epoch=True)
        train_recall = recall(outputs['preds'], outputs['labels']).item()    
        self.log("train_recall",train_recall,prog_bar=True,on_step=True, on_epoch=True)
        
        tp = []
        for result, pred, label in zip(outputs['results'], outputs['preds'], outputs['labels']):
            if pred == 1 and label == 1:
                tp.append(result.cpu())
        self.log("TP mean score:", np.array(tp).mean(),prog_bar=True, on_step=True, on_epoch=True)
        self.log("TP num:", len(tp), prog_bar=True, on_step=True, on_epoch=True)
    
        return {"loss":outputs["loss"].mean(),"preds":outputs['preds'],"labels":outputs['labels'], "results":outputs['results']}

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop.
        self.backbone.eval()
        batch_tokens, batch_labels = batch
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        with torch.no_grad():
            results = self.backbone(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        
        sequence_representations = []
        for i, _ in enumerate(token_representations):
            tokens_len = batch_lens[i]
            sequence_representations.append(torch.tensor(token_representations[i, 1:tokens_len-1].mean(0)))
        
        sequence_representations = torch.cat(sequence_representations, dim=-1).reshape(-1, token_representations.shape[-1])
        output = self.linear(sequence_representations)
        result = self.sigmoid(output)
        result = result.squeeze(dim=-1)
        loss_func = torch.nn.BCELoss()
        
        result = result.float()
        batch_labels = batch_labels.float()
        loss = loss_func(result, batch_labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True)
        
        preds = self.predict(result).to(self.device)
        return {"loss":loss,"preds":preds.detach(),"labels":batch_labels.detach(), "results":result.detach()}
    
    def test_step_end(self,outputs):
        acc = Accuracy('binary').to(self.device)
        precision = Precision('binary').to(self.device)
        recall = Recall('binary').to(self.device)
        auroc = AUROC('binary').to(self.device)

        
        test_acc = acc(outputs['preds'], outputs['labels']).item()    
        self.log("test_acc",test_acc,prog_bar=True, on_step=True, on_epoch=True)
        test_precision = precision(outputs['preds'], outputs['labels']).item()    
        self.log("test_precision",test_precision,prog_bar=True, on_step=True, on_epoch=True)
        test_recall = recall(outputs['preds'], outputs['labels']).item()    
        self.log("test_recall",test_recall,prog_bar=True, on_step=True, on_epoch=True)
        test_auroc = auroc(outputs['preds'], outputs['labels']).item()    
        self.log("test_auroc",test_auroc,prog_bar=True, on_step=True, on_epoch=True)        
        
        return {"loss":outputs["loss"].mean(),"preds":outputs['preds'],"labels":outputs['labels'], "results":outputs['results']}
    
    def test_epoch_end(self, outputs):
        roc = ROC('binary')
        auroc = AUROC('binary')
        
        results = []
        labels = []
        preds = []
        for output in outputs:
            results += output['results'].cpu()
            labels +=  output['labels'].cpu()
            preds += output['preds'].cpu()
        results = torch.tensor(results)
        labels = torch.tensor(labels).int()
        fpr, tpr, thresholds = roc(results, labels)
        auc = auroc(results, labels).item()
        roc_graph(fpr, tpr, auc)
        
        tp = []
        for result, pred, label in zip(results, preds, labels):
            if pred == 1 and label == 1:
                tp.append(result)
        distribution(tp)
        print("TP:", tp)
        return

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        self.backbone.eval()
        batch_tokens, batch_labels = batch
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = self.backbone(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        
        sequence_representations = []
        for i, _ in enumerate(token_representations):
            tokens_len = batch_lens[i]
            sequence_representations.append(torch.tensor(token_representations[i, 1:tokens_len-1].mean(0)))
        
        sequence_representations = torch.cat(sequence_representations, dim=-1).reshape(-1, token_representations.shape[-1])
        output = self.linear(sequence_representations)
        result = self.sigmoid(output)
        result = result.squeeze(dim=-1)
        
        loss_func = torch.nn.BCELoss()
        result = result.float()
        batch_labels = batch_labels.float()
        loss = loss_func(result, batch_labels)
        self.log('validation_loss', loss)
        
        preds = self.predict(result).to(self.device)
        return {"loss":loss,"preds":preds.detach(),"labels":batch_labels.detach()}
    
    def validation_step_end(self,outputs):
        acc = Accuracy('binary').to(self.device)
        precision = Precision('binary').to(self.device)
        recall = Recall('binary').to(self.device)
        
        validation_acc = acc(outputs['preds'], outputs['labels']).item()    
        self.log("validation_acc",validation_acc,prog_bar=True)
        validation_precision = precision(outputs['preds'], outputs['labels']).item()    
        self.log("validation_precision",validation_precision,prog_bar=True)
        validation_recall = recall(outputs['preds'], outputs['labels']).item()    
        self.log("validation_recall",validation_recall,prog_bar=True)
        return {"loss":outputs["loss"].mean()}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
