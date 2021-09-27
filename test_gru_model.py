#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 15:23:23 2021

@author: ravi
"""

#%% Loading all required packages

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

import sys
import random
import math
import time
import pickle
import joblib
import pylab
import logging

from load_data import LoadData
from utils import count_parameters, epoch_time

#%% 

def evaluate(model, iterator, criterion, pad_signature, len_loss_wt):
    
    model.eval()
    
    epoch_reg_loss  = []
    epoch_len_loss  = []
    epoch_loss      = []

    generated_seqs = []
    generated_attn = []

    with torch.no_grad():
    
        for i in range(iterator.batch_count()):
        
            #src = [batch size, src len, emb_dim]
            #tar = [batch size, tar len, emb_dim]

            src, trg, inp_seq_len, out_seq_len, _ = iterator[i]
            
            #create target for sequence length prediction

            seq_len_trg = (out_seq_len - 1) / (inp_seq_len - 1)

            #src = [src len, batch size, emb_dim]
            #tar = [tar len, batch size, emb_dim]

            src = src.permute(1, 0, 2)
            trg = trg.permute(1, 0, 2)

            #output = [batch size, (trg len - 1), emb dim]

            output, attn, pred_len = model(src, trg, 0) #turn off teacher forcing

            batch_size = output.shape[1]
            output_dim = output.shape[-1]

            #trg = [batch size, (trg len - 1), emb dim]
            #output = [batch size, (trg len - 1), emb dim]

            trg = trg[1:,:,:].permute(1, 0, 2)
            output = output.permute(1, 0, 2)
            
            nonpadding = torch.ne(torch.sum(trg, dim=-1), pad_signature)
            nonpadding = nonpadding.type(torch.float32)
                
            reg_loss = torch.sum(torch.sum(criterion(output, trg), 
                        dim=-1) * nonpadding) / (torch.sum(nonpadding) + 1e-7)
            len_loss = torch.mean(criterion(seq_len_trg, pred_len))
            
            loss = reg_loss + len_loss_wt*len_loss

            epoch_reg_loss.append(batch_size*reg_loss.item())
            epoch_len_loss.append(batch_size*len_loss.item())
            epoch_loss.append(batch_size*loss.item())

            generated_seqs.append(output.cpu().detach().numpy())
            generated_attn.append(attn.cpu().detach().numpy())
        
    return epoch_reg_loss, epoch_len_loss, \
        epoch_loss, generated_seqs, generated_attn