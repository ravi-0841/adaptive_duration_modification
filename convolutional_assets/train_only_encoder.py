#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:26:10 2021

@author: ravi
"""

import torch
import torch.nn as nn
import numpy as np

#%%
class TrainingEval(object):
    
    def __init__(self, emb_dim, model, 
                 learning_rate, device, 
                 clip, max_len, pad_signature, 
                 exp_mask=False):
        
        self.model = model
        self.learning_rate = learning_rate
        self.grad_clip = clip
        self.pad_signature = pad_signature
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.device = device
        self.exp_mask = exp_mask
        self.len_loss_wt = 5
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.learning_rate)
        self.criterion = nn.L1Loss(reduction = 'none')

#%%
    def train(self, iterator):

        self.model.train()

        epoch_loss      = 0
        
        generated_lens  = []

        for i in range(iterator.batch_count()):
            
            src, trg, inp_seq_len, out_seq_len, mask = iterator[i]
            
            #create target for sequence length prediction
            seq_len_trg = (out_seq_len - 1) / (inp_seq_len - 1)
            
            self.optimizer.zero_grad()
            pred_len, _, _ = self.model.encoder(src[:,1:,:])
            
            #put predicted data in the list
            generated_lens.append((pred_len.cpu().detach().numpy(), 
                                   seq_len_trg.cpu().numpy()))
            
            #output = [batch size, trg len - 1, output dim]
            cur_batch_size = src.shape[0]
            
            len_loss = torch.mean(self.criterion(seq_len_trg, pred_len))
            len_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            epoch_loss += cur_batch_size*len_loss.item()
            
        return epoch_loss/len(iterator), generated_lens

#%%
    def evaluate(self, iterator):

        self.model.eval()
    
        epoch_loss      = 0
        
        generated_lens  = []
        
        with torch.no_grad():
        
            for i in range(iterator.batch_count()):
            
                src, trg, inp_seq_len, out_seq_len, mask = iterator[i]
                
                #create target for sequence length prediction
                seq_len_trg = (out_seq_len - 1) / (inp_seq_len - 1)
    
                pred_len, _, _ = self.model.encoder(src[:,1:,:])
                
                #output predicted data in the list
                generated_lens.append((pred_len.cpu().detach().numpy(), 
                                       seq_len_trg.cpu().numpy()))
            
                #output = [batch size, trg len - 1, output dim]
                #trg = [batch size, trg len - 1, output dim]
                cur_batch_size = src.shape[0]
                
                len_loss = torch.mean(self.criterion(seq_len_trg, pred_len))
    
                epoch_loss += cur_batch_size*len_loss.item()
            
        return epoch_loss/len(iterator), generated_lens

#%%
    def test_predict(self, src):

        """
        src: [1, timestamps, dim]
        """
        self.model.eval()
        
        src_len = int(src.shape[1]) - 1
    
        with torch.no_grad():
            #Encoder operation to get embeddings and predicted length
            pred_len, encoder_conved, encoder_conved_embed = self.model.encoder(src[:,1:,:])
        
        return pred_len.cpu().detach().numpy()
