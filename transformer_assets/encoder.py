#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:58:17 2021

@author: ravi
"""

import torch
import torch.nn as nn
from transformer_assets.encoder_layer import EncoderLayer
from transformer_assets.position_encoding import PositionEncoding

#%%
class Encoder(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 device,
                 pos_encoding = False,
                 max_length = 350):
        super().__init__()

        self.device = device

        if pos_encoding:
            self.pos_embedding = PositionEncoding(max_length, embed_dim, device)
        else:
            self.pos_embedding = nn.Embedding(max_length, embed_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, 
                                                  hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout, 
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.length_layer = nn.Linear(embed_dim, 1)

        self.length_layer_dropout = nn.Dropout(0.75)
        
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, embed dim]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        src = self.dropout((src * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, embed dim]
        #encoder_memory = [[batch size, src len, embed dim]*num_layers]
        
        enc_mem = list()
        for layer in self.layers:
            enc_mem.append(src)
            src = layer(src, src_mask)
            
        #src = [batch size, src len, embed dim]
        
        #pred_len = [batch size, 1]

        pred_len = self.length_layer(self.length_layer_dropout(torch.mean(src, dim = 1)))
            
        return pred_len, src, enc_mem