#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:01:13 2021

@author: ravi
"""

import torch
import torch.nn as nn
from transformer_assets.decoder_layer import DecoderLayer
from transformer_assets.position_encoding import PositionEncoding

#%%
class Decoder(nn.Module):
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
        
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, 
                                                  hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout, 
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim])).to(device)
        
    def forward(self, src, trg, enc_src, enc_mem, 
                trg_mask, src_mask, attention_mask = None):
        
        #src = [batch size, src len, embed dim]
        #trg = [batch size, trg len, embed dim]
        #enc_src = [batch size, src len, embed dim]
        #enc_mem = [[batch size, src len, embed dim]*num_layers]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((trg * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, embed dim]
        
        enc_mem = enc_mem[::-1]
        for i, layer in enumerate(self.layers):
            trg, attention = layer(trg, enc_src, trg_mask, 
                                   src_mask, attention_mask = attention_mask)
            
            #add src at each layer output "Overkill"
            # trg += torch.matmul(attention.squeeze(1), enc_mem[i])
        
        #trg = [batch size, trg len, embed dim]
        #attention = [batch size, n heads, trg len, src len]

        attention = torch.mean(attention, dim=1, keepdim=True)
        output = self.fc_out(self.dropout(trg)) + torch.matmul(attention.squeeze(1), src)
        # output = self.fc_out(trg)
        
        #output = [batch size, trg len, embed dim]
            
        return output, attention