#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:02:17 2021

@author: ravi
"""

import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 pad_signature, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.pad_signature = pad_signature
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len, embed dim]
        
        src_mask = (torch.sum(src, dim = 2) != self.pad_signature).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len, embed dim]
        
        trg_pad_mask = (torch.sum(trg, dim = 2) != self.pad_signature).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg, attention_mask = None):
        
        #src = [batch size, src len, embed dim]
        #trg = [batch size, trg len, embed dim]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        pred_len, enc_src, enc_mem = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, embed dim]
        #enc_mem = [[batch size, src len, embed dim]*num_layers]
                
        output, attention = self.decoder(src, trg, enc_src, 
                                         enc_mem, trg_mask, src_mask, 
                                         attention_mask = attention_mask)
        
        #output = [batch size, trg len, embed dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return pred_len, output, attention