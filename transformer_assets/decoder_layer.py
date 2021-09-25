#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 10:59:42 2021

@author: ravi
"""

import torch
import torch.nn as nn
from transformer_assets.multi_head_attention_layer import MultiHeadAttentionLayer
from transformer_assets.position_wise_feedforward_layer import PositionwiseFeedforwardLayer

#%%
class DecoderLayer(nn.Module):
    def __init__(self, 
                 embed_dim, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device):
        super().__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attention = MultiHeadAttentionLayer(embed_dim, hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(embed_dim, hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(embed_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask, attention_mask = None):
        
        #trg = [batch size, trg len, embed dim]
        #enc_src = [batch size, src len, embed dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        
        #dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
            
        #trg = [batch size, trg len, embed dim]
            
        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, 
                                                 enc_src, src_mask, 
                                                 attention_mask = attention_mask)
        
        #dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
                    
        #trg = [batch size, trg len, embed dim]
        
        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        
        #dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        #trg = [batch size, trg len, embed dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return trg, attention