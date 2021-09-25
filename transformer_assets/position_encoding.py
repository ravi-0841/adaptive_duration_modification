#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:06:58 2021

@author: ravi
"""

import torch
import torch.nn as nn
import numpy as np

#%%
class PositionEncoding(object):

    '''Sinusoidal Positional_Encoding. See 3.5
    embedding_dim: scalar. Dimensionality of the embedding
    maxlen: scalar. Must be >= T

    returns
    2d tensor that has the shape (maxlen, embedding_dim).
    '''

    def __init__(self, max_len, embedding_dim, device):
        
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.device = device
        self.embedding = self.create_embedding()

    def create_embedding(self):

        # First part of the PE function: sin and cos argument
        position_enc = np.array([[pos / np.power(10000, (i-i%2)/self.embedding_dim) \
                                  for i in range(self.embedding_dim)] for pos in range(self.max_len)])
        
        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = torch.from_numpy(position_enc).type(torch.float32) # (maxlen, embedding_dim)
        
        # lookup
        outputs = nn.Embedding.from_pretrained(position_enc)
        
        return outputs.to(self.device)
    
    def __call__(self, x):
        return self.embedding(x)
