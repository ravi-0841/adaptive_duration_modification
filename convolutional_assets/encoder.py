#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 14:24:30 2021

@author: ravi
"""

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import positional_encoding

#%%
class Encoder(nn.Module):
    def __init__(self, 
                 emb_dim, 
                 hid_dim, 
                 n_layers, 
                 kernel_size, 
                 dropout, 
                 device,
                 pos_enc=False,
                 max_length = 150):
        super().__init__()
        
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        
        self.device = device
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.temp_scale = torch.sqrt(torch.FloatTensor([0.0067])).to(device)
        
        if pos_enc:
            self.pos_embedding = positional_encoding(max_length, emb_dim)
        else:
            self.pos_embedding = nn.Embedding(max_length, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self.activation = nn.Tanh()
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
                                              out_channels = 2 * hid_dim, 
                                              kernel_size = kernel_size, 
                                              padding = (kernel_size - 1) // 2)
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        self.len_dropout = nn.Dropout(0.9)
        
        #length prediction
        self.length_linear = nn.Linear(hid_dim, 1)
        self.length_activation = nn.Sigmoid()
        
    def forward(self, src):
        
        #src = [batch size, src len, emb_dim]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]


        #create position tensor
        #pos = [0, 1, 2, 3, ..., src len - 1]
        #pos = [batch size, src len]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)


        #combine embeddings by elementwise summing
        #embedded = [batch size, src len, emb dim]
        embedded = src + self.pos_embedding(pos)

        
        #pass embedded through linear layer to convert from emb dim to hid dim
        #conv_input = [batch size, src len, hid dim]
        conv_input = self.activation(self.emb2hid(embedded))

        
        #permute for convolutional layer
        #conv_input = [batch size, hid dim, src len]        
        conv_input = conv_input.permute(0, 2, 1)


        #begin convolutional blocks...
        
        for i, conv in enumerate(self.convs):
        
            #pass through convolutional layer
            #conved = [batch size, 2 * hid dim, src len]
            conved = conv(self.dropout(conv_input))


            #pass through GLU activation function
            #conved = [batch size, hid dim, src len]
            conved = F.glu(conved, dim = 1)

            
            #apply residual connection
            #conved = [batch size, hid dim, src len]
            conved = (conved + conv_input) * self.scale

    
            #set conv_input to conved for next loop iteration
            conv_input = conved

        
        #...end convolutional blocks
        
        #dropout on convolution output
        conved = self.dropout(conved)


        #length prediction based on cummulative sum across time
        encoder_sum = torch.sum(conved, dim = 2) * self.temp_scale #mean
        len_pred = self.length_linear(self.len_dropout(encoder_sum))


        #permute and convert back to emb dim
        #conved = [batch size, src len, emb dim]
        conved = self.activation(self.hid2emb(conved.permute(0, 2, 1)))
        conved_embed = (conved + embedded) * self.scale

        
        return len_pred, conved, embedded
