#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 14:27:30 2021

@author: ravi
"""

#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

#%%
class Decoder(nn.Module):
    def __init__(self, 
                 emb_dim, 
                 hid_dim, 
                 n_layers, 
                 kernel_size, 
                 dropout, 
                 pad_idx, 
                 pad_vector, 
                 device,
                 attn_mask=None,
                 max_length = 150):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.pad_idx = pad_idx
        self.device = device
        self.max_length = max_length
        self.lower_limit = -2**32 + 1
        self.attn_mask = attn_mask
        self.pad_vector = pad_vector
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.pos_embedding = nn.Embedding(max_length, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
        self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)
        
        self.fc_out = nn.Linear(hid_dim, emb_dim)
        self.activation = nn.Tanh()
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
                                              out_channels = 2 * hid_dim, 
                                              kernel_size = kernel_size)
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
      
    def calculate_attention(self, embedded, conved, encoder_conved, encoder_combined):
        
        #embedded = [batch size, trg len, emb dim]
        #conved = [batch size, hid dim, trg len]
        #encoder_conved = encoder_combined = [batch size, src len, emb dim]


        #permute and convert back to emb dim
        #conved_emb = [batch size, trg len, emb dim]
        conved_emb = self.activation(self.attn_hid2emb(conved.permute(0, 2, 1)))

        
        #combined = [batch size, trg len, emb dim]
        combined = (conved_emb + embedded) * self.scale


        #energy = [batch size, trg len, src len]
        energy = torch.matmul(conved_emb, encoder_conved.permute(0, 2, 1)) #originally combined instead of conved_emb
        
        
        #create attention masking
        if self.attn_mask is not None:
            src_len = energy.shape[2]
            tar_len = energy.shape[1]
            batch_size = energy.shape[0]
            diff_col = self.max_length - src_len
            diff_row = self.max_length - tar_len
            energy = torch.cat((energy, torch.zeros(batch_size, tar_len, 
                                                    diff_col).to(self.device)), dim = 2)
            energy = torch.cat((energy, torch.zeros(batch_size, diff_row, 
                                                    self.max_length).to(self.device)), dim = 1)
            padding_matrix = torch.eq(self.attn_mask, 0.).type(torch.float32)
            padding_matrix = self.lower_limit * padding_matrix
            energy += padding_matrix
            energy = energy[:, 0:tar_len, 0:src_len]


        #attention = [batch size, trg len, src len]        
        attention = F.softmax(energy, dim=2)


        #attended_encoding = [batch size, trg len, emd dim]
        attended_encoding = torch.matmul(attention, encoder_combined)

        
        #convert from emb dim -> hid dim
        #attended_encoding = [batch size, trg len, hid dim]
        attended_encoding = self.activation(self.attn_emb2hid(attended_encoding))
        attended_encoding = attended_encoding.permute(0, 2, 1)

        
        #apply residual connection
        #attended_combined = [batch size, hid dim, trg len]
        attended_combined = (conved + attended_encoding) * self.scale

        
        return attention, attended_combined
        
    def forward(self, trg, encoder_conved, encoder_combined):
        
        #trg = [batch size, trg len, output dim]
        #encoder_conved = encoder_combined = [batch size, src len, emb dim]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

            
        #create position tensor
        #pos = [batch size, trg len]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        
        #embed positions
        #pos_embedded = [batch size, trg len, emb dim=output dim]
        pos_embedded = self.pos_embedding(pos)
        

        #combine embeddings by elementwise summing
        #embedded = [batch size, trg len, emb dim]
        # embedded = self.dropout(trg + pos_embedded)
        embedded = trg + pos_embedded

        
        #pass embedded through linear layer to go through emb dim -> hid dim
        #conv_input = [batch size, trg len, hid dim]
        conv_input = self.activation(self.emb2hid(embedded))


        #pad vector needs to be projected as well
        pad_projected = self.activation(self.emb2hid(self.pad_vector.to(self.device)))
        pad_projected = pad_projected.permute(0, 2, 1)

        
        #permute for convolutional layer
        #conv_input = [batch size, hid dim, trg len]
        conv_input = conv_input.permute(0, 2, 1)

        
        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]

        
        for i, conv in enumerate(self.convs):
        
            #apply dropout
            conv_input = self.dropout(conv_input)

        
            #need to pad so decoder can't "cheat"
            # padding = torch.zeros(batch_size, hid_dim, 
            #                       self.kernel_size - 1).fill_(self.pad_idx).to(self.device)
            # padding = torch.zeros(batch_size, hid_dim, self.kernel_size - 1).to(self.device)
            padding = pad_projected.repeat(batch_size, 1, self.kernel_size - 1)


            #padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]
            padded_conv_input = torch.cat((padding, conv_input), dim = 2)

        
            #pass through convolutional layer
            #conved = [batch size, 2 * hid dim, trg len]
            conved = conv(padded_conv_input)

            
            #pass through GLU activation function
            #conved = [batch size, hid dim, trg len]
            conved = F.glu(conved, dim = 1)

            
            #calculate attention
            #attention = [batch size, trg len, src len]
            #conved = [batch size, hid dim, trg len]
            attention, conved = self.calculate_attention(embedded, 
                                                         conved, 
                                                         encoder_conved, 
                                                         encoder_combined)

            
            #apply residual connection orignally residual connection was added here
            conved = (conved + conv_input) * self.scale

            
            #set conv_input to conved for next loop iteration
            conv_input = conved


        #conved = [batch size, trg len, emb dim]            
        conved = conved.permute(0, 2, 1)


        #output = [batch size, trg len, output dim]            
        output = self.fc_out(self.dropout(conved))

            
        return output, attention