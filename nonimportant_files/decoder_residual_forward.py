#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:34:36 2021

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
                 max_length = 150):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.pad_idx = pad_idx
        self.device = device
        self.max_length = max_length
        self.lower_bound = -2**32 + 1
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
    

    def forward_attention_module(self, attention, mask=None):
        
        # attention = [batch size, trg len, src len]
        batch_size = attention.shape[0]
        trg_len = attention.shape[1]
        src_len = attention.shape[2]
        
        #concatenate zeros
        #forward_energy = [batch size, trg len + 1, src len + 1]
        forward_energy = torch.cat((torch.zeros(batch_size, 1, src_len).to(self.device), 
                                      attention), dim = 1)
        forward_energy = torch.cat((torch.zeros(batch_size, trg_len+1, 1).to(self.device), 
                                      forward_energy), dim = 2)
        forward_energy[:, 0, 0] = 1.
        
        #create forward attention matrix
        for t in range(1, trg_len + 1):
            for s in range(1, src_len + 1):
                forward_energy[:, t, s] *= (forward_energy[:, t-1, s-1] \
                                            + forward_energy[:, t-1, s] \
                                            + forward_energy[:, t, s-1])
        
        #forward_attention remove the zero padding
        forward_energy = forward_energy[:, 1:, 1:]
        
        if mask is not None:
            forward_energy += (1 - mask) * self.lower_bound
        
        #compute softmax
        forward_attention = F.softmax(forward_energy, dim = 2)

        return forward_attention
    
    
    def faster_forward_attention(self, attention, mask=None):
        
        #attention = [batch size, trg len, src len]
        batch_size = attention.shape[0]
        r, c = attention.shape[1] - 1, attention.shape[2] - 1
        
        #create a matrix of zeros
        cummulate_matrix = torch.zeros(batch_size, r+1, c+1).to(self.device)
        cummulate_matrix[:, 0, 0] = attention[:, 0, 0]
        
        for i in range(1, r+c):
            I = np.arange(max(0, i-c), min(r, i))
            J = I[::-1] + i - min(r, i) - max(0, i-c)
            cummulate_matrix[:, I+1, J+1] = attention[:, I+1, J+1] * (attention[:, I, J] \
                                                                      + attention[:, I, J+1] \
                                                                          + attention[:, I+1, J])
        
        #apply masking operation
        if mask is not None:
            cummulate_matrix += (1 - mask) * self.lower_bound
        
        #forward_attention using softmax
        forward_attention = F.softmax(cummulate_matrix, dim = 2)

        return forward_attention


    def calculate_attention(self, embedded, conved, encoder_conved, 
                            encoder_embed, mask=None):

        #embedded = [batch size, trg len, emb dim]
        #conved = [batch size, hid dim, trg len]
        #encoder_conved = [batch size, src len, emb dim]


        #permute and convert back to emb dim
        #conved_emb = [batch size, trg len, emb dim]
        conved_emb = self.activation(self.attn_hid2emb(conved.permute(0, 2, 1)))
        # conved_emb = (conved_emb + embedded) * self.scale


        #energy = [batch size, trg len, src len]
        energy = torch.matmul(conved_emb, encoder_conved.permute(0, 2, 1))


        if mask is not None:
            energy += (1 - mask) * self.lower_bound


        #attention = [batch size, trg len, src len]        
        attention = F.softmax(energy, dim=2)


        #compute forward attention
        attention = self.faster_forward_attention(attention, mask=mask)


        #attended_encoding = [batch size, trg len, emd dim]
        attended_encoding = torch.matmul(attention, encoder_embed)

        
        #convert from emb dim -> hid dim
        #attended_encoding = [batch size, trg len, hid dim]
        attended_encoding = self.activation(self.attn_emb2hid(attended_encoding))


        #attended_encoding = [batch size, hid dim, trg len]
        attended_encoding = attended_encoding.permute(0, 2, 1)

        
        #apply residual connection
        attended_combined = (conved + attended_encoding) * self.scale


        return attention, attended_combined
        
    def forward(self, src, trg, encoder_conved, encoder_embed, mask=None):
        
        #trg = [batch size, trg len, emb dim]
        #src = [batch size, src len, emb dim]
        #encoder_conved = encoder_conved_embed = [batch size, src len, emb dim]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]


        #create position tensor
        #pos = [0, 1, 2, 3, ..., src len - 1]
        #pos = [batch size, src len]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)


        #combine embeddings by elementwise summing
        #embedded = [batch size, trg len, emb dim]
        # embedded = self.dropout(trg + pos_embedded)
        embedded = trg + self.pos_embedding(pos)

        
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
                                                         encoder_embed, 
                                                         mask=mask)


            #add input to form residual connection
            conved = (conved + conv_input) * self.scale

            
            #attention over the input sequence
            #set conv_input to conved for next loop iteration
            # attended_embed = torch.matmul(attention, encoder_embed)
            # attended_embed_hid = self.activation(self.emb2hid(attended_embed))
            # conved = (conved + attended_embed_hid.permute(0, 2, 1))  * self.scale


            #close the feedback loop by setting output as input
            conv_input = conved


        #conved = [batch size, trg len, emb dim]
        conved = conved.permute(0, 2, 1)


        #output = [batch size, trg len, output dim]
        attended_src = torch.matmul(attention, src)
        output = (self.fc_out(self.dropout(conved)) + attended_src) * self.scale
        # output = attended_src


        return output, attention