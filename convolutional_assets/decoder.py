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
from utils import positional_encoding

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
                  pos_enc=False, 
                  max_length = 150):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.pad_idx = pad_idx
        self.device = device
        self.max_length = max_length
        self.lower_bound = -2**32 + 1
        self.pad_vector = pad_vector.to(device)
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        if pos_enc:
            self.pos_embedding = positional_encoding(max_length, emb_dim)
        else:
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
      
    def calculate_attention(self, embedded, conved, encoder_conved, 
                            encoder_embed, mask=None):

        #embedded = [batch size, trg len, emb dim]
        #conved = [batch size, hid dim, trg len]
        #encoder_conved = encoder_embed = [batch size, src len, emb dim]


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
        
    def forward(self, src, trg, encoder_conved, encoder_embed, 
                sample=False, mask=None):
        
        #trg = [batch size, trg len, emb dim]
        #src = [batch size, src len, emb dim]
        #encoder_conved = encoder_embed = [batch size, src len, emb dim]
                
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
        pad_projected = self.activation(self.emb2hid(self.pad_vector))
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


        #multinomial sampling
        #output = [batch size, trg len, output dim]
        #sample = [batch size, trg len, 1]
        if sample:
            # If max value should be picked
            sampling_max = torch.argmax(attention, 2).unsqueeze(2)
            sampling_max = sampling_max.repeat((1, 1, src.shape[2]))
            
            # If actual sampling should happen
            sample_object = torch.distributions.categorical.Categorical(probs=attention)
            sampling_act = sample_object.sample(sample_shape=[1]).permute(1, 2, 0)
            sampling_act = sampling_act.repeat((1, 1, src.shape[2]))

            sampling_mode = sampling_act
            attended_src = torch.gather(src, 1, sampling_mode)
            output = (self.fc_out(self.dropout(conved)) + attended_src) * self.scale
            # output = attended_src

        #output = [batch size, trg len, output dim]
        else:
            attended_src = torch.matmul(attention, src)
            output = (self.fc_out(self.dropout(conved)) + attended_src) * self.scale
            # output = attended_src


        return output, attention


#%% Balanced Decoder with attention
# class Decoder(nn.Module):
#     def __init__(self, 
#                  emb_dim, 
#                  hid_dim, 
#                  n_layers, 
#                  kernel_size, 
#                  dropout, 
#                  pad_idx, 
#                  pad_vector, 
#                  device,
#                  pos_enc=False, 
#                  max_length = 150):
#         super().__init__()
        
#         self.kernel_size = kernel_size
#         self.pad_idx = pad_idx
#         self.device = device
#         self.max_length = max_length
#         self.lower_bound = -2**32 + 1
#         self.pad_vector = pad_vector.to(device)
        
#         self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

#         if pos_enc:
#             self.pos_embedding = positional_encoding(max_length, emb_dim)
#         else:
#             self.pos_embedding = nn.Embedding(max_length, emb_dim)
        
#         self.emb2hid = nn.Linear(emb_dim, hid_dim)
#         self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
#         self.attn_hid2emb = nn.Linear(hid_dim, emb_dim)
#         self.attn_emb2hid = nn.Linear(emb_dim, hid_dim)

#         self.fc_out = nn.Linear(hid_dim, emb_dim)
        
#         self.activation = nn.Tanh()
        
#         self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
#                                               out_channels = 2 * hid_dim, 
#                                               kernel_size = kernel_size)
#                                     for _ in range(n_layers)])
        
#         self.dropout = nn.Dropout(dropout)
      
#     def calculate_attention(self, conv_input, conved, encoder_conved, 
#                             encoder_embed, mask=None):

#         #conv_input = [batch size, hid dim, trg len]
#         #conved = [batch size, hid dim, trg len]
#         #encoder_conved/encoder_embed = [batch size, src len, emb dim]


#         #permute and convert back to emb dim
#         #conved_emb = [batch size, trg len, emb dim]
#         conved_emb = (conved + conv_input) * self.scale
#         conved_emb = self.activation(self.attn_hid2emb(conved_emb.permute(0, 2, 1)))
        

#         #energy = [batch size, trg len, src len]
#         energy = torch.matmul(conved_emb, encoder_conved.permute(0, 2, 1))


#         if mask is not None:
#             energy += (1 - mask) * self.lower_bound


#         #attention = [batch size, trg len, src len]        
#         attention = F.softmax(energy, dim=2)


#         #attended_encoding = [batch size, trg len, emd dim]
#         attended_encoding = torch.matmul(attention, encoder_embed)

        
#         #convert from emb dim -> hid dim
#         #attended_encoding = [batch size, trg len, hid dim]
#         attended_encoding = self.activation(self.attn_emb2hid(attended_encoding))


#         #attended_encoding = [batch size, hid dim, trg len]
#         attended_encoding = attended_encoding.permute(0, 2, 1)


#         return attention, attended_encoding
        
#     def forward(self, src, trg, encoder_conved, encoder_embed, mask=None):
        
#         #trg = [batch size, trg len, emb dim]
#         #src = [batch size, src len, emb dim]
#         #encoder_conved = encoder_conved_embed = [batch size, src len, emb dim]
                
#         batch_size = trg.shape[0]
#         trg_len = trg.shape[1]


#         #create position tensor
#         #pos = [0, 1, 2, 3, ..., src len - 1]
#         #pos = [batch size, src len]
#         pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)


#         #combine embeddings by elementwise summing
#         #embedded = [batch size, trg len, emb dim]
#         # embedded = self.dropout(trg + pos_embedded)
#         embedded = trg + self.pos_embedding(pos)

        
#         #pass embedded through linear layer to go through emb dim -> hid dim
#         #conv_input = [batch size, trg len, hid dim]
#         conv_input = self.activation(self.emb2hid(embedded))


#         #pad vector needs to be projected as well
#         pad_projected = self.activation(self.emb2hid(self.pad_vector))
#         pad_projected = pad_projected.permute(0, 2, 1)

        
#         #permute for convolutional layer
#         #conv_input = [batch size, hid dim, trg len]
#         conv_input = conv_input.permute(0, 2, 1)

        
#         batch_size = conv_input.shape[0]
#         hid_dim = conv_input.shape[1]

        
#         for i, conv in enumerate(self.convs):
        
#             #apply dropout
#             conv_input = self.dropout(conv_input)

        
#             #need to pad so decoder can't "cheat"
#             padding = pad_projected.repeat(batch_size, 1, self.kernel_size - 1)


#             #padded_conv_input = [batch size, hid dim, trg len + kernel size - 1]
#             padded_conv_input = torch.cat((padding, conv_input), dim = 2)

        
#             #pass through convolutional layer
#             #conved = [batch size, 2 * hid dim, trg len]
#             conved = conv(padded_conv_input)

            
#             #pass through GLU activation function
#             #conved = [batch size, hid dim, trg len]
#             conved = F.glu(conved, dim = 1)

            
#             #calculate attention
#             #attention = [batch size, trg len, src len]
#             #conved = [batch size, hid dim, trg len]
#             attention, conved = self.calculate_attention(conv_input, 
#                                                          conved, 
#                                                          encoder_conved, 
#                                                          encoder_embed, 
#                                                          mask=mask)


#             #close the feedback loop by setting output as input
#             conv_input = conved


#         #conved = [batch size, trg len, emb dim]
#         conved = conved.permute(0, 2, 1)


#         #output = [batch size, trg len, output dim]
#         attended_src = torch.matmul(attention, src)
#         output = self.fc_out(self.dropout(conved))
#         # output = attended_src


#         return output, attention
