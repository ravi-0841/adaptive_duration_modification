#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 23:35:27 2021

@author: ravi
"""

#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

#%%
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg, sample=False, mask=None):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len - 1] (<eos> token sliced off the end)
           
        #calculate z^u (encoder_conved) and (z^u + e) (encoder_combined)
        #encoder_conved is output from final encoder conv. block
        #encoder_combined is encoder_conved plus (elementwise) src embedding plus 
        #positional embeddings
        #encoder_conved = [batch size, src len, emb dim]
        #encoder_combined = [batch size, src len, emb dim]
        pred_len, encoder_conved, encoder_conved_embed = self.encoder(src)

        
        #calculate prediction of next frame
        #output is a batch of predictions for each frame in the trg utterance
        #attention a batch of attention scores across the src utterance for 
        #each frame in the trg utterance
        #output = [batch size, trg len - 1, ouput dim]
        #attention = [batch size, trg len - 1, src len]
        output, attention = self.decoder(src, trg, encoder_conved, 
                                         encoder_conved_embed, 
                                         sample=sample, 
                                         mask=mask)

        
        return pred_len, output, attention
