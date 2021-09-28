#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 13:24:19 2021

@author: ravi
"""

#%% Loading all required packages

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

import sys
import random
import math
import time
import pickle
import joblib
import pylab
import logging

from load_data import LoadData
from utils import count_parameters, epoch_time

#%% Set the random seeds for reproducability.
SEED = 100 #1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

#%% Initialization of the GRU weights
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

# We use a simplified version of the weight initialization scheme used 
# in the paper. Here, we will initialize all biases to zero and all 
# weights from $\mathcal{N}(0, 0.01)$.

#%% Encoder model definition
class Encoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, 
                 dropout, device):
        super().__init__()
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.len_dropout = nn.Dropout(0.9)
        
        #length prediction
        self.length_linear = nn.Linear(enc_hid_dim*2, 1)
        self.length_activation = nn.Sigmoid()

        self.temp_scale = torch.sqrt(torch.FloatTensor([0.0067])).to(device)
        
    def forward(self, src):
        
        #src = [src len, batch size, emb_dim]
        #embedded = [src len, batch size, emb dim]        
        embedded = self.dropout(src)

        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]        
        outputs, hidden = self.rnn(embedded)
        
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        length_pred_embed = outputs.permute(1, 2, 0)
        encoder_sum = torch.sum(length_pred_embed, dim = 2) * self.temp_scale #mean
        len_pred = self.length_linear(self.len_dropout(encoder_sum))
        
        return outputs, hidden, len_pred

#%%  Attention model definition
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat decoder hidden state src_len times
        #hidden = [batch size, src len, dec hid dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        #encoder_outputs = [batch size, src len, enc hid dim * 2]        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        #energy = [batch size, src len, dec hid dim]        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))

        #attention= [batch size, src len]
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)

#%% Decoder definition
class Decoder(nn.Module):
    def __init__(self, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.attention = attention
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        # self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, emb_dim)
        self.fc_out = nn.Linear(dec_hid_dim, emb_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
             
        #input = [1, batch size, emb_dim]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]

        #embedded = [1, batch size, emb dim]
        embedded = self.dropout(input)

        #a = [batch size, src len]
        a = self.attention(hidden, encoder_outputs)

        #a = [batch size, 1, src len]        
        a = a.unsqueeze(1)

        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        #weighted = [batch size, 1, enc hid dim * 2]        
        weighted = torch.bmm(a, encoder_outputs)

        #weighted = [1, batch size, enc hid dim * 2]        
        weighted = weighted.permute(1, 0, 2)

        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]        
        rnn_input = torch.cat((embedded, weighted), dim = 2)

        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()

        #prediction = [1, batch size, emb dim]        
        # prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 2))
        prediction = self.fc_out(output)
        
        return prediction, hidden.squeeze(0), a

#%% Building the seq2seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, maxlen, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.maxlen = maxlen
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.8):
        
        #src = [src len, batch size, emb_dim]
        #trg = [trg len, batch size, emb_dim]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        src_len = src.shape[0]
        emb_dim = trg.shape[2]
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, emb_dim).to(self.device)
        
        #tensor to store attention
        attention = torch.zeros(batch_size, 1, src_len).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden, pred_len = self.encoder(src)
                
        #first input to the decoder is the <sos> tokens
        input = trg[0:1,:,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden, attn = self.decoder(input, hidden, encoder_outputs)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t:t+1] = output
            
            #place attn in attention matrix
            attention = torch.cat((attention, attn), dim = 1)
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t:t+1] if teacher_force else output

        return outputs[1:,:,:], attention[:,1:,:], pred_len

    def ar_decode(self, src, sos_token):
        
        #src = [src_len, batch_size, emb_dim]
        #sos_token = [1, batch_size, emb_dim]
        #teacher forcing ratio will be 1 here
        
        self.eval()
        
        with torch.no_grad():

            batch_size = src.shape[1]
            src_len = src.shape[0]
            emb_dim = src.shape[2]
            
            #tensor to store decoder outputs
            outputs = torch.zeros(1, batch_size, emb_dim).to(self.device)
            
            #tensor to store attention
            attention = torch.zeros(batch_size, 1, src_len).to(self.device)
            
            #encoder_outputs and hidden
            #converted pred_len from ratio to #frames
            encoder_outputs, hidden, pred_len = self.encoder(src)
            
            trg_len = min(self.maxlen, int(src_len*pred_len))
            
            input = sos_token
            
            for t in range(1, trg_len):
                
                #insert input token embedding, previous hidden state and all encoder hidden states
                #receive output tensor (predictions) and new hidden state
                output, hidden, attn = self.decoder(input, hidden, encoder_outputs)
                
                #place predictions in a tensor holding predictions for each token
                outputs[t:t+1] = output
                
                #place attn in attention matrix
                attention = torch.cat((attention, attn), dim = 1)

                #no teacher forcing, use predicted token
                input = output

            return outputs[1:,:,:], attention[:,1:,:]

#%% Training and evaluation function definition
def train(model, iterator, optimizer, criterion, clip, 
          pad_signature, len_loss_wt):
    
    model.train()
    
    epoch_reg_loss  = []
    epoch_len_loss  = []
    epoch_loss      = []
    
    generated_seqs = []
    generated_attn = []
    
    for i in range(iterator.batch_count()):
        
        #src = [batch size, src len, emb_dim]
        #tar = [batch size, tar len, emb_dim]

        src, trg, inp_seq_len, out_seq_len, _ = iterator[i]
        
        #create target for sequence length prediction

        seq_len_trg = (out_seq_len - 1) / (inp_seq_len - 1)
        
        #src = [src len, batch size, emb_dim]
        #tar = [tar len, batch size, emb_dim]

        src = src.permute(1, 0, 2)
        trg = trg.permute(1, 0, 2)
        
        optimizer.zero_grad()

        #output = [(trg len - 1), batch size, emb dim]        

        output, attn, pred_len = model(src, trg, 0.5)
        
        batch_size = output.shape[1]
        output_dim = output.shape[-1]
        
        #trg = [batch size, (trg len - 1), emb_dim]
        #output = [batch size, (trg len - 1), emb_dim]

        trg = trg[1:,:,:].permute(1, 0, 2)
        trg_len = output.shape[1]
        output = output.permute(1, 0, 2)
        
        nonpadding = torch.ne(torch.sum(trg, dim=-1), pad_signature)
        nonpadding = nonpadding.type(torch.float32)
            
        reg_loss = torch.sum(torch.sum(criterion(output, trg), 
                        dim = -1) * nonpadding) / (torch.sum(nonpadding) + 1e-7)
        len_loss = torch.mean(criterion(seq_len_trg, pred_len))
        
        loss = reg_loss + len_loss_wt*len_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_reg_loss.append(batch_size*reg_loss.item())
        epoch_len_loss.append(batch_size*len_loss.item())
        epoch_loss.append(batch_size*loss.item())

        generated_seqs.append(output.cpu().detach().numpy())
        generated_attn.append(attn.cpu().detach().numpy())
        
    return epoch_reg_loss, epoch_len_loss, \
        epoch_loss, generated_seqs, generated_attn


# ...and the evaluation loop, remembering to set the model to `eval` mode and turn off teaching forcing.


def evaluate(model, iterator, criterion, pad_signature, len_loss_wt):
    
    model.eval()
    
    epoch_reg_loss  = []
    epoch_len_loss  = []
    epoch_loss      = []

    generated_seqs = []
    generated_attn = []

    with torch.no_grad():
    
        for i in range(iterator.batch_count()):
        
            #src = [batch size, src len, emb_dim]
            #tar = [batch size, tar len, emb_dim]

            src, trg, inp_seq_len, out_seq_len, _ = iterator[i]
            
            #create target for sequence length prediction

            seq_len_trg = (out_seq_len - 1) / (inp_seq_len - 1)

            #src = [src len, batch size, emb_dim]
            #tar = [tar len, batch size, emb_dim]

            src = src.permute(1, 0, 2)
            trg = trg.permute(1, 0, 2)

            #output = [batch size, (trg len - 1), emb dim]

            output, attn, pred_len = model(src, trg, 0) #turn off teacher forcing

            batch_size = output.shape[1]
            output_dim = output.shape[-1]

            #trg = [batch size, (trg len - 1), emb dim]
            #output = [batch size, (trg len - 1), emb dim]

            trg = trg[1:,:,:].permute(1, 0, 2)
            output = output.permute(1, 0, 2)
            
            nonpadding = torch.ne(torch.sum(trg, dim=-1), pad_signature)
            nonpadding = nonpadding.type(torch.float32)
                
            reg_loss = torch.sum(torch.sum(criterion(output, trg), 
                        dim=-1) * nonpadding) / (torch.sum(nonpadding) + 1e-7)
            len_loss = torch.mean(criterion(seq_len_trg, pred_len))
            
            loss = reg_loss + len_loss_wt*len_loss

            epoch_reg_loss.append(batch_size*reg_loss.item())
            epoch_len_loss.append(batch_size*len_loss.item())
            epoch_loss.append(batch_size*loss.item())

            generated_seqs.append(output.cpu().detach().numpy())
            generated_attn.append(attn.cpu().detach().numpy())
        
    return epoch_reg_loss, epoch_len_loss, \
        epoch_loss, generated_seqs, generated_attn

if __name__ == '__main__':
    #%% Defining hyperparameters of the model
    BATCH_SIZE = 2  #128
    EMB_DIM = 80
    ENC_HID_DIM = 64 #512
    DEC_HID_DIM = 64 #512
    ENC_DROPOUT = 0.2
    DEC_DROPOUT = 0.2
    LEARNING_RATE = 0.00001
    PAD_IDX = 10
    MAXLEN = 1400
    PAD_SIGNATURE = PAD_IDX * EMB_DIM
    N_EPOCHS = 20
    CLIP = 0.1
    LEN_LOSS_WT = 5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('running on device: ', device)
    
    #%% Generating train, valid and test iterator
    train_data_iterator = LoadData(pkl_file='./data/VESUS/train_neutral_sad_world_mvn_5ms.pkl', 
                                    batch_size=BATCH_SIZE, device=device, 
                                    augment=True, padwith=PAD_IDX)
    valid_data_iterator = LoadData(pkl_file='./data/VESUS/valid_neutral_sad_world_mvn_5ms.pkl', 
                                    batch_size=1, device=device, 
                                    augment=False, padwith=PAD_IDX)
    
    print("Number of batches: {}".format(train_data_iterator.batch_count()))
    
    #%% Create dataloaders and model
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT, device)
    dec = Decoder(EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    
    model = Seq2Seq(enc, dec, MAXLEN, device).to(device)
    model.load_state_dict(torch.load('./models/CMU/gru-vc-model.pt'))
    #model.apply(init_weights)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    #%% Optimizer and loss criterion
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss(reduction='none')
    
    #%% Training the model
    best_valid_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
        
        _, train_len_loss, train_loss, train_seqs, train_attn = train(model, train_data_iterator, 
                                                   optimizer, criterion, 
                                                   CLIP, PAD_SIGNATURE, LEN_LOSS_WT)
        _, valid_len_loss, valid_loss, valid_seqs, valid_attn = evaluate(model, valid_data_iterator, 
                                                      criterion, PAD_SIGNATURE, LEN_LOSS_WT)
    
        end_time = time.time()
    
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if np.mean(valid_loss) < best_valid_loss:
            best_valid_loss = np.mean(valid_loss)
            torch.save(model.state_dict(), './models/VESUS/gru-neutral-sad-vesus-model.pt')
        
        train_len_loss = np.mean(train_len_loss)
        train_loss = np.mean(train_loss)
        valid_len_loss = np.mean(valid_len_loss)
        valid_loss = np.mean(valid_loss)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\t |Train Length Loss: {train_len_loss:.6f} \t Train Loss: {train_loss:.6f}|')
        print(f'\t |Val. Length Loss: {valid_len_loss:.6f} \t Val. Loss: {valid_loss:.6f}|')
        
        sys.stdout.flush()
    
    #%% Testing
    # model.load_state_dict(torch.load('./models/VESUS/gru-neutral-angry-vesus-model.pt'))
    
    # _, test_len_loss, test_loss, test_seqs, test_attn = evaluate(model, test_data_iterator, criterion, PAD_SIGNATURE)
    # with open('./cmu_test_pred.pkl', 'wb') as f:
    #     joblib.dump({'test_len_loss':test_len_loss, 'test_loss':test_loss}, f)
    
    # print(f'|Test Length Loss: {np.mean(test_len_loss):.6f} Test Loss: {np.mean(test_loss):.6f} |')
