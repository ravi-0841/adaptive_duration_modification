#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 14:31:15 2021

@author: ravi
"""

#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

import random
import math
import time
import pickle
import pylab

from encoder_nonlinear import Encoder
from decoder_nonlinear import Decoder
from seq2seq_nonlinear import Seq2Seq
from load_data import LoadData
from utils import count_parameters, epoch_time, create_attn_mask
from train_evaluate_nonlinear import TrainingEval


#%% Define all hyperparameters and constants
SEED = 123 #123
EMB_DIM = 64
HID_DIM = 256 #32 each conv. layer has 2 * hid_dim filters
ENC_LAYERS = 6 #6 number of conv. blocks in encoder
DEC_LAYERS = 6 #6 number of conv. blocks in decoder
LEARNING_RATE = 0.0001 #0.0001 initial learning rate for Adam optimizer
ENC_KERNEL_SIZE = 5 #5 must be odd!
DEC_KERNEL_SIZE = 5 #5 can be even or odd
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2
BATCH_SIZE = 8 #8
PAD_IDX = 10
MAXLEN = 350
PAD_SIGNATURE = PAD_IDX * EMB_DIM
PAD_VECTOR = torch.randn(1, 1, EMB_DIM)
N_EPOCHS = 1000  #100
CLIP = 10  #0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
attn_matrix = create_attn_mask([1]*11, MAXLEN, device)


#%% Setting seed for reproducibility
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True


#%% Defining encoder, decoder and optimization criterion
enc = Encoder(EMB_DIM, HID_DIM, ENC_LAYERS, 
              ENC_KERNEL_SIZE, ENC_DROPOUT, 
              device, MAXLEN)
dec = Decoder(EMB_DIM, HID_DIM, DEC_LAYERS, 
              DEC_KERNEL_SIZE, DEC_DROPOUT, 
              PAD_IDX, PAD_VECTOR, device, None, MAXLEN)

model = Seq2Seq(enc, dec).to(device)
print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.L1Loss(reduction='none')

#%% Generating train, valid and test iterator
train_data_iterator = LoadData(pkl_file='./data/train_neutral_angry_mvn.pkl', 
                                batch_size=BATCH_SIZE, device=device, 
                                augment=True, padwith=PAD_IDX)
valid_data_iterator = LoadData(pkl_file='./data/valid_neutral_angry_mvn.pkl', 
                                batch_size=1, device=device, 
                                augment=False, padwith=PAD_IDX)
test_data_iterator = LoadData(pkl_file='./data/test_neutral_angry_mvn.pkl', 
                                batch_size=1, device=device, 
                                augment=False, padwith=PAD_IDX)

print("Number of batches: {}".format(train_data_iterator.batch_count()))


#%% Class which trains, evaluates and decodes the model
training_evaluation = TrainingEval(EMB_DIM, model, optimizer, 
                                   criterion, device, CLIP, 
                                   MAXLEN, PAD_SIGNATURE, False)

# training_evaluation.model.load_state_dict(torch.load('model_nonlinear.pt'))

#%% Computing validation loss and saving model
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    try:
        train_loss, train_gen_seqs = training_evaluation.train(train_data_iterator)
        valid_loss, valid_gen_seqs = training_evaluation.evaluate(valid_data_iterator)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model_nonlinear.pt')

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.6f} | Val. Loss: {valid_loss:.6f}')
    except Exception as ex:
        print(ex)

    train_data_iterator.shuffle_data()


#%% Generating samples from test data auto-regressively
training_evaluation.model.load_state_dict(torch.load('model_nonlinear.pt'))

for i in range(10):
    q = np.random.randint(len(test_data_iterator))
    prediction, target, attention = training_evaluation.ar_decode(test_data_iterator, q, device)
    pylab.figure(figsize=(11,11))
    pylab.subplot(311), pylab.imshow(prediction.squeeze().T), pylab.title("Predicted {}".format(q))
    pylab.subplot(312), pylab.imshow(attention.squeeze().T), pylab.title("Attention")
    pylab.subplot(313), pylab.imshow(target.squeeze().T), pylab.title("Target")
    pylab.savefig('/home/ravi/Desktop/conv_seq2seq_test_{0}'.format(q))
    pylab.close()
