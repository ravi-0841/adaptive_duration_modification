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

import numpy as np

import os
import time
import pylab
import pickle
import sys
import random
import argparse

from convolutional_assets.encoder_residual import Encoder
from convolutional_assets.decoder_residual import Decoder
from convolutional_assets.seq2seq_residual import Seq2Seq
from convolutional_assets.train_only_encoder import TrainingEval

from load_data import LoadData
from utils import count_parameters, epoch_time
from itakura_parallelogram import ItakuraParallelogram

#%% Define all hyperparameters and constants
def train_eval_model(args):

    if os.path.exists("./data/pad_vector.pkl"):
        with open("./data/pad_vector.pkl", "rb") as f:
            pad_vec = pickle.load(f)
            pad_vec = pad_vec['pad_vec']
            f.close()
    else:
        pad_vec = np.random.randn(1,1,args.embed_dim)
        with open("./data/pad_vector.pkl", "wb") as f:
            pickle.dump({'pad_vec':pad_vec}, f)
            f.close()


    SEED            = 123
    EMO_PAIR        = args.emotion_pair
    EMBED_DIM       = args.embed_dim
    HIDDEN_DIM      = args.hidden_dim #each conv. layer has 2 * hid_dim filters
    ENC_LAYERS      = args.encoder_layers
    DEC_LAYERS      = args.decoder_layers
    LEARNING_RATE   = args.learning_rate
    ENC_KERNEL_SIZE = args.encoder_kernel
    DEC_KERNEL_SIZE = args.decoder_kernel
    ENC_DROPOUT     = args.encoder_dropout
    DEC_DROPOUT     = args.decoder_dropout
    SIN_ENCODE      = args.sinusoid_encoding
    BATCH_SIZE      = args.batch_size
    PAD_IDX         = args.pad_with
    MAXLEN          = args.maxlen
    SLOPE           = args.itakura_slope
    PAD_SIGNATURE   = PAD_IDX * EMBED_DIM
    PAD_VECTOR      = torch.from_numpy(pad_vec).float() #torch.randn(1, 1, EMBED_DIM)
    N_EPOCHS        = args.num_epochs
    GRAD_CLIP       = args.grad_clip
    
    MODEL_DIR       = os.path.join(args.model_dir, EMO_PAIR)
    MODEL_NAME      = args.model_name
    
    MODEL_NAME      = "layers_{0}_hid_{1}-".format(ENC_LAYERS, HIDDEN_DIM) + MODEL_NAME
    model_saver     = os.path.join(MODEL_DIR, MODEL_NAME)

    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('running on device: ', device)

#%% Setting seed for reproducibility
#    random.seed(SEED)
#    np.random.seed(SEED)
#    torch.manual_seed(SEED)
#    torch.cuda.manual_seed(SEED)
#    torch.backends.cudnn.deterministic = True

#%% Defining encoder, decoder and optimization criterion
    enc             = Encoder(EMBED_DIM, HIDDEN_DIM, ENC_LAYERS, 
                            ENC_KERNEL_SIZE, ENC_DROPOUT, 
                            device, SIN_ENCODE, MAXLEN)
    dec             = Decoder(EMBED_DIM, HIDDEN_DIM, DEC_LAYERS, 
                            DEC_KERNEL_SIZE, DEC_DROPOUT, 
                            PAD_IDX, PAD_VECTOR, device, 
                            SIN_ENCODE, MAXLEN)
    
    model           = Seq2Seq(enc, dec).to(device)

    train_eval      = TrainingEval(EMBED_DIM, model, LEARNING_RATE, 
                                       device, GRAD_CLIP, MAXLEN, 
                                       PAD_SIGNATURE, False)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    
#%% Generating train, valid and test iterator
    train_data_iterator = LoadData(pkl_file = './data/CMU/cmu_train_world_mvn_5ms.pkl', 
                                    batch_size=BATCH_SIZE, device=device, 
                                    slope=SLOPE, augment=True, padwith=PAD_IDX)
    valid_data_iterator = LoadData(pkl_file = './data/CMU/cmu_valid_world_mvn_5ms.pkl', 
                                    batch_size=1, device=device, 
                                    slope=SLOPE, augment=False, padwith=PAD_IDX)
#    train_data_iterator = LoadData(pkl_file = './data/VESUS/train_'+EMO_PAIR+'_world_mvn_5ms.pkl', 
#                                    batch_size=BATCH_SIZE, device=device, 
#                                    slope=SLOPE, augment=True, padwith=PAD_IDX)
#    valid_data_iterator = LoadData(pkl_file = './data/VESUS/valid_'+EMO_PAIR+'_world_mvn_5ms.pkl', 
#                                   batch_size=1, device=device, 
#                                    slope=SLOPE, augment=False, padwith=PAD_IDX)

    print("Number of batches: {}".format(train_data_iterator.batch_count()))

#%% Computing validation loss and saving model
    # train_eval.model.load_state_dict(torch.load("./models/CMU/soft_sampling/layers_{0}_hid_{1}-cmu-convolutional-model.pt".format(ENC_LAYERS, HIDDEN_DIM)))

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        try:
            train_loss, train_pred_len = train_eval.train(train_data_iterator)
            valid_loss, valid_pred_len = train_eval.evaluate(valid_data_iterator)
    
            end_time = time.time()
    
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_saver)
    
            print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Length Loss: {train_loss:.6f} | Valid Length Loss: {valid_loss:.6f}')
        except Exception as ex:
            print(ex)
    
        train_data_iterator.shuffle_data()
        sys.stdout.flush()


#%% Arguments list
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description = "Convolutional model parameters")
    
    parser.add_argument("--emotion_pair", type = str, help = "Emotion pair", default = "voice_conversion")
    parser.add_argument("--embed_dim", type = int, help = "Embedding dimension or feature dimension", default = 80)
    parser.add_argument("--hidden_dim", type = int, help = "Projection dimension", default = 256)
    parser.add_argument("--encoder_layers", type = int, help = "Number of encoder layers", default = 4)
    parser.add_argument("--decoder_layers", type = int, help = "Number of decoder layers", default = 4)
    parser.add_argument("--encoder_kernel", type = int, help = "Conv. kernel size in encoder", default = 3)
    parser.add_argument("--decoder_kernel", type = int, help = "Conv. kernel size in decoder", default = 3)
    parser.add_argument("--encoder_dropout", type = float, help = "Dropout in encoder", default = 0.2)
    parser.add_argument("--decoder_dropout", type = float, help = "Dropout in decoder", default = 0.2)
    parser.add_argument("--learning_rate", type = float, help = "Optimizer learning rate", default = 0.0001) #0.0001/CMU 0.00005/VESUS
    parser.add_argument("--pad_with", type = int, help = "Padding number", default = 10)
    parser.add_argument("--itakura_slope", type = float, help = "slope of the itakura mask", default = 1.25)
    parser.add_argument("--maxlen", type = int, help = "Maximum length of sequence", default = 1400)
    parser.add_argument("--num_epochs", type = int, help = "Number of training epochs", default = 500)
    parser.add_argument("--batch_size", type = int, help = "Minibatch size", default = 16) #8
    parser.add_argument("--grad_clip", type = float, help = "Gradient clip value", default = 0.1)
    parser.add_argument("--sinusoid_encoding", type = bool, help = "Use sinusoidal encoding or not", default = True)
    parser.add_argument("--model_dir", type = str, help = "model directory", default = "./models/CMU")
    # parser.add_argument("--model_name", type = str, help = "model name", default = "vesus-convolutional-model-5ms-sum-drop-9.pt")
    parser.add_argument("--model_name", type = str, help = "model name", default = "cmu-only-encoder.pt")

    args = parser.parse_args()

    train_eval_model(args)






























