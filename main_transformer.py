#!/usr/bin/env python
# coding: utf-8

#%%
import torch
import time
import pylab
import sys
import random
import argparse
import os
import pickle

import numpy as np

from transformer_assets.encoder import Encoder
from transformer_assets.decoder import Decoder
from transformer_assets.seq2seq import Seq2Seq
from transformer_assets.train_evaluate import TrainingEval
from itakura_parallelogram import ItakuraParallelogram
from load_data import LoadData

from utils import count_parameters, epoch_time

#%%
# SEED = 1234

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

#%%

def train_test_model(args):

    EMBED_DIM       = args.embed_dim
    HIDDEN_DIM      = args.hidden_dim
    ENC_LAYERS      = args.encoder_layers
    DEC_LAYERS      = args.decoder_layers
    ENC_HEADS       = args.encoder_heads
    DEC_HEADS       = args.decoder_heads
    ENC_PF_DIM      = args.encoder_pf_dim
    DEC_PF_DIM      = args.decoder_pf_dim
    ENC_DROPOUT     = args.encoder_dropout
    DEC_DROPOUT     = args.decoder_dropout
    LEARNING_RATE   = args.learning_rate
    PAD_IDX         = args.pad_with
    SLOPE           = args.itakura_slope
    MAXLEN          = args.maxlen
    N_EPOCHS        = args.num_epochs
    CLIP            = args.grad_clip
    BATCH_SIZE      = args.batch_size
    SINUSOID        = args.sinusoid_encoding
    
    MODEL_DIR       = args.model_dir
    MODEL_NAME      = args.model_name
    
    PAD_SIGNATURE   = PAD_IDX * EMBED_DIM

    MODEL_NAME      = "layers_{0}_hid_{1}-".format(ENC_LAYERS, HIDDEN_DIM) + MODEL_NAME
    model_saver     = os.path.join(MODEL_DIR, MODEL_NAME)
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#%%
    enc     = Encoder(EMBED_DIM, 
                      HIDDEN_DIM, 
                      ENC_LAYERS, 
                      ENC_HEADS, 
                      ENC_PF_DIM, 
                      ENC_DROPOUT, 
                      device, 
                      pos_encoding = SINUSOID, 
                      max_length = MAXLEN)
    
    dec     = Decoder(EMBED_DIM, 
                      HIDDEN_DIM, 
                      DEC_LAYERS, 
                      DEC_HEADS, 
                      DEC_PF_DIM, 
                      DEC_DROPOUT, 
                      device, 
                      pos_encoding = SINUSOID, 
                      max_length = MAXLEN)
    
    model   = Seq2Seq(enc, dec, PAD_SIGNATURE, device).to(device)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
#%%
    train_eval = TrainingEval(model, LEARNING_RATE, CLIP, 
                                    device, PAD_SIGNATURE, 
                                    MAXLEN)
    
#%%
#    train_data_iterator = LoadData(pkl_file='./data/CMU/cmu_train_world_mvn_5ms.pkl', 
#                                    slope = SLOPE, batch_size = BATCH_SIZE, device = device, 
#                                    augment = True, padwith = PAD_IDX, masking = False)
#    valid_data_iterator = LoadData(pkl_file='./data/CMU/cmu_valid_world_mvn_5ms_new.pkl', 
#                                    slope = SLOPE, batch_size = 1, device = device, 
#                                    augment = False, padwith = PAD_IDX, masking = False)
    train_data_iterator = LoadData(pkl_file='./data/VESUS/train_neutral_sad_world_mvn_5ms.pkl', 
                                    slope = SLOPE, batch_size = BATCH_SIZE, device = device, 
                                    augment = True, padwith = PAD_IDX, masking = False)
    valid_data_iterator = LoadData(pkl_file='./data/VESUS/valid_neutral_sad_world_mvn_5ms.pkl', 
                                    slope = SLOPE, batch_size = 1, device = device, 
                                    augment = False, padwith = PAD_IDX, masking = False)
    
    print("Number of training batches: {}".format(train_data_iterator.batch_count()))
    
    
#%%
    train_eval.model.load_state_dict(torch.load("./models/CMU/layers_{0}_hid_{1}-cmu-nomask-transformer-model-5ms.pt".format(ENC_LAYERS, HIDDEN_DIM)))
    best_valid_loss = float('inf')
    
    for epoch in range(N_EPOCHS):
        
        start_time = time.time()
    
        train_reg_loss, train_len_loss, train_loss, \
            train_gen_seqs, train_gen_lens = train_eval.train(train_data_iterator)
        valid_reg_loss, valid_len_loss, valid_loss, \
            valid_gen_seqs, valid_gen_lens = train_eval.evaluate(valid_data_iterator)
        
#        encoder_wts = train_eval.model.encoder.layers[0].self_attention.fc_k.weight.cpu().detach().numpy()
#        encoder_grd = train_eval.model.encoder.layers[0].self_attention.fc_k.weight.grad.cpu().detach().numpy()
#        
#        pylab.figure()
#        pylab.subplot(211), pylab.hist(encoder_wts.flatten(), bins=100, density=True)
#        pylab.title('Encoder layer 1 fc_k weights, Epoch- {}'.format(epoch+1))
#        pylab.subplot(212), pylab.hist(encoder_grd.flatten(), bins=100, density=True)
#        pylab.title('Encoder layer 1 fc_k grads, Epoch- {}'.format(epoch+1))
#        pylab.savefig('./logs/epoch_{}'.format(epoch+1))
#        pylab.close()
    
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_saver)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Reg. Loss: {train_reg_loss:.6f} | Valid Reg. Loss: {valid_reg_loss:.6f}')
        print(f'\tTrain Len. Loss: {train_len_loss:.6f} | Valid Len. Loss: {valid_len_loss:.6f}')
        sys.stdout.flush()
    
#%%
    train_eval.model.load_state_dict(torch.load(model_saver))
    
    # create itakura object
    itakura_object = ItakuraParallelogram()

    output_dir = "./outputs/VESUS/transformer/vesus-neutral-sad-nomask-layers-{0}-hid-{1}-5ms".format(ENC_LAYERS, HIDDEN_DIM)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # create the directory to write png files
    for i in range(len(valid_data_iterator)):
        q = i  #np.random.randint(len(valid_data_iterator))
        prediction, target, attention = train_eval.ar_decode(valid_data_iterator, 
                                                              q, itakura_object = itakura_object)
    
        a = 1 / (attention.squeeze() + 1e-12)
        acc_mat = itakura_object.accumulated_cost_matrix(a)
        path = itakura_object.return_constrained_path(acc_mat, steps_limit=1)
    
        pylab.figure(figsize=(6.5,9.5))
        pylab.subplot(221), pylab.imshow(prediction.squeeze().T), pylab.title("Predicted {}".format(q))
        pylab.subplot(222), pylab.imshow(attention.squeeze().T), pylab.title("Attention")
        pylab.subplot(223), pylab.imshow(target.squeeze().T), pylab.title("Target")
        pylab.subplot(224), pylab.imshow(np.log10(attention.squeeze() + 1e-10))
        pylab.plot(path[0,:], path[1,:], 'r-'), pylab.title("Attention with path")
        pylab.savefig(os.path.join(output_dir, "valid_{0}".format(q)))
        pylab.close()

#%%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Argument parser for hyperparameters")
    
    parser.add_argument("--embed_dim", type = int, help = "Embedding dimension or feature dimension", default = 80)
    parser.add_argument("--hidden_dim", type = int, help = "Projection dimension", default = 256)
    parser.add_argument("--encoder_layers", type = int, help = "Number of encoder layers", default = 2)
    parser.add_argument("--decoder_layers", type = int, help = "Number of decoder layers", default = 2)
    parser.add_argument("--encoder_heads", type = int, help = "Number of attention heads in encoder", default = 4)
    parser.add_argument("--decoder_heads", type = int, help = "Number of attention heads in decoder", default = 4)
    parser.add_argument("--encoder_pf_dim", type = int, help = "Pointwise feedforward dimension in encoder", default = 256)
    parser.add_argument("--decoder_pf_dim", type = int, help = "Pointwise feedforward dimension in decoder", default = 256)
    parser.add_argument("--encoder_dropout", type = float, help = "Dropout in encoder", default = 0.2)
    parser.add_argument("--decoder_dropout", type = float, help = "Dropout in decoder", default = 0.2)
    parser.add_argument("--learning_rate", type = float, help = "Optimizer learning rate", default = 0.0001) #0.0001 for CMU and 0.00001 for VESUS
    parser.add_argument("--pad_with", type = int, help = "Padding number", default = 10)
    parser.add_argument("--itakura_slope", type = float, help = "slope of the itakura mask", default = 1.25)
    parser.add_argument("--maxlen", type = int, help = "Maximum length of sequence", default = 1400)
    parser.add_argument("--num_epochs", type = int, help = "Number of training epochs", default = 100)
    parser.add_argument("--batch_size", type = int, help = "Minibatch size", default = 4)
    parser.add_argument("--grad_clip", type = float, help = "Gradient clip value", default = 0.1)
    parser.add_argument("--sinusoid_encoding", type = bool, help = "Use sinusoidal encoding or not", default = True)
    parser.add_argument("--model_dir", type = str, help = "model directory", default = "./models/VESUS")
    parser.add_argument("--model_name", type = str, help = "model name", default = "vesus-neutral-sad-nomask-transformer-model-5ms.pt")
    
    args = parser.parse_args()
    
    train_test_model(args)























