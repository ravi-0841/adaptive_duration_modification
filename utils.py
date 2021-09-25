# -*- coding: utf-8 -*-
# /usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Utility functions
'''

import numpy as np
import torch
import torch.nn as nn


def shuffle_data(data_dict):
    idx = np.random.permutation(np.arange(0,len(data_dict['encoder_input_seqs']),1))
    for (k,v) in data_dict.items():
        shuff_v = [v[i] for i in idx]
        data_dict[k] = shuff_v
    
    return data_dict


def format_data(data_array, seqlens, padwith=10):
    """
    Only meant for padding each array in a minibatch to the max 
    timestep of that minibatch
    data_array elements should have shape- (#features, #timesteps)
    Padding happens on timesteps dimension
    """
    maxseqlen = max(seqlens)
    for i in range(len(data_array)):
        array = data_array[i]
        if array.shape[1] < maxseqlen:
            difference = maxseqlen - array.shape[1]
            array = np.concatenate((array, 
                        padwith*np.ones((array.shape[0], difference))), axis=1)
            data_array[i] = array
    return data_array


def format_data_with_augmentation(encoder_input_arrays, decoder_target_arrays, 
                                  input_seqlens, target_seqlens, padwidth=10):
    """
    each data array elements should have shape- (#features, #timesteps)
    Padding happens on timesteps dimension
    """
    
    probability_interval = 0.25
    
    for i in range(len(encoder_input_arrays)):
        
        q = np.random.rand()
        if q<probability_interval:
            encoder_input_arrays[i] = np.concatenate((encoder_input_arrays[i][:,0:1], 
                                                      np.fliplr(encoder_input_arrays[i][:,1:-1]), 
                                                      encoder_input_arrays[i][:,-1:]), axis=1)
            decoder_target_arrays[i] = np.concatenate((decoder_target_arrays[i][:,0:1], 
                                                      np.fliplr(decoder_target_arrays[i][:,1:-1]), 
                                                      decoder_target_arrays[i][:,-1:]), axis=1)
        
        if q>=probability_interval and q<2*probability_interval:
            input_len = encoder_input_arrays[i].shape[1]
            target_len = decoder_target_arrays[i].shape[1]
            
            # Cutoff the input/output sequence to a maximum of 35%
            cutoff = np.random.rand()*0.25
            encoder_input_arrays[i] = np.concatenate((encoder_input_arrays[i][:, 0:1], 
                                                      encoder_input_arrays[i][:, int(input_len*cutoff):]), axis=1)
            decoder_target_arrays[i] = np.concatenate((decoder_target_arrays[i][:, 0:1], 
                                                      decoder_target_arrays[i][:,int(target_len*cutoff):]), axis=1)
            
            input_seqlens[i] = encoder_input_arrays[i].shape[1]
            target_seqlens[i] = decoder_target_arrays[i].shape[1]
    
    encoder_input_arrays = np.asarray(format_data(encoder_input_arrays, 
                                       seqlens=input_seqlens, padwith=padwidth), np.float32)
    decoder_target_arrays = np.asarray(format_data(decoder_target_arrays, 
                                       seqlens=target_seqlens, padwith=padwidth), np.float32)
    
    return encoder_input_arrays, decoder_target_arrays, input_seqlens, target_seqlens


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def _circulant_matrix(a, W):
    a = np.asarray(a)
    p = np.zeros(W-1,dtype=a.dtype)
    b = np.concatenate((p,a,p))
    s = b.strides[0]
    strided = np.lib.stride_tricks.as_strided
    return strided(b[W-1:], shape=(W,len(a)+W-1), strides=(-s,s))


def create_attn_mask(kernel, MAXLEN, device):
    assert len(kernel)%2 != 0, "kernel should be of odd length"
    cm = _circulant_matrix(kernel, MAXLEN)
    start_index = int((len(kernel)- 1) / 2)
    end_index = int(MAXLEN + len(kernel) - start_index - 1)
    cm = np.asarray(cm[:, start_index:end_index], np.float32)
    return torch.from_numpy(cm).to(device)


def positional_encoding(max_len, embedding_dim):

    '''Sinusoidal Positional_Encoding. See 3.5
    embedding_dim: scalar. Dimensionality of the embedding
    maxlen: scalar. Must be >= T

    returns
    2d tensor that has the shape (maxlen, embedding_dim).
    '''

    # position indices
    position_ind = torch.arange(0, max_len).unsqueeze(0)

    # First part of the PE function: sin and cos argument
    position_enc = np.array([[pos / np.power(10000, (i-i%2)/embedding_dim) for i in range(embedding_dim)] for pos in range(max_len)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    position_enc = torch.from_numpy(position_enc).type(torch.float32) # (maxlen, embedding_dim)

    # lookup
    outputs = nn.Embedding.from_pretrained(position_enc)

    return outputs























