#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 21:59:51 2020

@author: ravi
"""

#import tensorflow as tf
#from utils import calc_num_batches

import logging
import scipy.io.wavfile as scwav
import numpy as np
import librosa
import os
import scipy.signal as scisig
import pyworld as pw
import pickle

from tqdm import tqdm
from glob import glob
from hparams import Hparams
from sklearn.preprocessing import normalize


#%%
with open("./data/start_end_token.pkl", "rb") as f:
    start_end_tokens = pickle.load(f)
    f.close()

with open("./data/start_end_token_world_spect.pkl", "rb") as f:
    start_end_tokens_world_spect = pickle.load(f)
    f.close()

with open("./data/start_end_token_world_80.pkl", "rb") as f:
    start_end_tokens_world_80 = pickle.load(f)
    f.close()

with open("./data/start_end_token_world_128.pkl", "rb") as f:
    start_end_tokens_world_128 = pickle.load(f)
    f.close()


#%%
def mvn(list_array):
    
    aggregate_array = np.empty((list_array[0].shape[0], 0))
    for cur_item in list_array:
        aggregate_array = np.concatenate((aggregate_array, cur_item), axis=1)
    
    mean = np.mean(aggregate_array, axis=1, keepdims=True)
    stan = np.std(aggregate_array, axis=1, keepdims=True)
    
    del aggregate_array
    
    new_list_array = []
    for cur_item in list_array:
        new_item = (cur_item - mean) / (stan + 1e-6)
        new_item = np.concatenate((start_end_tokens['<sos>'], 
                                   cur_item, 
                                   start_end_tokens['<eos>']), axis=1)
        new_list_array.append(new_item)
    
    return new_list_array


def make_seqover(target_arrays):
    seqovers = []
    for i in target_arrays:
        seqover = np.zeros((1, i.shape[1]))
        seqover[0,-1] = 1
        seqovers.append(seqover)
    return seqovers


def load_data(fpath1, fpath2):
    '''Loads source and target data.
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.

    Returns
    sents1: list of source sents
    sents2: list of target sents
    '''
    utts1, utts2 = [], []
    files1 = sorted(glob(os.path.join(fpath1, '*.wav')))
    files2 = sorted(glob(os.path.join(fpath2, '*.wav')))
    for file1, file2 in zip(files1, files2):
        utts1.append(file1)
        utts2.append(file2)
    return utts1, utts2


#%%
def _compute_stft_energy(data, sr=16000, n_fft=1024, n_mels=40, 
                    hop_len=0.015, win_len=0.025):

    data = -1 + 2*((data - np.min(data)) / (np.max(data) - np.min(data)))
    data = data - np.mean(data)
    data = scisig.lfilter([1, -0.97], [1], data)

    filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, norm='slaney')
    spect = np.abs(librosa.stft(y=np.asarray(data, np.float64), n_fft=n_fft, 
            hop_length=int(hop_len*sr), win_length=int(win_len*sr)))
    filterbank_energy = np.dot(filterbank, spect**2)
    filterbank_energy = np.log10(filterbank_energy + 1e-20)

    mu, std = np.mean(filterbank_energy, axis=0, keepdims=True), np.std(filterbank_energy, axis=0, keepdims=True)
    return (filterbank_energy - mu) / (std + 1e-10)


def _compute_col_norm_energy(data, sr=16000, n_fft=1024, n_mels=40, 
                    hop_len=0.015, win_len=0.025):

    data = -1 + 2*((data - np.min(data)) / (np.max(data) - np.min(data)))
    data = data - np.mean(data)
    data = scisig.lfilter([1, -0.97], [1], data)

    filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, norm='slaney')
    spect = np.abs(librosa.stft(y=np.asarray(data, np.float64), n_fft=n_fft, 
            hop_length=int(hop_len*sr), win_length=int(win_len*sr)))
    filterbank_energy = np.dot(filterbank, spect**2)
    filterbank_energy = np.log10(filterbank_energy + 1e-20)

    filterbank_energy = normalize(filterbank_energy, norm='l2', axis=0)
    return filterbank_energy


def _compute_world_energy(data, sr=16000, n_fft=1024, n_mels=40, 
                    hop_len=0.01, win_len=0.01):

    assert hop_len==win_len, "window size and window stride should be same"

    data = -1 + 2*((data - np.min(data)) / (np.max(data) - np.min(data)))
    data = data - np.mean(data)
    data = scisig.lfilter([1, -0.97], [1], data)

    f0, sp, ap = pw.wav2world(data, sr, frame_period=int(1000*hop_len))
    filterbank = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, norm='slaney')
    sp = np.dot(filterbank, sp.T**2)        #compute filterbank energy
    sp = np.log10(sp + 1e-20)               #compute log of the energy

    mu, std = np.mean(sp, axis=0, keepdims=True), np.std(sp, axis=0, keepdims=True)
    return (sp - mu) / (std + 1e-10)


def _compute_world_mfcc(data, sr=16000, n_fft=1024, n_mels=40, 
                    hop_len=0.01, win_len=0.01):

    assert hop_len==win_len, "window size and window stride should be same"

    data = -1 + 2*((data - np.min(data)) / (np.max(data) - np.min(data)))
    data = data - np.mean(data)
    data = scisig.lfilter([1, -0.97], [1], data)

    f0, sp, ap = pw.wav2world(data, sr, frame_period=int(1000*hop_len))
    mfcc = np.transpose(pw.code_spectral_envelope(sp, sr, n_mels))

    mu, std = np.mean(mfcc, axis=0, keepdims=True), np.std(mfcc, axis=0, keepdims=True)
    return (mfcc - mu) / (std + 1e-10)


#%%
def encode(input_file, type, n_mels=40, hop_len=0.015, win_len=0.025):
    '''Converts string to number. Used for `generator_fn`.
    input_file: path of source utterance
    type: "x" (source side) or "y" (target side)

    Returns
    mel filterbank energy
    '''
    sr, data = scwav.read(input_file)
    data = np.asarray(data, np.float64)
    energy = _compute_world_energy(data, sr=sr, n_mels=n_mels, 
                             hop_len=hop_len, win_len=win_len)
    if type=="x": energy = np.concatenate((start_end_tokens_world_80['<sos>'], 
                                           energy, 
                                           start_end_tokens_world_80['<eos>']), axis=-1)
    else: energy = np.concatenate((start_end_tokens_world_80['<sos>'], 
                                   energy, 
                                   start_end_tokens_world_80['<eos>']), axis=-1)

    return energy


def encode_mfcc(input_file, type, n_mels=40, hop_len=0.015, win_len=0.025):
    '''Converts string to number. Used for `generator_fn`.
    input_file: path of source utterance
    type: "x" (source side) or "y" (target side)

    Returns
    mel filterbank energy
    '''
    sr, data = scwav.read(input_file)
    data = np.asarray(data, np.float64)
    energy = _compute_world_mfcc(data, sr=sr, n_mels=n_mels, 
                             hop_len=hop_len, win_len=win_len)
    if type=="x": energy = np.concatenate((start_end_tokens_world_80['<sos>'], 
                                           energy, 
                                           start_end_tokens_world_80['<eos>']), axis=-1)
    else: energy = np.concatenate((start_end_tokens_world_80['<sos>'], 
                                   energy, 
                                   start_end_tokens_world_80['<eos>']), axis=-1)

    return energy


def encode_col_norm(input_file, type, n_mels=40, hop_len=0.015, win_len=0.025):
    '''Converts string to number. Used for `generator_fn`.
    input_file: path of source utterance
    type: "x" (source side) or "y" (target side)

    Returns
    mel filterbank energy
    '''
    sr, data = scwav.read(input_file)
    data = np.asarray(data, np.float64)
    energy = _compute_col_norm_energy(data, sr=sr, n_mels=n_mels, 
                             hop_len=hop_len, win_len=win_len)
    if type=="x": energy = np.concatenate((start_end_tokens_world_80['<sos>']/(50+np.linalg.norm(start_end_tokens_world_80['<sos>'])), 
                                           energy, 
                                           start_end_tokens_world_80['<eos>']/(50+np.linalg.norm(start_end_tokens_world_80['<eos>']))), axis=-1)
    else: energy = np.concatenate((start_end_tokens_world_80['<sos>']/(50+np.linalg.norm(start_end_tokens_world_80['<sos>'])), 
                                   energy, 
                                   start_end_tokens_world_80['<eos>']/(50+np.linalg.norm(start_end_tokens_world_80['<eos>']))), axis=-1)

    return energy


def encode_wo_tokens(input_file, type, n_mels=40, hop_len=0.015, win_len=0.025):
    '''Converts string to number. Used for `generator_fn`.
    input_file: path of source utterance
    type: "x" (source side) or "y" (target side)

    Returns
    mel filterbank energy
    '''
    sr, data = scwav.read(input_file)
    data = np.asarray(data, np.float64)
    energy = _compute_stft_energy(data, sr=sr, n_mels=n_mels, 
                             hop_len=hop_len, win_len=win_len)
    return energy


#%%
def generator_fn(utts1, utts2, n_mels, hop_len, win_len):
    '''Generates training / evaluation data
    utts1: list of source emotion utterances wav path
    utts2: list of target emotion utterances wav path

    yields
    xs: tuple of
        x: float. array of mel filterbank energy of source utterance
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) utterance
    labels: tuple of
        decoder_input: decoder_input: array of encoded decoder inputs
        y: array of mel filterbank energy of target utterance
        y_seqlen: int. sequence length of y
        sent2: str. target utterance
    '''
    
    encoder_input_seqs = list()
    decoder_input_seqs = list()
    decoder_target_seqs = list()
    
    input_seqlens = list()
    target_seqlens = list()
    
    decoder_seqover = list()
    
    
    for utt1, utt2 in tqdm(zip(utts1, utts2)):
#        print(utt1)
        x = encode(utt1, "x", n_mels=n_mels, hop_len=hop_len, win_len=win_len)
        y = encode(utt2, "y", n_mels=n_mels, hop_len=hop_len, win_len=win_len)
        decoder_input, decoder_target = y[:,:-1], y
        
        seqover = np.zeros((1, y.shape[1]))
        seqover[0,-1] = 1

        x_seqlen, y_seqlen = x.shape[1], y.shape[1]
        
        encoder_input_seqs.append(x)
        decoder_input_seqs.append(decoder_input)
        decoder_target_seqs.append(decoder_target)
        input_seqlens.append(x_seqlen)
        target_seqlens.append(y_seqlen)
        decoder_seqover.append(seqover)
        
    return encoder_input_seqs, input_seqlens, decoder_input_seqs, \
            decoder_target_seqs, decoder_seqover, target_seqlens


def generator_fn_wo_tokens(utts1, utts2, n_mels, hop_len, win_len):
    '''Generates training / evaluation data
    utts1: list of source emotion utterances wav path
    utts2: list of target emotion utterances wav path

    yields
    xs: tuple of
        x: float. array of mel filterbank energy of source utterance
        x_seqlen: int. sequence length of x
        sent1: str. raw source (=input) utterance
    labels: tuple of
        decoder_input: decoder_input: array of encoded decoder inputs
        y: array of mel filterbank energy of target utterance
        y_seqlen: int. sequence length of y
        sent2: str. target utterance
    '''
    
    encoder_input_seqs = list()
    decoder_input_seqs = list()
    decoder_target_seqs = list()
    
    input_seqlens = list()
    target_seqlens = list()
    
    
    for utt1, utt2 in tqdm(zip(utts1, utts2)):
#        print(utt1)
        x = encode_wo_tokens(utt1, "x", n_mels=n_mels, hop_len=hop_len, win_len=win_len)
        y = encode_wo_tokens(utt2, "y", n_mels=n_mels, hop_len=hop_len, win_len=win_len)

        x_seqlen, y_seqlen = x.shape[1], y.shape[1]
        
        encoder_input_seqs.append(x)
        decoder_input_seqs.append(y)
        decoder_target_seqs.append(y)
        input_seqlens.append(x_seqlen)
        target_seqlens.append(y_seqlen)
        
    return encoder_input_seqs, input_seqlens, decoder_input_seqs, \
            decoder_target_seqs, target_seqlens


#%%
if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("# hparams")
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    sents1, sents2 = load_data(hp.valid1_cmu, hp.valid2_cmu)
    encoder_input_seqs, input_seqlens, decoder_input_seqs, \
            decoder_target_seqs, decoder_seqover, target_seqlens \
            = generator_fn(sents1, sents2, n_mels=hp.d_in, hop_len=hp.hop_size, 
                          win_len=hp.win_size)

    data_dict = {
                    'encoder_input_seqs': encoder_input_seqs, 
                    'input_seqlens': input_seqlens, 
                    'decoder_input_seqs': decoder_input_seqs, 
                    'decoder_target_seqs': decoder_target_seqs, 
                    'decoder_seqover': decoder_seqover, 
                    'target_seqlens': target_seqlens
                }
    
    with open('./data/CMU/cmu_valid_world_mvn_5ms_new.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
        f.close()


#%%
# if __name__=='__main__':
#     logging.basicConfig(level=logging.INFO)
#     logging.info("# hparams")
#     hparams = Hparams()
#     parser = hparams.parser
#     hp = parser.parse_args()
#     sents1, sents2 = load_data(hp.eval1, hp.eval2)
#     encoder_input_seqs, input_seqlens, decoder_input_seqs, \
#             decoder_target_seqs, target_seqlens \
#             = generator_fn_wo_tokens(sents1, sents2, n_mels=hp.d_in, hop_len=hp.hop_size, 
#                            win_len=hp.win_size)
    
#     encoder_input_seqs = mvn(encoder_input_seqs)
#     decoder_input_seqs = mvn(decoder_input_seqs)
#     decoder_target_seqs = mvn(decoder_target_seqs)
#     input_seqlens = [i+2 for i in input_seqlens]
#     target_seqlens = [i+2 for i in target_seqlens]
#     decoder_seqover = make_seqover(decoder_target_seqs)
    
#     data_dict = {
#                     'encoder_input_seqs': encoder_input_seqs, 
#                     'input_seqlens': input_seqlens, 
#                     'decoder_input_seqs': decoder_input_seqs, 
#                     'decoder_target_seqs': decoder_target_seqs, 
#                     'decoder_seqover': decoder_seqover, 
#                     'target_seqlens': target_seqlens
#                 }
    
#     with open('./data/valid_neutral_angry_mvn_global.pkl', 'wb') as f:
#         pickle.dump(data_dict, f)
#         f.close()
