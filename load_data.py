#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 17:49:24 2021

@author: ravi
"""

import numpy as np
import pickle
import torch

from utils import format_data, shuffle_data, format_data_with_augmentation
from itakura_parallelogram import ItakuraParallelogram

#%%
class LoadData(torch.utils.data.Dataset):
    
    def __init__(self, pkl_file, batch_size, device, 
                 slope=1.25, augment=False, padwith=10, masking=True):
        
        self.batch_size = batch_size
        self.pkl_file = pkl_file
        
        with open(self.pkl_file, "rb") as f:
            self.data_dict = pickle.load(f)
            f.close()

        self.pad_signature = padwith
        self.device = device
        self.augment = augment
        self.lower_bound = -2**32 + 1
        self.itakura_object = ItakuraParallelogram()
        self.slope = slope
        self.masking = masking

        
    def __len__(self):
        return len(self.data_dict['encoder_input_seqs'])

    
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        start = idx*self.batch_size
        end = (idx+1)*self.batch_size
        
        #get data for formatting
        encoder_input_arrays = self.data_dict['encoder_input_seqs'][start:end]
        input_seqlens = self.data_dict['input_seqlens'][start:end]
        decoder_target_arrays = self.data_dict['decoder_target_seqs'][start:end]
        target_seqlens = self.data_dict['target_seqlens'][start:end]
        
        
        #if augment==True, flip/truncate sequences with probability 0.25
        if self.augment:
            input_data, target_data, \
                input_seqlens, target_seqlens = format_data_with_augmentation(encoder_input_arrays, 
                                                                    decoder_target_arrays, 
                                                                    input_seqlens, 
                                                                    target_seqlens, 
                                                                    padwidth=self.pad_signature)
        else:
            input_data = np.asarray(format_data(encoder_input_arrays, 
                                           input_seqlens, padwith=self.pad_signature), np.float32)
            
            target_data = np.asarray(format_data(decoder_target_arrays, 
                                           target_seqlens, padwith=self.pad_signature), np.float32)
        
        #create the attention mask
        if self.masking:
            attention_mask = self.create_mask(input_seqlens, 
                                              target_seqlens, 
                                              input_data.shape[2], 
                                              target_data.shape[2])
        else:
            attention_mask = []
            for (i,t) in zip(input_seqlens, target_seqlens):
                mask = np.zeros((target_data.shape[2]-1, input_data.shape[2]-1))
                mask[:i, :t] = 1.
                attention_mask.append(mask)
        
        return torch.from_numpy(np.transpose(input_data, axes=[0,2,1])).to(self.device), \
                torch.from_numpy(np.transpose(target_data, axes=[0,2,1])).to(self.device), \
                torch.from_numpy(np.reshape(np.asarray(self.data_dict['input_seqlens'][start:end], np.float32), (-1,1))).to(self.device), \
                torch.from_numpy(np.reshape(np.asarray(self.data_dict['target_seqlens'][start:end], np.float32), (-1,1))).to(self.device), \
                torch.from_numpy(np.asarray(attention_mask, np.float32)).to(self.device)

    
    def batch_count(self):
        if len(self)%self.batch_size == 0:    
            return len(self) // self.batch_size
        else:
            return len(self) // self.batch_size + 1

    
    def shuffle_data(self):
        self.data_dict = shuffle_data(self.data_dict)


    def create_mask(self, input_seqlens, target_seqlens, 
                    padded_input_seqlen, padded_output_seqlen):

        attention_mask = []
        for i,t in zip(input_seqlens, target_seqlens):
            cords = self.itakura_object.itakura_parallelogram(i-1, t-1, self.slope)
            mask = np.zeros((padded_output_seqlen-1, padded_input_seqlen-1))
            for col in range(i-1):
                mask[cords[0,col]:cords[1,col], col] = 1.
            attention_mask.append(mask)
        
        return attention_mask
            


























