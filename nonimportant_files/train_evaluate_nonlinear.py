#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:26:10 2021

@author: ravi
"""

import torch
import numpy as np

class TrainingEval(object):
    
    def __init__(self, emb_dim, model, optimizer, criterion, device, 
                 clip, max_len, pad_signature, pad_vector, exp_mask=False):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.grad_clip = clip
        self.pad_signature = pad_signature
        self.pad_vector = pad_vector
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.device = device
        self.exp_mask = exp_mask


    def train(self, iterator):
        self.model.train()
    
        epoch_loss = 0
        
        generated_seqs = []
        
        for i in range(iterator.batch_count()):
            
            src, trg, _, _ = iterator[i]
            
            self.optimizer.zero_grad()
            
            output, attn = self.model(src, trg[:,:-1,:]) #originally src
            generated_seqs.append((output.cpu(), attn.cpu()))
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len - 1, target dim]
            cur_batch_size = output.shape[0]
            output_dim = output.shape[-1]
            trg = trg[:,1:,:]
            trg_len = output.shape[1]
            
            #create an exponentially decaying mask
            if self.exp_mask:
                loss_mask = torch.exp(-1 * torch.arange(0, trg_len, 1) / (trg_len/2))
                loss_mask = loss_mask.repeat(cur_batch_size,1).to(self.device)
            else:
                loss_mask = torch.ones(cur_batch_size, trg_len).to(self.device)
            
            nonpadding = torch.ne(torch.sum(trg, dim=-1), self.pad_signature)
            nonpadding = nonpadding.type(torch.float32) * loss_mask
            
            loss = torch.sum(torch.sum(self.criterion(output, trg), 
                            dim=-1) * nonpadding) / (torch.sum(nonpadding) + 1e-7)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            epoch_loss += cur_batch_size*loss.item()
            
        return epoch_loss / len(iterator), generated_seqs


    def evaluate(self, iterator):
        
        self.model.eval()
    
        epoch_loss = 0
        
        generated_seqs = []
        
        with torch.no_grad():
        
            for i in range(iterator.batch_count()):
            
                src, trg, _, _ = iterator[i]
    
                output, attn = self.model(src, trg[:,:-1,:])
                generated_seqs.append((output.cpu(), attn.cpu()))
            
                #output = [batch size, trg len - 1, output dim]
                #trg = [batch size, trg len - 1, output dim]
                cur_batch_size = output.shape[0]
                output_dim = output.shape[-1]
                trg = trg[:,1:,:]
                
                nonpadding = torch.ne(torch.sum(trg, dim=-1), self.pad_signature)
                nonpadding = nonpadding.type(torch.float32)
            
                loss = torch.sum(torch.sum(self.criterion(output, trg), 
                            dim=-1) * nonpadding) / (torch.sum(nonpadding) + 1e-7)
    
                epoch_loss += cur_batch_size*loss.item()
            
        return epoch_loss / len(iterator), generated_seqs


    def ar_decode(self, iterator, index, device):

        self.model.eval()
        
        src, trg, _, _ = iterator[index]
        
        #length of src signal
        src_len = src.shape[1]
    
        with torch.no_grad():
            encoder_conved, encoder_combined = self.model.encoder(src)
    
        gen_tensor = src[:,0:1,:] #gen_tensor = torch.zeros(1, 1, self.emb_dim).to(device)
    
        for i in range(min(self.max_len, int(1.5*src_len))):
    
            with torch.no_grad():
                output, attention = self.model.decoder(gen_tensor, 
                                                       encoder_conved, 
                                                       encoder_combined)
    
            gen_tensor = torch.cat((gen_tensor, output[:,i:i+1,:]), dim=1)
            cur_timestep = output[:,i:i+1, :].cpu().numpy()
    
            if np.sum(cur_timestep) == self.pad_signature :
                break
        
        return gen_tensor[:,1:,:].cpu().numpy(), trg.cpu().numpy(), attention.cpu().numpy()