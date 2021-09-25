#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:26:10 2021

@author: ravi
"""

import torch
import torch.nn as nn
import numpy as np

#%%
class TrainingEval(object):
    
    def __init__(self, emb_dim, model, 
                 learning_rate, device, 
                 clip, max_len, pad_signature, 
                 exp_mask=False):
        
        self.model = model
        self.learning_rate = learning_rate
        self.grad_clip = clip
        self.pad_signature = pad_signature
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.device = device
        self.exp_mask = exp_mask
        self.len_loss_wt = 5
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.learning_rate)
        self.criterion = nn.L1Loss(reduction = 'none')

#%%
    def train(self, iterator):

        self.model.train()
    
        epoch_reg_loss  = 0
        epoch_len_loss  = 0
        epoch_loss      = 0
        
        generated_seqs  = []
        generated_lens  = []

        for i in range(iterator.batch_count()):
            
            src, trg, inp_seq_len, out_seq_len, mask = iterator[i]
            
            #create target for sequence length prediction
            seq_len_trg = (out_seq_len - 1) / (inp_seq_len - 1)
            
            self.optimizer.zero_grad()
            
            if np.random.uniform(0,1) <= 0.25:
                pred_len, output, attn = self.model(src[:,1:,:], trg[:,:-1,:], 
                                                sample=True, 
                                                mask=mask)
            else:
                pred_len, output, attn = self.model(src[:,1:,:], trg[:,:-1,:], 
                                                    sample=False, 
                                                    mask=mask)
            
            #put predicted data in the list
            generated_seqs.append((output.cpu(), attn.cpu()))
            generated_lens.append((pred_len.cpu().detach().numpy(), 
                                   seq_len_trg.cpu().numpy()))
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len - 1, target dim]
            cur_batch_size = output.shape[0]
            trg_len = output.shape[1]
            trg = trg[:,1:,:]
            
            #create an exponentially decaying mask
            if self.exp_mask:
                loss_mask = torch.exp(-1 * torch.arange(0, trg_len, 1) / (trg_len/2))
                loss_mask = loss_mask.repeat(cur_batch_size,1).to(self.device)
            else:
                loss_mask = torch.ones(cur_batch_size, trg_len).to(self.device)
            
            nonpadding = torch.ne(torch.sum(trg, dim=-1), self.pad_signature)
            nonpadding = nonpadding.type(torch.float32) * loss_mask
            
            reg_loss = torch.sum(torch.sum(self.criterion(output, trg), 
                            dim=-1) * nonpadding) / (torch.sum(nonpadding) + 1e-7)
            len_loss = torch.mean(self.criterion(seq_len_trg, pred_len))
            loss = reg_loss + len_loss*self.len_loss_wt
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            epoch_reg_loss += cur_batch_size*reg_loss.item()
            epoch_len_loss += cur_batch_size*len_loss.item()
            epoch_loss += cur_batch_size*loss.item()
            
        return epoch_reg_loss/len(iterator), epoch_len_loss/len(iterator), \
            epoch_loss/len(iterator), generated_seqs, generated_lens

#%%
    def evaluate(self, iterator):

        self.model.eval()
    
        epoch_reg_loss  = 0
        epoch_len_loss  = 0
        epoch_loss      = 0
        
        generated_seqs  = []
        generated_lens  = []
        
        with torch.no_grad():
        
            for i in range(iterator.batch_count()):
            
                src, trg, inp_seq_len, out_seq_len, mask = iterator[i]
                
                #create target for sequence length prediction
                seq_len_trg = (out_seq_len - 1) / (inp_seq_len - 1)
    
                pred_len, output, attn = self.model(src[:,1:,:], trg[:,:-1,:], 
                                                    sample=False, 
                                                    mask=mask)
                
                #put predicted data in the list
                generated_seqs.append((output.cpu(), attn.cpu()))
                generated_lens.append((pred_len.cpu().detach().numpy(), 
                                       seq_len_trg.cpu().numpy()))
            
                #output = [batch size, trg len - 1, output dim]
                #trg = [batch size, trg len - 1, output dim]
                cur_batch_size = output.shape[0]
                trg = trg[:,1:,:]
                
                nonpadding = torch.ne(torch.sum(trg, dim=-1), self.pad_signature)
                nonpadding = nonpadding.type(torch.float32)
            
                reg_loss = torch.sum(torch.sum(self.criterion(output, trg), 
                            dim=-1) * nonpadding) / (torch.sum(nonpadding) + 1e-7)
                len_loss = torch.mean(self.criterion(seq_len_trg, pred_len))
                loss = reg_loss + len_loss*self.len_loss_wt
    
                epoch_reg_loss += cur_batch_size*reg_loss.item()
                epoch_len_loss += cur_batch_size*len_loss.item()
                epoch_loss += cur_batch_size*loss.item()
            
        return epoch_reg_loss/len(iterator), epoch_len_loss/len(iterator), \
            epoch_loss/len(iterator), generated_seqs, generated_lens

#%%
    def ar_decode(self, iterator, index, device, itakura_object=None):

        self.model.eval()
        
        src, trg, inp_seq_len, out_seq_len, _ = iterator[index]
        
        src_len = int(inp_seq_len) - 1
    
        with torch.no_grad():
            #Encoder operation to get the predicted length
            pred_len, encoder_conved, encoder_conved_embed = self.model.encoder(src[:,1:,:])

        pred_len = torch.clamp(pred_len, 0.8, 1.25)
        print(pred_len.item(), int(out_seq_len-1)/int(inp_seq_len-1))
        
        #target length
        pred_trg_len = min(self.max_len, int((inp_seq_len - 1)*pred_len)+1)
        
        #create the mask
        if itakura_object is not None:
            mask = itakura_object.itakura_mask(src_len, pred_trg_len, 1.25)
            mask = torch.from_numpy(mask).to(device)
            mask = mask.unsqueeze(0)

        gen_tensor = src[:,0:1,:] #start token
        
        #generating frames autoregressively
        for i in range(pred_trg_len):
    
            with torch.no_grad():
                #Decoder operation to get the output at time i
                output, attention = self.model.decoder(src[:,1:,:], 
                                                       gen_tensor, 
                                                       encoder_conved, 
                                                       encoder_conved_embed, 
                                                       sample=False, 
                                                       mask=mask[:,0:i+1,:])
    
            gen_tensor = torch.cat((gen_tensor, output[:,i:i+1,:]), dim=1)
        
        return src[:,1:,:].cpu().numpy(), gen_tensor[:,1:,:].cpu().numpy(), \
            trg.cpu().numpy(), attention.cpu().numpy()

#%%
    def test_decode(self, src, slope=1.25, itakura_object=None):

        """
        src: [1, timestamps, dim]
        """
        self.model.eval()
        
        src_len = int(src.shape[1]) - 1
    
        with torch.no_grad():
            #Encoder operation to get embeddings and predicted length
            pred_len, encoder_conved, encoder_conved_embed = self.model.encoder(src[:,1:,:])
        
        pred_len = torch.clamp(pred_len, 1/slope, slope)
        
        #target length
        pred_trg_len = min(self.max_len, int(src_len*pred_len))
        
        #create the mask
        if itakura_object is not None:
            mask = itakura_object.itakura_mask(src_len, pred_trg_len, slope)
            mask = torch.from_numpy(mask).to(self.device)
            mask = mask.unsqueeze(0)
    
        gen_tensor = src[:,0:1,:] #start token

        #generating frames autoregressively
        for i in range(pred_trg_len):
            
            with torch.no_grad():
                #Decoder operation to get the output at time i
                output, attention = self.model.decoder(src[:,1:,:], 
                                                  gen_tensor, 
                                                  encoder_conved, 
                                                  encoder_conved_embed, 
                                                  sample=False, 
                                                  mask=mask[:,0:i+1,:])
    
            gen_tensor = torch.cat((gen_tensor, output[:,i:i+1,:]), dim = 1)
        
        return gen_tensor[:,1:,:].cpu().numpy(), attention.cpu().numpy()
