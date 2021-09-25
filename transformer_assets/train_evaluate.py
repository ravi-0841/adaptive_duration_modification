#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:50:49 2021

@author: ravi
"""

import torch
import torch.nn as nn


#%%
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

#%%
class TrainingEval(object):
    
    def __init__(self, model, learning_rate, clip, 
                 device, pad_signature, max_len):

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate) #optimizer
        self.criterion = nn.L1Loss(reduction = 'none') #criterion
        self.grad_clip = clip
        self.device = device
        self.PAD_SIGNATURE = pad_signature
        self.MAXLEN = max_len
        self.loss_weight = 1 
        
        self.model.apply(initialize_weights)
    
#%%
    def train(self, iterator):
        
        self.model.train()
        
        epoch_reg_loss = 0
        epoch_len_loss = 0
        epoch_loss = 0
    
        generated_seqs = []
        generated_lens = []
        
        for i in range(iterator.batch_count()):
                
            src, trg, inp_seq_len, out_seq_len, mask = iterator[i]
            mask = None
            
            #create target for sequence length prediction
            seq_len_trg = (out_seq_len - 1) / (inp_seq_len - 1)
            
            self.optimizer.zero_grad()
            
            #output = [batch size, trg len - 1, embed dim]
            #trg = [batch size, trg len - 1, embed dim]
            
            pred_len, output, attention = self.model(src[:,1:,:], trg[:,:-1,:], 
                                                     attention_mask = mask)
            trg = trg[:,1:,:]
            
            #put predicted data in the list
            generated_seqs.append((output.cpu(), attention.cpu()))
            generated_lens.append((pred_len.cpu().detach().numpy(), 
                                   seq_len_trg.cpu().numpy()))
            
            cur_batch_size = src.shape[0]
                
            nonpadding = torch.ne(torch.sum(trg, dim = 2), self.PAD_SIGNATURE)
            nonpadding = nonpadding.type(torch.float32)
            
            reg_loss = torch.sum(torch.sum(self.criterion(output, trg), 
                            dim = 2) * nonpadding) / (torch.sum(nonpadding) + 1e-7)
            len_loss = torch.mean(self.criterion(seq_len_trg, pred_len))
            loss = reg_loss + len_loss*self.loss_weight
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            epoch_loss += loss.item() * cur_batch_size
            
            epoch_reg_loss += cur_batch_size*reg_loss.item()
            epoch_len_loss += cur_batch_size*len_loss.item()
            
        return epoch_reg_loss/len(iterator), epoch_len_loss/len(iterator), \
            epoch_loss, generated_seqs, generated_lens
    
#%%
    def evaluate(self, iterator):
        
        self.model.eval()
        
        epoch_reg_loss = 0
        epoch_len_loss = 0
        epoch_loss = 0
    
        generated_seqs = []
        generated_lens = []
        
        with torch.no_grad():

            for i in range(iterator.batch_count()):
                    
                src, trg, inp_seq_len, out_seq_len, mask = iterator[i]
                mask = None
                
                #create target for sequence length prediction
                seq_len_trg = (out_seq_len - 1) / (inp_seq_len - 1)
                
                #output = [batch size, trg len - 1, embed dim]
                #trg = [batch size, trg len - 1, embed dim]
                pred_len, output, attention = self.model(src[:,1:,:], trg[:,:-1,:], 
                                                         attention_mask = mask)
                trg = trg[:,1:,:]
                
                #put predicted data in the list
                generated_seqs.append((output.cpu(), attention.cpu()))
                generated_lens.append((pred_len.cpu().detach().numpy(), 
                                       seq_len_trg.cpu().numpy()))
                
                cur_batch_size = src.shape[0]
                    
                nonpadding = torch.ne(torch.sum(trg, dim = 2), self.PAD_SIGNATURE)
                nonpadding = nonpadding.type(torch.float32)
                
                reg_loss = torch.sum(torch.sum(self.criterion(output, trg), 
                                dim = 2) * nonpadding) / (torch.sum(nonpadding) + 1e-7)
                len_loss = torch.mean(self.criterion(seq_len_trg, pred_len))
                loss = reg_loss + len_loss*self.loss_weight
                
                epoch_loss += loss.item() * cur_batch_size
                
                epoch_reg_loss += cur_batch_size*reg_loss.item()
                epoch_len_loss += cur_batch_size*len_loss.item()
                
            return epoch_reg_loss/len(iterator), epoch_len_loss/len(iterator), \
                epoch_loss, generated_seqs, generated_lens
    
#%%
    def ar_decode(self, iterator, index, itakura_object = None):
    
        self.model.eval()
        
        src, trg, inp_seq_len, out_seq_len, _ = iterator[index]

        
        src_len = int(inp_seq_len) - 1
        
        src_mask = self.model.make_src_mask(src[:,1:,:])
    
        with torch.no_grad():
            pred_len, enc_src, enc_mem = self.model.encoder(src[:,1:,:], src_mask)
        
        pred_len = torch.clamp(pred_len, 0.8, 1.25)
        
        #target length
        pred_trg_len = min(self.MAXLEN, int((inp_seq_len - 1)*pred_len))
        print(pred_len.item(), int(out_seq_len-1)/int(inp_seq_len-1))
        
        #create the mask
        if itakura_object is not None:
            mask = itakura_object.itakura_mask(src_len, pred_trg_len, 1.25)
            mask = torch.from_numpy(mask).to(self.device)
            mask = mask.unsqueeze(0)

        mask = None # mask[:,0:i+1,:]
        gen_tensor = src[:,0:1,:] #start token
        
        #generating frames autoregressively
        for i in range(pred_trg_len):
            
            gen_mask = self.model.make_trg_mask(gen_tensor)
            
            with torch.no_grad():
                output, attention = self.model.decoder(src[:,1:,:], 
                                                  gen_tensor, 
                                                  enc_src, 
                                                  enc_mem, 
                                                  gen_mask, 
                                                  src_mask, 
                                                  attention_mask = mask)
    
            gen_tensor = torch.cat((gen_tensor, output[:,i:i+1,:]), dim = 1)
        
        return gen_tensor[:,1:,:].cpu().numpy(), trg.cpu().numpy(), attention.cpu().numpy()

#%%
    def test_decode(self, src, slope = 1.25, itakura_object = None):
        
        """
        src: [1, timestamps, dim]
        """
        self.model.eval()
        
        src_len = src.shape[1] - 1
        
        src_mask = self.model.make_src_mask(src[:,1:,:])
    
        with torch.no_grad():
            pred_len, enc_src, enc_mem = self.model.encoder(src[:,1:,:], src_mask)
        
        pred_len = torch.clamp(pred_len, 1/slope, slope)
        
        #target length
        pred_trg_len = min(self.MAXLEN, int(src_len*pred_len))
        
        #create the mask
        if itakura_object is not None:
            mask = itakura_object.itakura_mask(src_len, pred_trg_len, 1.25)
            mask = torch.from_numpy(mask).to(self.device)
            mask = mask.unsqueeze(0)

        mask = None # mask[:,0:i+1,:]
        gen_tensor = src[:,0:1,:] #start token
        
        #generating frames autoregressively
        for i in range(pred_trg_len):
            
            gen_mask = self.model.make_trg_mask(gen_tensor)
            
            with torch.no_grad():
                output, attention = self.model.decoder(src[:,1:,:], 
                                                  gen_tensor, 
                                                  enc_src, 
                                                  enc_mem, 
                                                  gen_mask, 
                                                  src_mask, 
                                                  attention_mask = mask)
    
            gen_tensor = torch.cat((gen_tensor, output[:,i:i+1,:]), dim = 1)
        
        return gen_tensor[:,1:,:].cpu().numpy(), attention.cpu().numpy()

















