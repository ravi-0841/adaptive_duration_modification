#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 12:37:35 2020

@author: ravi
"""

import librosa
import numpy as np
import pyworld as pw
import scipy.io.wavfile as scwav


fs, src = scwav.read('/home/ravi/Downloads/Emo-Conv/neutral-angry/all_above_0.5/neutral/340.wav')
fs, tar = scwav.read('/home/ravi/Downloads/Emo-Conv/neutral-angry/all_above_0.5/angry/340.wav')

src = -1 + 2*((src - np.min(src)) / (np.max(src) - np.min(src)))
tar = -1 + 2*((tar - np.min(tar)) / (np.max(tar) - np.min(tar)))
src = src - np.mean(src)
tar = tar - np.mean(tar)

src_mfc = librosa.feature.mfcc(y=src, sr=fs, hop_length=160, win_length=160, n_fft=1024, n_mels=128)
tar_mfc = librosa.feature.mfcc(y=tar, sr=fs, hop_length=160, win_length=160, n_fft=1024, n_mels=128)
_, cords = librosa.sequence.dtw(X=src_mfc, Y=tar_mfc, metric='cosine')
cords = np.flipud(cords)

f0, sp, ap = pw.wav2world(np.asarray(src, np.float64), fs, frame_period=10)

new_sp = np.empty((0, 513))
new_ap = np.empty((0,513))
new_f0 = []
for i in range(cords[-1,1] + 1):
    try:
        idx = list(cords[:,1]).index(i)
        new_sp = np.concatenate((new_sp, sp[cords[idx,0],:].reshape(1,-1)), axis=0)
        new_ap = np.concatenate((new_ap, ap[cords[idx,0],:].reshape(1,-1)), axis=0)
        new_f0.append(f0[cords[idx,0]])
    except Exception as ex:
        print(ex)

new_f0 = np.asarray(new_f0, order='C')
new_sp = np.ascontiguousarray(new_sp)
new_ap = np.ascontiguousarray(new_ap)
recon = pw.synthesize(new_f0, new_sp, new_ap, fs, frame_period=10)
recon = ((recon - np.min(recon)) / (np.max(recon) - np.min(recon)))
recon = recon - np.mean(recon)
recon = np.asarray(recon, np.float32)
scwav.write('/home/ravi/Desktop/duration_modified_340.wav', fs, recon)