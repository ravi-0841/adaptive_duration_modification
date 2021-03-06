#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 20:54:05 2021

@author: ravi
"""

import librosa
import pylab
import pickle
import torch
import os

import numpy as np
import pyworld as pw
import scipy.signal as scisig
import scipy.io.wavfile as scwav
import edit_distance

from glob import glob
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from hparams import Hparams
from itakura_parallelogram import ItakuraParallelogram
from transformer_assets.train_evaluate import TrainingEval

from transformer_assets.encoder import Encoder
from transformer_assets.decoder import Decoder
from transformer_assets.seq2seq import Seq2Seq

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
def extract_world_features(wavfile: str, hparams: Hparams):

    """

    Parameters
    ----------
    wavfile : str
       string contains audio file location (.wav)
    f0 : Hparams
       Contains arguments for feature generation

    Returns
    -------
    sp : np.ndarray
        shape = (1, timestamps, n_fft//2 + 1)
    f0 : np.ndarray
        shape = (timestamps, )
    ap : np.ndarray
        shape = (timestamps, n_fft//2 + 1)
        
        These features are not preprocessed via pre-emphasis 
        filter which is important for low freq. regions.

    """
    
    sr, data = scwav.read(wavfile)
    data = np.asarray(data, np.float64)

    assert hparams.hop_size==hparams.win_size, "window size != hop size"

    data = -1 + 2*((data - np.min(data)) / (np.max(data) - np.min(data)))
    data = data - np.mean(data)

    f0, sp, ap = pw.wav2world(data, sr, 
                              frame_period=int(1000*hparams.hop_size))
    
    return (f0, sp, ap)

#%%
def extract_filterbank_features(wavfile: str, hparams: Hparams):
    
    """

    Parameters
    ----------
    wavfile : str
       string contains audio file location (.wav)
    f0 : Hparams
       Contains arguments for feature generation

    Returns
    -------
    energy : np.ndarray
        shape = (1, timestamps, dim)
    sp : np.ndarray
        shape = (1, timestamps, n_fft//2 + 1)
    f0 : np.ndarray
        shape = (timestamps, )
    ap : np.ndarray
        shape = (timestamps, n_fft//2 + 1)

    """
    
    sr, data = scwav.read(wavfile)
    data = np.asarray(data, np.float64)

    assert hparams.hop_size==hparams.win_size, "window size != hop size"

    data = -1 + 2*((data - np.min(data)) / (np.max(data) - np.min(data)))
    data = data - np.mean(data)
    data = scisig.lfilter([1, -0.97], [1], data)

    f0, sp, ap = pw.wav2world(data, sr, 
                              frame_period=int(1000*hparams.hop_size))
    filterbank = librosa.filters.mel(sr=sr, n_fft=1024, 
                                     n_mels=hparams.d_in, norm='slaney')

    energy = np.dot(filterbank, sp.T**2)    #compute filterbank energy
    energy = np.log10(energy + 1e-20)       #compute log of the energy
    mu, std = np.mean(energy, axis=0, keepdims=True), \
                np.std(energy, axis=0, keepdims=True)
    
    energy = (energy - mu) / (std + 1e-10)
    energy = np.concatenate((start_end_tokens_world_80['<sos>'], 
                             energy, 
                             start_end_tokens_world_80['<eos>']), axis=-1)
    energy = np.asarray(energy, np.float32)

    return np.expand_dims(energy.T, axis=0), (f0, sp, ap)

#%%
def compute_dtw_path(model: TrainingEval, 
                     itk_obj: ItakuraParallelogram, 
                     src: np.ndarray, 
                     slope: float, 
                     steps_limit: int):
    
    """

    Parameters
    ----------
    model : TrainingEvaluation
       class containing model and generate function
    itk_obj : ItakuraParallelogram
       class containing itakura mask function and dtw path
    src : np.ndarray
        shape = (1, timestamps, dim)
        output of get_features function
    slope : float
        Masking slope
    steps_limit : int
        # of consecutive horizontal/vertical moves

    Returns
    -------
    path : np.ndarray
        shape = (alignment indices, 2)

    """

    prediction, attention = model.test_decode(src, slope, itakura_object=itk_obj)

    a = 1 / (attention.squeeze() + 1e-12)
    acc_mat = itk_obj.accumulated_cost_matrix(a)
    path = itk_obj.return_constrained_path(acc_mat, steps_limit=steps_limit)
    return attention, path

#%%
def organize_by_path(sp: np.ndarray, f0: np.ndarray, 
                     ap: np.ndarray, path: np.ndarray):

    """

    Parameters
    ----------
    sp : np.ndarray
       shape = (timestamps, dim)
    f0 : np.ndarray
       shape = (timestamps, )
    ap : np.ndarray
       shape = (timestamps, dim)
    path : np.ndarray 
        shape = (2, path len)
        0th row contains source/input feature frame indices
        1th row contains target frame indices

    Returns
    -------
    new_sp : np.ndarray
        shape = (timestamps, dim)
    new_f0 : np.ndarray
        shape = (timestamps, )
    new_ap : np.ndarray
        shape = (timestamps, dim)

    """
    
    path = path.T
    
    new_sp = np.empty((0, 513))
    new_ap = np.empty((0,513))
    new_f0 = []
    for i in range(path[-1,1] + 1):
        try:
            idx = list(path[:,1]).index(i)
            new_sp = np.concatenate((new_sp, sp[path[idx,0],:].reshape(1,-1)), axis=0)
            new_ap = np.concatenate((new_ap, ap[path[idx,0],:].reshape(1,-1)), axis=0)
            new_f0.append(f0[path[idx,0]])
        except Exception as ex:
            pass

    new_sp = np.ascontiguousarray(new_sp)
    new_f0 = np.ascontiguousarray(new_f0)
    new_ap = np.ascontiguousarray(new_ap)
    
    return (new_sp, new_f0, new_ap)

#%%
def path_histogram(path_cords):
    hist = {'diag':0, 'vert':0, 'horz':0}
    if path_cords.shape[1] != 2:
        path_cords = path_cords.T
    for i in range(1, path_cords.shape[0]):
        if (path_cords[i,0] - path_cords[i-1,0]) >= 1:
            if (path_cords[i,1] - path_cords[i-1,1]) >= 1:
                hist['diag'] += 1
            else:
                hist['horz'] += 1
        else:
            hist['vert'] += 1
    
    hist_values = np.asarray(list(hist.values())) + 1
    hist_values = hist_values / np.sum(hist_values)
    return hist_values

#%%
def path_code(path_cords):
    code = []
    if path_cords.shape[1] != 2:
        path_cords = path_cords.T
    for i in range(1, path_cords.shape[0]):
        if (path_cords[i,0] - path_cords[i-1,0]) >= 1:
            if (path_cords[i,1] - path_cords[i-1,1]) >= 1:
                code.append(0)
            else:
                code.append(-1)
        else:
            code.append(1)

    return code

#%%
def compute_symmetric_KL(p, q):

    '''
    
    Parameters
    ----------
    p : np.ndarray
        probability distribution
    q : np.ndarray
        probability distribution

    Returns
    -------
    [KL(p||q) + KL(q||p)] / 2

    '''
    p_q = np.sum(p * np.log10(p / q))
    q_p = np.sum(q * np.log10(q / p))
    return (p_q + q_p) / 2.

#%%
if __name__ == "__main__":
    
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    
    itk_obj = ItakuraParallelogram()
    
    EMBED_DIM = 80
    HIDDEN_DIM = 256  #256
    ENC_LAYERS = 6
    DEC_LAYERS = 6
    ENC_HEADS = 4
    DEC_HEADS = 4
    ENC_PF_DIM = 256  #256
    DEC_PF_DIM = 256  #256
    ENC_DROPOUT = 0.2
    DEC_DROPOUT = 0.2
    LEARNING_RATE = 0.0001 #0.0001 for cmu
    PAD_IDX = 10
    MAXLEN = 1400
    CLIP = 0.1
    BATCH_SIZE = 1
    STEPS_LIMIT = 3
    SLOPE = 1e10
    PAD_SIGNATURE = PAD_IDX * EMBED_DIM
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    enc = Encoder(EMBED_DIM, 
                  HIDDEN_DIM, 
                  ENC_LAYERS, 
                  ENC_HEADS, 
                  ENC_PF_DIM, 
                  ENC_DROPOUT, 
                  device, 
                  pos_encoding = True, 
                  max_length = MAXLEN)
    dec = Decoder(EMBED_DIM, 
                  HIDDEN_DIM, 
                  DEC_LAYERS, 
                  DEC_HEADS, 
                  DEC_PF_DIM, 
                  DEC_DROPOUT, 
                  device, 
                  pos_encoding = True, 
                  max_length = MAXLEN)
    
    model = Seq2Seq(enc, dec, PAD_SIGNATURE, device).to(device)

    train_eval = TrainingEval(model, LEARNING_RATE, CLIP, 
                                    device, PAD_SIGNATURE, MAXLEN)

    # train_eval.model.load_state_dict(torch.load('./models/VESUS/layers_6_hid_256-vesus-neutral-sad-nomask-transformer-model-5ms.pt'))
    train_eval.model.load_state_dict(torch.load('./models/CMU/layers_6_hid_256-cmu-nomask-transformer-model-5ms.pt'))
    
#%%
    
    # valid_src_folder    = sorted(glob(os.path.join("/home/ravi/Downloads/Emo-Conv/neutral-sad/test/neutral/", "*.wav")))
    # valid_tar_folder    = sorted(glob(os.path.join("/home/ravi/Downloads/Emo-Conv/neutral-sad/test/sad/", "*.wav")))

    valid_src_folder    = sorted(glob(os.path.join("/home/ravi/Desktop/adaptive_duration_modification/data/CMU-ARCTIC/test/source/", "*.wav")))
    valid_tar_folder    = sorted(glob(os.path.join("/home/ravi/Desktop/adaptive_duration_modification/data/CMU-ARCTIC/test/target/", "*.wav")))

    # output_folder       = "/home/ravi/Desktop/VESUS/neutral_sad/transformer_model/".format(ENC_LAYERS, 
    #                                                                                        HIDDEN_DIM, 
    #                                                                                        SLOPE, 
    #                                                                                        STEPS_LIMIT)
    output_folder       = "/home/ravi/Desktop/CMU/voice_conversion/transformer_model/".format(ENC_LAYERS, 
                                                                                              HIDDEN_DIM, 
                                                                                              SLOPE, 
                                                                                              STEPS_LIMIT)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    kl_array            = list()
    eddist_array        = list()
    len_pred_array      = list()
    
    for (src_wavfile, tar_wavfile) in tqdm(zip(valid_src_folder, valid_tar_folder)):
        
        # src_wavfile = '/home/ravi/Downloads/Emo-Conv/neutral-angry/valid/neutral/233.wav'
        
        (src_f0, src_sp, src_ap) = extract_world_features(src_wavfile, hp)
        energy, _ = extract_filterbank_features(src_wavfile, hp)
        
        energy_torch_cuda = torch.from_numpy(energy).to(device)
        attention, cords_attn = compute_dtw_path(train_eval, itk_obj, 
                                                 energy_torch_cuda, 
                                                 slope=SLOPE, 
                                                 steps_limit=STEPS_LIMIT)
        prob_cords_attn = path_histogram(cords_attn)
        # print('Attention model ', prob_cords_attn)
    
        new_sp, new_f0, new_ap = organize_by_path(src_sp, src_f0, src_ap, cords_attn)
        
        pylab.figure()
        pylab.subplot(121)
        pylab.imshow(np.log10(attention.squeeze() + 1e-10))
        pylab.plot(cords_attn[0,:], cords_attn[1,:], 'r-', linewidth=2.)
        pylab.axis('off')
        pylab.title('Attention map and DTW path')
        # pylab.xlabel('source sequence'), pylab.ylabel('target sequence')

        #%%
        # pylab.subplot(221)
        # pylab.imshow(np.log10(src_sp.T ** 2)), pylab.title('input spectrum')
        # pylab.subplot(222)
        # pylab.imshow(np.log10(new_sp.T ** 2)), pylab.title('modified spectrum')
        # pylab.subplot(223)
        # pylab.imshow(np.log10(attention.squeeze() + 1e-10))
        # pylab.plot(cords_attn[0,:], cords_attn[1,:], 'r-'), pylab.title('Attention with DTW path')
        
        # speech = pw.synthesize(new_f0, new_sp, new_ap, 16000, frame_period=int(1000*hp.hop_size))
        # speech = -0.5 + ((speech - np.min(speech)) / (np.max(speech) - np.min(speech)))
        # speech = speech - np.mean(speech)
        # scwav.write(output_folder+os.path.basename(src_wavfile), 
        #             16000, np.asarray(speech, np.float32))


        # DTW mechanism    
        fs, src = scwav.read(src_wavfile)
        src = np.asarray(src, np.float64)
    
        fs, tar = scwav.read(tar_wavfile)
        tar = np.asarray(tar, np.float64)
    
        window_samples = int(16000 * hp.hop_size)
        src_mfc = librosa.feature.mfcc(y=src, sr=fs, hop_length=window_samples, 
                                        win_length=window_samples, n_fft=1024, n_mels=128)
        tar_mfc = librosa.feature.mfcc(y=tar, sr=fs, hop_length=window_samples, 
                                        win_length=window_samples, n_fft=1024, n_mels=128)

        # Unconstrained DTW mechanism
        # cost = pairwise_distances(src_mfc.T, tar_mfc.T, metric='cosine')
        # acc_cost, cords = librosa.sequence.dtw(X=src_mfc, Y=tar_mfc, metric='cosine')
        # cords = np.flipud(cords)

        # pylab.subplot(224)
        # pylab.imshow(cost.T)
        # pylab.plot(cords[:,0], cords[:,1], 'r-'), pylab.title('Real DTW (using target)')
        # pylab.suptitle(os.path.basename(src_wavfile)[:-4])

        # Constrained DTW mechanism
        cost = pairwise_distances(tar_mfc.T, src_mfc.T, metric='cosine')
        mask = itk_obj.itakura_mask(src_mfc.shape[1], tar_mfc.shape[1], max_slope=SLOPE)
        mask = (1 - mask) * 1e10
        cost += mask
        acc_mat = itk_obj.accumulated_cost_matrix(cost)
        cords_dtw = itk_obj.return_constrained_path(acc_mat, steps_limit=STEPS_LIMIT)
        prob_cords_dtw = path_histogram(cords_dtw)
        # print('DTW model ', prob_cords_dtw)

        pylab.subplot(122) #224
        pylab.imshow(np.log10(1/pairwise_distances(tar_mfc.T, src_mfc.T, metric='cosine') + 1e-10))
        pylab.plot(cords_dtw[0,:], cords_dtw[1,:], 'r-', linewidth=2.)
        pylab.axis('off')
        pylab.title('Real DTW (using target)')
        pylab.suptitle(os.path.basename(src_wavfile)[:-4])

        pylab.savefig(output_folder+os.path.basename(src_wavfile)[:-4]+'.png')
        pylab.close()
        
        # pred_len = new_sp.shape[0]/src_sp.shape[0]
        # tar_len = tar_mfc.shape[1]/src_mfc.shape[1]
        # len_pred_array.append(np.abs(tar_len - pred_len))
        # kl_div = compute_symmetric_KL(prob_cords_attn, prob_cords_dtw)
        # kl_array.append(kl_div)
        
        # attn_code = path_code(cords_attn)
        # dtw_code = path_code(cords_dtw)
        # sm = edit_distance.SequenceMatcher(a=dtw_code, b=attn_code)
        # eddist_array.append(sm.ratio())
        
        # print('\n', pred_len, tar_len, kl_div, sm.ratio())
        print('\n', src_wavfile)

    # with open("/home/ravi/Desktop/VESUS/neutral_sad/transformer_results.pkl", "wb") as f:
    # with open("/home/ravi/Desktop/CMU/voice_conversion/transformer_results.pkl", "wb") as f:
    #     pickle.dump({'kl':kl_array, 
    #                   'edit':eddist_array, 
    #                   'len_pred':len_pred_array}, f)
    #     f.close()

#%% Standalone testing
    
    # wavfile = '/home/ravi/Downloads/Emo-Conv/neutral-angry/valid/neutral/233.wav'
    
    # (src_f0, src_sp, src_ap) = extract_world_features(wavfile, hp)
    # energy, _ = extract_filterbank_features(wavfile, hp)
    
    # energy_torch_cuda = torch.from_numpy(energy).to(device)
    # attention, cords = compute_dtw_path(train_eval, itk_obj, energy_torch_cuda)

    # new_sp, new_f0, new_ap = organize_by_path(src_sp, src_f0, src_ap, cords)
    
    # pylab.figure(), pylab.subplot(211)
    # pylab.imshow(np.log10(src_sp.T ** 2))
    # pylab.subplot(212)
    # pylab.imshow(np.log10(new_sp.T ** 2))
    
    # pylab.figure()
    # pylab.imshow(np.log10(attention.squeeze() + 1e-10))
    # pylab.plot(cords[0,:], cords[1,:], 'r-')
    
    # speech = pw.synthesize(new_f0, new_sp, new_ap, 16000, frame_period=5)
    # speech = -0.5 + ((speech - np.min(speech)) / (np.max(speech) - np.min(speech)))
    # speech = speech - np.mean(speech)
    # scwav.write('/home/ravi/Desktop/test.wav', 16000, np.asarray(speech, np.float32))


