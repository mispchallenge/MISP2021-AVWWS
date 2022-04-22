import numpy as np
import torch
import sys
import os
sys.path.append("..")
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
from scipy.io import wavfile
import random
from tools.network_feature_extract import FilterBank
import librosa

def change_pitch(wav):
    fs , data = wavfile.read(wav)
    n_step = np.random.uniform(-4, 4)
    y_pitched = librosa.effects.pitch_shift(data / 32768.0,fs,n_steps=n_step) 
    return (y_pitched * 32768.0).astype(np.int16)

def change_time(wav):
    fs , stereo = wavfile.read(wav)
    time_factor = np.random.uniform(0.5, 2)
    y_stretch = librosa.effects.time_stretch(stereo / 32768.0, time_factor)
    return (y_stretch * 32768.0).astype(np.int16)

def get_wavdata(wav,scale=15):
    number = np.random.randint(100)
    if number < scale:
        return change_pitch(wav)
    elif number > (100-scale):
        return change_time(wav)
    else:
        _ , data = wavfile.read(wav)
        return data

class myDataset(Dataset):
    def __init__(self, scp_path_wakeup, scp_path_tongyong, fb40_train_mean, fb40_train_var,scale=10):
        self.scp_path_wakeup = scp_path_wakeup
        self.scp_path_tongyong = scp_path_tongyong
        self.fb40_train_mean = fb40_train_mean      
        self.fb40_train_var = fb40_train_var        
        
        with open(self.scp_path_wakeup) as f:
            lines = f.readlines()
        self.files_scp_wakeup = [line.strip() for line in lines]

        with open(self.scp_path_tongyong) as f:
            lines = f.readlines()
        self.files_scp_tongyong = [line.strip() for line in lines]
        self.FeaExt = FilterBank()
        self.files_scp = self.files_scp_wakeup + self.files_scp_tongyong
        self.scale = scale

    def __getitem__(self, idx):
        def load_data_success(idx):
            audio_path = self.files_scp[idx]      
            data = get_wavdata(audio_path,self.scale)
            mel_spec, _ = self.FeaExt(torch.from_numpy(data))
            mel_spec = mel_spec.numpy().T
            T = mel_spec.shape[0]//4
            if T > 2:return True
            else: return False
        
        cur_idx = idx
        while not load_data_success(cur_idx):       
            cur_idx = random.randint(0, len(self.files_scp)-1)

        audio_path = self.files_scp[cur_idx]

        data = get_wavdata(audio_path,self.scale)
        
        mel_spec, _ = self.FeaExt(torch.from_numpy(data))
        mel_spec = mel_spec.numpy().T

        T = mel_spec.shape[0]//4
        mel_spec = mel_spec[:4*T]   

        mean = self.fb40_train_mean*4*T  
        var = self.fb40_train_var*4*T

   
        mel_spec_norm = (mel_spec-np.tile(self.fb40_train_mean,(4*T,1)))/np.sqrt(np.tile(self.fb40_train_var,(4*T,1))+1e-6)  # (x-mu) / sqrt(σ)
        

        if cur_idx < len(self.files_scp_wakeup):
            data_label = 1.0
        else:
            data_label = 0.0

        return mel_spec_norm, data_label 

    def __len__(self):
        return len(self.files_scp)


def myCollateFn(sample_batch):

    sample_batch = sorted(sample_batch, key=lambda x: x[0].shape[0], reverse=True)  
    data_feature = [torch.from_numpy(x[0]) for x in sample_batch]  
    data_label = torch.tensor([x[1] for x in sample_batch]).unsqueeze(-1)
    data_length = [x.shape[0]//4 for x in data_feature]   
    data_feature = pad_sequence(data_feature, batch_first=False, padding_value=0.0)
    return data_feature, data_label, data_length   

class myDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(myDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = myCollateFn
