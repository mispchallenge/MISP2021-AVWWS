#!/usr/bin/env python
# _*_ coding: UTF-8 _*_
import os
import codecs
import argparse,time,math
import numpy as np
import scipy.io.wavfile as wf
from multiprocessing import Pool
from nara_wpe.wpe import wpe_v8 as wpe
from nara_wpe.utils import stft, istft

def wpe_worker(wav_scp, data_root='MISP_data_root', output_root='MISP_data_out_root', processing_id=None, processing_num=None):
    sampling_rate = 16000
    iterations = 5
    
    stft_options = dict(
        size=512,
        shift=128,
        window_length=None,
        fading=True,
        pad=True,
        symmetric_window=False
    )

    with codecs.open(wav_scp, 'r') as handle:
        lines_content_file = handle.readlines()
        
    lines_content = lines_content_file  
    wav_lines = [*map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content)]
    
    data_root = '/'.join(wav_lines[0].strip().split(' ')[0].split('/')[:-5])
    for wav_idx in range(len(wav_lines)):

        if processing_id is None:
            processing_token = True
        else:
            if wav_idx % processing_num == processing_id:
                processing_token = True 
            else:
                processing_token = False
                
        if processing_token:
            wav_list = wav_lines[wav_idx].strip().split(' ')
            os.system('mkdir -p {}'.format(os.path.join(output_root,'/'.join(wav_list[0].split('/')[-5:-1]))))
            signal_list = []
            for f in wav_list:
                _, data = wf.read(f)
                if data.dtype == np.int16:
                    data = np.float32(data) / 32768
                signal_list.append(data)
            
            try:
                y = np.stack(signal_list, axis=0)
            except:
                mlen = len(signal_list[0])
                for i in range(1, len(signal_list)):
                    mlen = min(mlen, len(signal_list[i]))
                for i in range(len(signal_list)):
                    signal_list[i] = signal_list[i][:mlen]
                y = np.stack(signal_list, axis=0)

            Y = stft(y, **stft_options).transpose(2, 0, 1)  # (freqyency bins,channel,frames)
            Z = wpe(Y, iterations=iterations, statistics_mode='full').transpose(1, 2, 0)
            z = istft(Z, size=stft_options['size'], shift=stft_options['shift'])

            for d in range(len(signal_list)):
                store_path = wav_list[d].replace(data_root, output_root)
                tmpwav = np.int16(z[d,:] * 32768)
                wf.write(store_path, sampling_rate, tmpwav)
    return None

def wpe_manager(wav_scp, processing_num=1, data_root='MISP_121h', output_root='MISP_121h_WPE_',start=0,end=1):
    if processing_num > 1:
        pool = Pool(processes=processing_num)
        for i in range(processing_num):
            pool.apply_async(wpe_worker, kwds={
                'wav_scp': wav_scp, 'processing_id': i, 'processing_num': processing_num, 'data_root': data_root,
                'output_root': output_root})
        pool.close()
        pool.join()
    else:
        wpe_worker(wav_scp, data_root=data_root, output_root=output_root)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser('run_wpe')
    parser.add_argument('--root_path', type=str, default='/yrfs1/intern/gzzou2/MISP_TEST', help='root path')
    parser.add_argument('--nj', type=int, default=1, help='number of process')
    args = parser.parse_args()
    root_path = args.root_path
    processing_num = args.nj
    data_root = os.path.join(root_path,'')
    output_root = os.path.join(root_path,'dev_eval/WPE')
    wav_scp = os.path.join(root_path,'dev_eval/wpe.scp')
    wpe_manager(wav_scp=wav_scp, processing_num=processing_num, data_root=data_root, output_root=output_root)
