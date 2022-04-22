import os
import glob
import argparse
import soundfile as sf
import numpy as np

'''
noise_wav_path = '/yrfs1/intern/gzzou2/hszhou2/MISP2021/dataset/MISP2021_AVWWS/noise/audio/train/far/'
noise_wav_path = '/yrfs1/intern/gzzou2/hszhou2/MISP2021/dataset_misp2021/MISP2021_AVWWS/noise/audio/train/far'
clean_wav_path = '/yrfs1/intern/gzzou2/MISP_task1_data/reverb/'
'''

def create_scp(data_path,mod,root_path):
    f = open(os.path.join(root_path,mod+'.scp'),'w')    
    for root, dirs, files in os.walk(data_path):
        files.sort()
        for file in files:
            f.write(os.path.join(root,file) + '\n')
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", default='/yrfs1/intern/gzzou2/MISP_task1_data/reverb/', type=str, help="data root dir")
    parser.add_argument("--mod", default='clean', type=str, help="clean or noise")
    parser.add_argument('--root_path',default='/yrfs1/intern/gzzou2/MISP_TEST',type=str, help="scp dir")
    args = parser.parse_args() 
    mod = args.mod
    root_path = args.root_path
    data_path = os.path.join(root_path,'reverb')
    if mod =='noise':
        data_path = '/yrfs1/intern/gzzou2/hszhou2/MISP2021/dataset_misp2021/MISP2021_AVWWS/noise/audio/train/far'
    create_scp(data_path,mod,root_path)


