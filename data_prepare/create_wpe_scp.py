import os
import glob
import argparse
from scipy.io import wavfile
import numpy as np
import wave,random

def create_scp(data_root,scp_dir_root):
    f = open(os.path.join(scp_dir_root,'wpe.scp') ,'w')
    for root, dirs, files in os.walk(data_root):
        files.sort()
        for file in files:
            file_name  = os.path.join(root ,file)
            if file.endswith('0.wav'):  
                for name in range(0,5):
                    f.write(file_name.replace('0.wav','{}.wav'.format(name)) + ' ')            
            f.write(file_name.replace('0.wav','5.wav') + '\n')
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',default='/yrfs1/intern/gzzou2/MISP_TEST',type=str, help="root path")
    args = parser.parse_args() 
    root_path = args.root_path
    scp_dir_root = root_path
    data_root = os.path.join(root_path,'noise/add_noise')
    create_scp(data_root=data_root,scp_dir_root=scp_dir_root)
    print('finish!')

