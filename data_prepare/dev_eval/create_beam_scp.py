import os
import glob
import argparse
from scipy.io import wavfile
import numpy as np
import wave


def create_scp(data_path,scp_dir_root):
    f = open(os.path.join(scp_dir_root,'beamforming.scp'),'w')
    for root, dirs, files in os.walk(data_path):
        if 'near' not in root:
            files.sort()    
            for file in files:
                file_name = root + '/' + file
                if file.endswith('0.wav') :
                    f.write(file_name + '\n')
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',default='/yrfs1/intern/gzzou2/MISP_TEST',type=str, help="root path")
    args = parser.parse_args() 
    root_path = args.root_path
    root_path = os.path.join(root_path,'dev_eval')
    data_path = os.path.join(root_path,'WPE')
    scp_dir_root = root_path
    create_scp(data_path,scp_dir_root)
    print('finish!')
