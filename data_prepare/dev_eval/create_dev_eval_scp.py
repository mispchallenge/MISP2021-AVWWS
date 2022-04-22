import os
import glob
import argparse
from scipy.io import wavfile
import numpy as np
import wave,random


def create_scp(data_root,ps='positive',mod='dev',dist='near'):
    f = open(os.path.join(data_root,'{}_{}_{}.scp'.format(mod,ps,dist)) ,'w')

    if dist == 'near':
        data_root = os.path.join(data_root,'WPE')
    else:
        data_root = os.path.join(data_root,'Beamforming')

    for root, dirs, files in os.walk(data_root):
        if ps in root and mod in root and dist in root:
            files.sort()
            for file in files:
                file_name  = os.path.join(root ,file)           
                f.write(file_name + '\n')
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',default='/yrfs1/intern/gzzou2/MISP_TEST',type=str, help="root path")
    args = parser.parse_args() 
    root_path = args.root_path
    root_path = os.path.join(root_path,'dev_eval')

    create_scp(data_root=root_path,ps='positive',mod='dev',dist='near')
    create_scp(data_root=root_path,ps='negative',mod='dev',dist='near')
    create_scp(data_root=root_path,ps='positive',mod='dev',dist='middle')
    create_scp(data_root=root_path,ps='negative',mod='dev',dist='middle')
    create_scp(data_root=root_path,ps='positive',mod='dev',dist='far')
    create_scp(data_root=root_path,ps='negative',mod='dev',dist='far')

    create_scp(data_root=root_path,ps='positive',mod='eval',dist='near')
    create_scp(data_root=root_path,ps='negative',mod='eval',dist='near')
    create_scp(data_root=root_path,ps='positive',mod='eval',dist='middle')
    create_scp(data_root=root_path,ps='negative',mod='eval',dist='middle')
    create_scp(data_root=root_path,ps='positive',mod='eval',dist='far')
    create_scp(data_root=root_path,ps='negative',mod='eval',dist='far')
    print('finish!')    