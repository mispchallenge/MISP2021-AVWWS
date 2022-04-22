import os
import glob
import argparse
from scipy.io import wavfile
import numpy as np
import wave,random


# def create_scp(data_root,scp_dir_root,ps='positive',mod='dev',dist='near'):
#     f = open(os.path.join(scp_dir_root,'wpe.scp') ,'w')
#     for root, dirs, files in os.walk(data_root):
#         files.sort()
#         for file in files:
#             file_name  = os.path.join(root ,file)
#             if file.endswith('0.wav'):  
#                 for name in range(0,5):
#                     f.write(file_name.replace('0.wav','{}.wav'.format(name)) + ' ')            
#             f.write(file_name.replace('0.wav','5.wav') + '\n')
#     f.close()

def create_scp(data_root,scp_dir_root):
    f = open(os.path.join(scp_dir_root,'wpe.scp') ,'w')
    for root, dirs, files in os.walk(data_root):
        if 'audio' in root and 'train' not in root and 'near' not in root:
            channel_num = 6
            if 'middle' in root:
                channel_num = 2

            files.sort()
            for file in files:
                file_name  = os.path.join(root ,file)
                if file.endswith('0.wav'):  
                    for name in range(0,channel_num-1):
                        f.write(file_name.replace('0.wav','{}.wav'.format(name)) + ' ')            
                f.write(file_name.replace('0.wav','{}.wav'.format(channel_num-1)) + '\n')
    for root, dirs, files in os.walk(data_root):
        if 'audio' in root and 'train' not in root and 'near' in root:
            for file in files:
                file_name  = os.path.join(root ,file)
                if file.endswith('.wav'):  
                    f.write(file_name + '\n')
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',default='/yrfs1/intern/gzzou2/MISP_TEST',type=str, help="root path")
    parser.add_argument('--data_root',default='/yrfs1/intern/gzzou2/hszhou2_dataset_2022/MISP2021_AVWWS',type=str, help="root path")
    args = parser.parse_args() 
    root_path = args.root_path
    data_root = args.data_root
    root_path = os.path.join(root_path,'dev_eval')
    os.system('mkdir -p {}'.format(root_path))
    create_scp(data_root,root_path)
    print('finish!')

    # create_scp(data_root=data_root,scp_dir_root=args.root_path,ps='positive',mod='dev',dist='near')
    # create_scp(data_root=data_root,scp_dir_root=args.root_path,ps='negative',mod='dev',dist='near')
    # create_scp(data_root=data_root,scp_dir_root=args.root_path,ps='positive',mod='dev',dist='middle')
    # create_scp(data_root=data_root,scp_dir_root=args.root_path,ps='negative',mod='dev',dist='middle')
    # create_scp(data_root=data_root,scp_dir_root=args.root_path,ps='positive',mod='dev',dist='far')
    # create_scp(data_root=data_root,scp_dir_root=args.root_path,ps='negative',mod='dev',dist='far')

    # create_scp(data_root=data_root,scp_dir_root=args.root_path,ps='positive',mod='eval',dist='near')
    # create_scp(data_root=data_root,scp_dir_root=args.root_path,ps='negative',mod='eval',dist='near')
    # create_scp(data_root=data_root,scp_dir_root=args.root_path,ps='positive',mod='eval',dist='middle')
    # create_scp(data_root=data_root,scp_dir_root=args.root_path,ps='negative',mod='eval',dist='middle')
    # create_scp(data_root=data_root,scp_dir_root=args.root_path,ps='positive',mod='eval',dist='far')
    # create_scp(data_root=data_root,scp_dir_root=args.root_path,ps='negative',mod='eval',dist='far')
