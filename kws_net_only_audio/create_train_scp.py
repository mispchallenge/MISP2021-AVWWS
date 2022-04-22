import os
import argparse


def create_train_scp(root_path,data_path,mod):
    f = open(os.path.join(root_path,'{}_train.scp'.format(mod)),'w')   
    
    root_path = os.path.join(root_path,'Beamforming')
    for root, dirs, files in os.walk(root_path):
        if mod in root:
            files.sort()
            for file in files:
                f.write(os.path.join(root,file) + '\n')

    for root, dirs, files in os.walk(data_path):
        if mod in root and 'far' not in root and 'train' in root and 'audio' in root:
            files.sort()
            for file in files:
                f.write(os.path.join(root,file) + '\n')

    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/yrfs1/intern/gzzou2/MISP_TEST', help='root path')
    parser.add_argument("--data_root", default='/yrfs1/intern/gzzou2/hszhou2_dataset_2022/MISP2021_AVWWS', type=str, help="scp dir")
    args = parser.parse_args()
    root_path = args.root_path
    data_path = args.data_root
    
    create_train_scp(root_path,data_path,'positive')
    create_train_scp(root_path,data_path,'negative')
    print('Finish!')
