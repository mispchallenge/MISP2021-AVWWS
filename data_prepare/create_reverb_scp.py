import os
import glob
import argparse

def create_scp(data_path,mod,root_path):
    f = open(os.path.join(root_path,mod+'.scp'),'w')
    for root, dirs, files in os.walk(data_path):
        files.sort()
        for file in files:
            f.write(root + file + '\n')
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='/yrfs1/intern/gzzou2/hszhou2/MISP2021/dataset_misp2021', type=str, help="scp dir")
    parser.add_argument("--root_path", default='/yrfs1/intern/gzzou2/MISP_TEST', type=str, help="scp dir")
    args = parser.parse_args() 
    data_root = args.data_root
    root_path = args.root_path
    positive_path = os.path.join(data_root,'MISP2021_AVWWS/positive/audio/train/near/')
    negative_path = os.path.join(data_root,'MISP2021_AVWWS/negative/audio/train/near/')
    create_scp(positive_path,'positive',root_path)
    create_scp(negative_path,'negative',root_path)
    print('finish!')