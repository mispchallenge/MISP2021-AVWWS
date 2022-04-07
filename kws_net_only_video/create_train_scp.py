import os
import glob
import codecs
import argparse
import numpy as np

def find_npy(data_root, scp_dir, scp_name='positive'):
    all_wav_paths = []
    for i in data_root:
        all_wav_paths += glob.glob(i)
    
    lines = ['' for _ in range(1)]
    for wav_idx in range(len(all_wav_paths)):
        line = all_wav_paths[wav_idx]
        line += '\n'
        lines[0] += line
 
    if not os.path.exists(scp_dir):
        os.makedirs(scp_dir, exist_ok=True)

    with codecs.open(os.path.join(scp_dir, '{}.scp'.format(scp_name)), 'w') as handle:
        handle.write(lines[0][:-1])

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default='../get_lip_feature/MISP2021_AVWWS/', type=str)
    args = parser.parse_args()

    data_root = args.data_root
    
    # prepare middle field file
    positive_train_middle = [data_root+'positive/video/train/middle/*.npy']
    negative_train_middle = [data_root+'negative/video/train/middle/*.npy']
    positive_dev_middle = [data_root+'positive/video/dev/middle/*[0-9].npy']
    negative_dev_middle = [data_root+'negative/video/dev/middle/*[0-9].npy']
    positive_eval_middle = [data_root+'positive/video/eval/middle/*[0-9].npy']
    negative_eval_middle = [data_root+'negative/video/eval/middle/*[0-9].npy']
    find_npy(positive_train_middle, 'scp_dir', 'positive_train_mid')
    find_npy(negative_train_middle, 'scp_dir', 'negative_train_mid')
    find_npy(positive_dev_middle, 'scp_dir', 'positive_dev_mid')
    find_npy(negative_dev_middle, 'scp_dir', 'negative_dev_mid')
    find_npy(positive_eval_middle, 'scp_dir', 'positive_eval_mid')
    find_npy(negative_eval_middle, 'scp_dir', 'negative_eval_mid')

    # # prepare far field file
    positive_train_far = [data_root+'positive/video/train/far/*.npy']
    negative_train_far = [data_root+'negative/video/train/far/*.npy']
    positive_dev_far = [data_root+'positive/video/dev/far/*[0-9].npy']
    negative_dev_far = [data_root+'negative/video/dev/far/*[0-9].npy']
    positive_eval_far = [data_root+'positive/video/eval/far/*[0-9].npy']
    negative_eval_far = [data_root+'negative/video/eval/far/*[0-9].npy']
    find_npy(positive_train_far, 'scp_dir', 'positive_train_far')
    find_npy(negative_train_far, 'scp_dir', 'negative_train_far')
    find_npy(positive_dev_far, 'scp_dir', 'positive_dev_far')
    find_npy(negative_dev_far, 'scp_dir', 'negative_dev_far')
    find_npy(positive_eval_far, 'scp_dir', 'positive_eval_far')
    find_npy(negative_eval_far, 'scp_dir', 'negative_eval_far')
