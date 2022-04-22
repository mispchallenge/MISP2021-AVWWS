from reader.data_reader_audio import myDataLoader, myDataset
import argparse
import os
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import sys
import copy
import random
from tqdm import tqdm
sys.path.append("..")
from tools import utils
from model.audio_kwsnet import KWS_Net
from model.KWS import KWS
from tools.data_utils import MyDistributedSampler
import torch.distributed as dist
import torch.utils.data.distributed

# python decode.py --project $project

def main(args):

    model_path = args.project

    root_path = args.root_path
    fb40_train_mean_var = np.load(os.path.join(root_path,'train_mean_var_fb40_.npz'))    # fb40_train_mean_var = np.load('./dataset_respilt_dev_eval/generate_dataset_av/train_mean_dev_fb40_id2.npz')
    fb40_train_mean = fb40_train_mean_var['_mean']   # (40,)
    fb40_train_var = fb40_train_mean_var['_var']     # (40,)

    file_dev_positive_far_path = os.path.join(root_path,'dev_positive_far.scp')
    file_dev_negative_far_path = os.path.join(root_path,'dev_negative_far.scp')
    
    file_dev_positive_near_path = os.path.join(root_path,'dev_positive_near.scp')
    file_dev_negative_near_path = os.path.join(root_path,'dev_negative_near.scp')

    file_dev_positive_middle_path = os.path.join(root_path,'dev_positive_middle.scp')
    file_dev_negative_middle_path = os.path.join(root_path,'dev_negative_middle.scp')

    # define the dataloader
    print("loading the dataset ...")

    dataset_dev_far = myDataset(file_dev_positive_far_path, file_dev_negative_far_path, fb40_train_mean, fb40_train_var)
    dataset_dev_near = myDataset(file_dev_positive_near_path, file_dev_negative_near_path, fb40_train_mean, fb40_train_var)
    dataset_dev_middle = myDataset(file_dev_positive_middle_path, file_dev_negative_middle_path, fb40_train_mean, fb40_train_var)


    distribute_dev_sampler_far = MyDistributedSampler(dataset_dev_far)
    distribute_dev_sampler_near = MyDistributedSampler(dataset_dev_near)
    distribute_dev_sampler_middle = MyDistributedSampler(dataset_dev_middle)
  

    dataloader_dev_far = myDataLoader(dataset=dataset_dev_far,
                            batch_size=args.minibatchsize_dev,
                            sampler=distribute_dev_sampler_far,
                            shuffle=False,
                            num_workers=args.dev_num_workers,
                            drop_last=False)
    dataloader_dev_near = myDataLoader(dataset=dataset_dev_near,
                            batch_size=args.minibatchsize_dev,
                            sampler=distribute_dev_sampler_near,
                            shuffle=False,
                            num_workers=args.dev_num_workers,
                            drop_last=False)
    dataloader_dev_middle = myDataLoader(dataset=dataset_dev_middle,
                            batch_size=args.minibatchsize_dev,
                            sampler=distribute_dev_sampler_middle,
                            shuffle=False,
                            num_workers=args.dev_num_workers,
                            drop_last=False)
    dataloader_list = [dataloader_dev_far,dataloader_dev_near,dataloader_dev_middle]
    dist_list = ['far','near','middle']


    print("- dev far samples {} , dev batch {} ".format(len(dataset_dev_far), len(dataloader_dev_far)))
    print("- dev near samples {} , dev batch {} ".format(len(dataset_dev_near), len(dataloader_dev_near)))
    print("- dev middle samples {} , dev batch {} ".format(len(dataset_dev_middle), len(dataloader_dev_middle)))
    
    nnet = KWS_Net(args=args)
    nnet = nnet.cuda()

    dic = {'audio':[0, 0, 1, 1]}
    epoch = 0
    start_time = time.time()

    for imodel in range(args.trained_model_num): 
        trained_model = os.path.join(model_path, "{}_{}.pt".format(args.model_name, imodel))
        trained_kws_model =  torch.load(trained_model)
        nnet.load_state_dict(trained_kws_model)
        nnet.eval()
        pre_list, pre_list_d, label_list = [], [], []
        with torch.no_grad():
            for feature, data_label, current_frame in dataloader_dev_near:
                feature = feature.cuda()
                data_label = data_label.cuda()
                outputs = nnet(feature, current_frame)
                pre_list_d.append((torch.sigmoid(outputs)).data.cpu().numpy())
                label_list.extend(list(data_label.data.cpu().numpy()[0]))
            
            for threshold in range(args.threshold_num):
                threshold /= args.threshold_num
                audio_score = list(np.ceil(np.array(pre_list_d) - threshold))
                audio_score = [int(ii[0][0]) for ii in audio_score]
                dic['audio'],my_list= utils.cal_score2(imodel, threshold, dic, audio_score, label_list)

            
    print(" " * 100)
    print(" " * 100)
    print("Finally, for the audio, far result: epoch = %d, threshold = %.3f, FAR=%.4f, FRR:%.4f, Score:%4f" %(dic['audio'][0], dic['audio'][1], dic['audio'][2], dic['audio'][3], dic['audio'][2]+dic['audio'][3]))
    # np.save('/yrfs1/intern/gzzou2/code/demo/dlpdir/logdir/decodedir/audio_decode/pitch_time/pitch_time/ori_npy/eval_near_id16',dic_list)

    end_time = time.time()
    print("Time used {} seconds.".format(end_time - start_time))
        
    print("*" * 100)
    print(" " * 100)
    #############################################################################################
    #############################################################################################


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default=2e-4, type=float, help="learning rate") # 2e-4 -> 2e-3
    parser.add_argument("--lr_scheduler",default=0.7,type=int)
    parser.add_argument("--minibatchsize_train", default=64, type=int)  # 64改为150
    parser.add_argument("--minibatchsize_dev", default=1, type=int)
    parser.add_argument("--input_dim", default=256, type=int)
    parser.add_argument("--hidden_sizes", default=256, type=int)
    parser.add_argument("--output_dim", default=1, type=int)
    parser.add_argument("--video_encoder_dim", default=256, type=int)
    parser.add_argument("--lstm_num_layers", default=1, type=int)
    parser.add_argument("--seed", default=617, type=int)
    parser.add_argument("--project", default='save_audio_model', type=str)  
    parser.add_argument("--model_name", default='KWS_Lite_Net', type=str)
    parser.add_argument("--start_iter", default=0, type=int)
    parser.add_argument("--end_iter", default=20, type=int)
    parser.add_argument("--train_num_workers", default=1, type=int, help="number of training workers")
    parser.add_argument("--dev_num_workers", default=1, type=int, help="number of validation workers")
    
    #############################################################################################
    parser.add_argument("--threshold_num", default=100, type=int, help="find model threshold")
    parser.add_argument("--trained_model_num", default=10, type=int, help="video trained model num")
    #############################################################################################
    args = parser.parse_args() 
    # run main
    main(args)
