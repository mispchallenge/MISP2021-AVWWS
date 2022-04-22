from reader.data_reader_audio_pitch_time import myDataset as myDataset_pitch_time
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
from tools.data_utils import MyDistributedSampler
import torch.distributed as dist
import torch.utils.data.distributed


# python -m torch.distributed.launch --nproc_per_node=4 --master_port=12335 train.py
def main(args):
    use_cuda = args.use_cuda    
    gpu_id,gpu_ids = 0,args.gpu_id   

    if use_cuda:
        init_dist(args, gpu_ids)     
        gpu_id = gpu_ids[args.local_rank]
    model_path = args.project
    log_dir = args.logdir
    logger = utils.get_logger(log_dir + '/' + args.project)
    seed_torch(args.seed)

    root_path = args.root_path
    fb40_train_mean_var = np.load(os.path.join(root_path,'train_mean_var_fb40_.npz'))
    fb40_train_mean = fb40_train_mean_var['_mean']   # (40,)
    fb40_train_var = fb40_train_mean_var['_var']     # (40,)

    file_train_positive_path = os.path.join(root_path,'positive_train.scp')   
    file_train_negative_path = os.path.join(root_path,'negative_train.scp')
    file_dev_positive_far_path = os.path.join(root_path,'dev_positive_far.scp')
    file_dev_negative_far_path = os.path.join(root_path,'dev_negative_far.scp')
    
    file_dev_positive_near_path = os.path.join(root_path,'dev_positive_near.scp')
    file_dev_negative_near_path = os.path.join(root_path,'dev_negative_near.scp')

    file_dev_positive_middle_path = os.path.join(root_path,'dev_positive_middle.scp')
    file_dev_negative_middle_path = os.path.join(root_path,'dev_negative_middle.scp')
    scale = args.scale

    print("loading the dataset ...")
    dataset_train = myDataset_pitch_time(file_train_positive_path, file_train_negative_path, fb40_train_mean, fb40_train_var,scale)
    dataset_dev_far = myDataset(file_dev_positive_far_path, file_dev_negative_far_path, fb40_train_mean, fb40_train_var)
    dataset_dev_near = myDataset(file_dev_positive_near_path, file_dev_negative_near_path, fb40_train_mean, fb40_train_var)
    dataset_dev_middle = myDataset(file_dev_positive_middle_path, file_dev_negative_middle_path, fb40_train_mean, fb40_train_var)

    distribute_train_sampler = MyDistributedSampler(dataset_train) 
    distribute_dev_sampler_far = MyDistributedSampler(dataset_dev_far)
    distribute_dev_sampler_near = MyDistributedSampler(dataset_dev_near)
    distribute_dev_sampler_middle = MyDistributedSampler(dataset_dev_middle)
  
    dataloader_train = myDataLoader(dataset=dataset_train,
                            batch_size=args.minibatchsize_train,
                            sampler=distribute_train_sampler,
                            shuffle=False,
                            num_workers=args.train_num_workers,
                            drop_last=True)
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

    all_file = len(dataloader_train)  
    dataloader_list = [dataloader_dev_far,dataloader_dev_near,dataloader_dev_middle]
    dist_list = ['far','near','middle']
    print("- training samples {} ,  training batch {}".format(len(dataset_train), len(dataloader_train)))
    print("- dev far samples {} , dev batch {} ".format(len(dataset_dev_far), len(dataloader_dev_far)))
    print("- dev near samples {} , dev batch {} ".format(len(dataset_dev_near), len(dataloader_dev_near)))
    print("- dev middle samples {} , dev batch {} ".format(len(dataset_dev_middle), len(dataloader_dev_middle)))

    nnet = KWS_Net(args)    

    if use_cuda:
        nnet = nnet.cuda(gpu_id)  
        if len(gpu_ids) > 1:
            nnet = torch.nn.parallel.DistributedDataParallel(nnet, device_ids=[gpu_id], output_device = gpu_id)
   
    # training setups
    optimizer = optim.Adam(nnet.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=args.lr_scheduler) 
    BCE_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))  

    dev_loss = float('inf')  
    pbar = tqdm(range(args.end_iter)) 
    print(' traing start --------')
    for iter_ in pbar:
        start_time = time.time()
        if args.distributed:
            distribute_train_sampler.set_epoch(iter_)

        running_loss = 0.0  
        nnet.train()
        num_t = 0  
    
        for audio_feature, data_label_torch, current_frame in dataloader_train:
            optimizer.zero_grad()
            audio_feature = audio_feature.cuda(gpu_id) 
            data_label_torch = data_label_torch.cuda(gpu_id) 

            # forward
            outputs = nnet(audio_feature, current_frame)
            # outputs = nnet(audio_feature)
            loss = BCE_loss(outputs, data_label_torch)

            loss.backward()  
            optimizer.step() 

            running_loss += loss.item()
            num_t += 1
            if (num_t % 100 == 0 or num_t == all_file):
                logger.info("Iteration:{0}, chuck: {1}/{2}, loss = {3:.6f} ".format(iter_, num_t, all_file, running_loss/num_t))
        
        # eval
        nnet.eval()
        with torch.no_grad():
            cur_devloss = 0.0
            for i,dataloader_dev in enumerate(dataloader_list):
                pre_list, pre_list_d, label_list = [], [], []
                running_loss_dev = 0.0
                pre_sum = 0.0
                for audio_feature_dev, data_label_torch_dev, current_frame_dev in dataloader_dev:
                    audio_feature_dev = audio_feature_dev.cuda(gpu_id)
                    data_label_torch_dev = data_label_torch_dev.cuda(gpu_id)

                    outputs_dev = nnet(audio_feature_dev, current_frame_dev)
                    loss_dev = BCE_loss(outputs_dev, data_label_torch_dev)

                    running_loss_dev += loss_dev.item()
                    outputs_dev_np = (torch.ceil(torch.sigmoid(outputs_dev)-0.5)).data.cpu().numpy()
                    
                    pre_list.extend(outputs_dev_np[:,0])
                    label_list.extend(list(data_label_torch_dev.data.cpu().numpy()))

                cur_devloss += running_loss_dev

                TP, FP, TN, FN = utils.cal_indicator(pre_list, label_list)
                print(TP, FP, TN, FN)

                accuracy = (TP+TN)/(TP+FN+TN+FP)

                if (TP+FP) == 0:
                    precision = 0
                else:
                    precision = TP/(TP+FP)

                recall = TP/(TP+FN)        

                FPR = FP/(FP+TN) 
                TPR = TP/(TP+FN)

                if (TP+FP) == 0:
                    FA = 0
                else:
                    FA = TP/(TP+FP)
                all_file_dev = len(dataloader_dev)
                logger.info("data_dist %s Epoch:%d,  Valid loss=%.4f, Acc=%.4f, Pre=%.4f, Recall=%.4f, FPR=%.4f, TPR:%.4f" %(dist_list[i],iter_,
                     running_loss_dev/all_file_dev, accuracy, precision, recall, FPR, TPR))

        if dev_loss <= cur_devloss:
            scheduler.step()
        dev_loss = cur_devloss

        end_time = time.time()
        logger.info("Time used for each epoch training: {} seconds.".format(end_time - start_time))
        logger.info("*" * 50)

        if dist.get_rank()==0:
            torch.save(nnet.module.state_dict(), os.path.join(model_path, "{}_{}.pt".format(args.model_name, iter_)))
       
        end_time = time.time()
        logger.info("Time used for each epoch training: {} seconds.".format(end_time - start_time))
        logger.info("*" * 50)

def init_dist(args, gpu_ids):
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])    
        args.rank = int(os.environ["RANK"])
        torch.cuda.set_device(gpu_ids[args.local_rank])  
    args.distributed = args.world_size > 0
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)

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
    # Arguement Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr",default=2e-4, type=float, help="learning rate") # 2e-4 -> 2e-3
    parser.add_argument("--lr_scheduler",default=0.7,type=int)
    parser.add_argument("--minibatchsize_train", default=64, type=int)  # 64改为150
    parser.add_argument("--minibatchsize_dev", default=1, type=int)
    parser.add_argument("--logdir", default='./log/', type=str)
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
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--use_cuda", default=True, help="use gpu")
    parser.add_argument("--gpu_id", default=[0,1,2,3], type=list, help="gpu id")
    parser.add_argument("--train_num_workers", default=8, type=int, help="number of training workers")
    parser.add_argument("--dev_num_workers", default=8, type=int, help="number of validation workers")
    parser.add_argument("--scale", default=10, type=int, help="DA scale")
    parser.add_argument("--root_path", default='/yrfs1/intern/gzzou2/MISP_TEST', type=str)
    parser.add_argument('--world_size', default=-1, type=int, help='number of nodes for distributed training') 
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')    
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training') 
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend') 
    parser.add_argument("--local_rank", type=int)  
    args = parser.parse_args() 
    main(args)
