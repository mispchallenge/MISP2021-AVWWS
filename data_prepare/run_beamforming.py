import os
import codecs
import argparse,time,math
import numpy as np
import scipy.io.wavfile as wf
from multiprocessing import Pool

def readfile(_file):
	tmp_list = []
	with open(_file) as fh:
		line = fh.readline()
		while line:
			tmp_list.append(line.split('\n')[0]) 
			line = fh.readline()
	return tmp_list

def run_beamforming(wavscp,output_root,root_path,channel_dir='./channels_name_dir/',conf_dir='./configure_dir/'):

    os.system('mkdir -p {} ; mkdir -p {}'.format(channel_dir,conf_dir))
    file_list = readfile('./Beamformer_tool/cfg-files/RT06s_audio_samples_for_doing_beamforming.cfg')
    
    with open(wavscp,'r') as f:
        lines = f.readlines()
    wav_file_list = lines
    wav_file_list = [wav.strip() for wav in wav_file_list]

    for i,file in enumerate(wav_file_list):

        file_split = file.split('/')
        file_name = file_split[-1]

        channel_num = 6 
        name_tmp = '_ch0.wav'
        name_head = file_name.replace(name_tmp,'')  

        file_channel_name  = [] 
        file_channel_name.append(name_head +' ')   
        file_channel_name.extend([file_name.replace('_ch0','_ch{}'.format(i)) + ' ' if i != channel_num-1 else file_name.replace('_ch0','_ch{}'.format(i)) for i in range(0,channel_num)])

        channel_files = channel_dir + 'channels_conf'
        source_dir = '/'.join(file_split[:-1])
        result_dir = os.path.join(output_root, '/'.join(file_split[-6:-1])  + '/')
        os.system('mkdir -p {}'.format(result_dir))

        with open(channel_files,'w') as f:
            f.writelines(it for it in file_channel_name)
        
        configure = file_list[:49]

        configure.append('source_dir = {}'.format(source_dir) + '\n')
        configure.append('channels_file = {}'.format(channel_files) + '\n')
        configure.append('show_id = {}'.format(name_head + '\n'))
        configure.append('result_dir = {}'.format(result_dir) + '\n')
        configure_file = conf_dir + 'conf.cfg'

        with open(configure_file,'w') as fh:
            fh.writelines(it + '\n' for it in configure)
            
        os.system('./Beamformer_tool/BeamformIt -s {} \
            --config_file={} '.format(name_head,configure_file))

        rm_file = result_dir + name_head
        os.system('rm {} {} {} {}'.format(rm_file + '.del',rm_file + '.del2',rm_file + '.info',rm_file + '.weat'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('run_beamforming')
    parser.add_argument('--root_path', type=str, default='/yrfs1/intern/gzzou2/MISP_TEST', help='root path')
    args = parser.parse_args()


    root_path = args.root_path
    wav_scp = os.path.join(root_path,'beamforming.scp')
    data_root = os.path.join(root_path,'WPE')
    output_root = os.path.join(root_path,'Beamforming')
    
    channel_dir = os.path.join(root_path,'scp_dir/beam_channels_name_dir/')
    conf_dir =  os.path.join(root_path,'scp_dir/beam_configure_dir/')

    run_beamforming(wav_scp,output_root=output_root,root_path=root_path,channel_dir=channel_dir,conf_dir=conf_dir)
    print('Finish!')
