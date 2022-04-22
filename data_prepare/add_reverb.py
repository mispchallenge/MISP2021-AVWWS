import os
import glob
import argparse
import numpy as np
import pyroomacoustics as pra
import random
import time,math
from scipy.io import wavfile

def read_roominfo(_rpath):
	tmp_dic = {}
	with open(_rpath) as fh:
		line = fh.readline()
		while line:
			tmp = line.split('\n')[0].split(' ')
			tmp_dic[tmp[0]] = tmp[1]
			line = fh.readline()
	return tmp_dic

def checkdir(directory):
    try:
        os.makedirs(directory)
    except OSError:
        pass

def generate_reverb(save_file_path,file_path,room_info,rt60):
    room_dim = []
    room_id = file_path.split('/')[-1].split('_')[0][1:]
    room_dim.append(float(room_info[room_id][0:3])/100)
    room_dim.append(float(room_info[room_id][4:7])/100)
    room_dim.append(float(room_info[room_id][8:11])/100)

    mic_locs = np.c_[
        [room_dim[0]/2+0.075, room_dim[1]-0.5, 0.80],  
        [room_dim[0]/2+0.050, room_dim[1]-0.5, 0.80],
        [room_dim[0]/2+0.025, room_dim[1]-0.5, 0.80],
        [room_dim[0]/2-0.025, room_dim[1]-0.5, 0.80],
        [room_dim[0]/2-0.050, room_dim[1]-0.5, 0.80],
        [room_dim[0]/2-0.075, room_dim[1]-0.5, 0.80],
    ]

    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(room_dim, fs=16000, materials=pra.Material(e_absorption), max_order=max_order)
    fs, audio = wavfile.read(file_path)
    wavlen = len(audio)

    speaker_location1 = random.uniform(-0.2,0.2)
    speaker_location2 = random.uniform(0.05,0.25)
    room.add_source([room_dim[0]/2+speaker_location1, speaker_location2, 1.00], signal=audio, delay=0.00)
    room.add_microphone_array(mic_locs)
    room.compute_rir()
    room.simulate()
    file_name = file_path.split('/')[-1].split('.')[0]
    for i in range(0,6):
        wavfile.write(os.path.join(save_file_path,file_name+'_ch'+str(i)+'.wav'),fs,room.mic_array.signals[i,:wavlen].astype(np.int16))

def main(args):
    mod = args.mod
    room_path = args.room_path
    root_path = args.root_path

    file_scp = [os.path.join(root_path,'positive.scp'),os.path.join(root_path,'negative.scp')]
    scales = [60,15]

    for i,label in enumerate(file_scp):
        if mod == label.split('/')[-1].split('.')[0]:

            with open(label) as f:
                files = f.readlines()

            files = [line.strip() for line in files]
            label = label.split('/')[-1].split('.')[0]
            save_path = os.path.join(root_path,'reverb/' + label + '/audio/train/')
            rt60_np = np.linspace(0.15,0.45,scales[i]) 
            for rt in rt60_np:
                os.system('mkdir -p {}'.format(save_path+'rt60_'+str(round(rt,4))))

            room_info = read_roominfo(room_path) 
            for it,file in enumerate(files):
                for rt60 in rt60_np:
                    generate_reverb(os.path.join(save_path,'rt60_'+str(round(rt60,4))),file,room_info,rt60)
    print("finish")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mod", default='positive', type=str, help="positive or negative")
    parser.add_argument("--room_path", default='./all_room_info.txt', type=str, help="room info")
    parser.add_argument("--root_path", default='/yrfs1/intern/gzzou2/MISP_TEST', type=str, help="room info")
    args = parser.parse_args() 
    main(args)
    args.mod = 'negative'
    main(args)

