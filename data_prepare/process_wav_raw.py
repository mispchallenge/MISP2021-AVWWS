import os
import glob
import argparse
from scipy.io import wavfile
import numpy as np
import wave

def wav2raw(wavfile,rawfile,data_type=np.int16):
    f = open(wavfile,'rb')
    f.seek(0)
    f.read(44)
    data = np.fromfile(f,dtype=data_type)
    data.tofile(rawfile)

def raw2wav(raw_file,wav_file,channels=1,bits=16,sample_rate=16000):
    raw = open(raw_file,'rb')
    raw_data = raw.read()
    raw.close()
    wavfile = wave.open(wav_file,'wb')
    wavfile.setnchannels(channels)
    wavfile.setsampwidth(bits//8)
    wavfile.setframerate(sample_rate)
    wavfile.writeframes(raw_data)
    wavfile.close()

parser = argparse.ArgumentParser()
parser.add_argument("--root_path", default='/yrfs1/intern/gzzou2/MISP_TEST', type=str)
parser.add_argument("--mode", default='wav2raw', type=str, help='wav2raw')
parser.add_argument("--data_type", default='noise', type=str, help='noise or clean')
args = parser.parse_args()  

root_path = args.root_path
data_type = args.data_type
wav_scp_path = os.path.join(root_path,data_type+'.scp')
mode = args.mode

if mode =='wav2raw':
    default_save_data_path =  os.path.join(root_path,'noise/raw/' + data_type + '_raw/')
    os.system('mkdir -p {}'.format(default_save_data_path))
    raw_scp_path = os.path.join(root_path,  data_type + '_raw' + '.scp')
    file = open(raw_scp_path,'w')
    with open(wav_scp_path) as f:
        files = f.readlines()
    wavs = files
    wavs = [line.strip() for line in wavs]
    line = ''
    save_data_path = default_save_data_path
    for wav in wavs:
        wav_split = wav.split('/')
        if data_type =='clean':
            save_data_path = os.path.join(default_save_data_path,'/'.join(wav_split[-5:-1]))
            os.system('mkdir -p {}'.format(save_data_path))
        raw = wav_split[-1].replace('.wav','.raw')
        wav2raw(wav,os.path.join(save_data_path,raw))
        line = line + os.path.join(save_data_path,raw) + '\n'
    file.write(line[:-1])
    file.close()
else:
    print("mode error!")

print('finish!')













