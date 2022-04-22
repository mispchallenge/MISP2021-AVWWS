import os
import glob
import argparse
from scipy.io import wavfile
import numpy as np
import wave
import time,math

def timesince(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m*60
    return '%dm %ds' % (m,s)

# 生成配置文件
def generate_file(noise,clean,outpath,conf_name_list):
    file_list = [noise,clean,outpath]
    for i,file_name in enumerate(conf_name_list):
        f = open(file_name,'w')
        f.write(file_list[i])
        f.close()
    
# 生成6个通道的噪音文件
def generate_mulchannel_noises(noise_path):
    noise_spilit = noise_path.split('/')
    temp = noise_spilit[-1].split('_')
    temps,noises= [],[]
    for i in range(0,6):
        temp[-2] = str(i)
        temps.append(temp[:])
    for temp in temps:
        noise_spilit[-1] = '_'.join(temp)
        noises.append('/'.join(noise_spilit))
    return noises

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

# 生成6个通道的clean以及输出路径文件
def generate_mulchannel_cleans(clean_path,save_data_root):
    os.system('mkdir -p {}'.format(os.path.join(save_data_root,'/'.join(clean_path.split('/')[-5:-1]))))
    cleans_split = clean_path.split('_')
    temp = list(cleans_split[-1])
    temps,cleans,output = [],[],[]
    for i in range(0,6):
        temp[2] = str(i)
        temps.append(temp[:])
    for i,temp in enumerate(temps):
        cleans_split[-1] = ''.join(temp)
        cleans.append('_'.join(cleans_split))
        output.append(os.path.join(save_data_root,'/'.join(cleans[i].split('/')[-5:])))
    return cleans,output

def get_noise(clean,noises_files,root_path):
    noise_num,clean_file = len(noises_files), open(clean,'rb')
    lenclean,noise,count = len(clean_file.read()),'',1000
    clean_file.close()
    while count:
        noise = noises_files[np.random.randint(0,noise_num)]
        noise_file = open(noise,'rb')
        lennoise = len(noise_file.read())
        noise_file.close()
        if lenclean <= lennoise:
            return noise
        count -= 1
    max_noise_file = os.path.join(root_path,'noise/raw/noise_raw/R15_S233234235236_Far_0_205.raw')
    return max_noise_file

def main(args):
    root_path = args.root_path
    tool_path = args.tool_path
    
    save_data_root_ = os.path.join(root_path,'noise/add_noise')
    noise_scp = os.path.join(root_path,'noise_raw.scp')
    clean_scp = os.path.join(root_path,'clean_raw.scp')
    configure = os.path.join(root_path,'scp_dir/configurefile')
    log = os.path.join(root_path,'scp_dir/logfiles')

    os.system('mkdir -p {}'.format(configure))
    os.system('mkdir -p {}'.format(log))

    for snr in [-15,-10,-5,0,5,10,15]:
        count = 0
        
        save_data_root = os.path.join(save_data_root_,'SNR_'+str(snr))
        os.system('mkdir -p {}'.format(save_data_root))
        log_path =  os.path.join(log,'SNR{}_{}_file.scp'.format(snr,'log'))

        namelist = ['noise','clean','out']
        conf_name_list = []
        for name in namelist:
            conf_name_list.append(os.path.join(configure,'SNR{}_{}_file.scp'.format(snr,name)))

        with open(noise_scp) as f:
            noises_files = f.readlines()
        noises_files = [line.strip() for line in noises_files]
        noise_num = len(noises_files)


        with open(clean_scp) as f:
            cleans_files_temp = f.readlines()
        cleans_files = cleans_files_temp
        cleans_files = [line.strip() for line in cleans_files]

        start_time = time.time()

        for it,clean in enumerate(cleans_files):

            noise = get_noise(clean,noises_files,root_path)
            noises,(cleans,output)  = generate_mulchannel_noises(noise),generate_mulchannel_cleans(clean,save_data_root=save_data_root)
            for i,file in enumerate(noises):
                noise,clean,outpath = noises[i],cleans[i],output[i]
                generate_file(noise,clean,outpath,conf_name_list)
                print(outpath)
                os.system("{} -i {} -o {} -n {} -s {} -r 1000 -multiple {} -m snr_8khz -u -d -e {}".
                    format(tool_path,conf_name_list[1],conf_name_list[2],noise,snr,1,log_path))
                raw2wav(outpath,outpath.replace('.raw','.wav'))
                os.system('rm {}'.format(outpath))
            
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default='/yrfs1/intern/gzzou2/MISP_TEST', type=str)
    parser.add_argument("--tool_path", default='./AddNoise_MultiOutput', type=str)

    args = parser.parse_args()  
    main(args)
    print('finish!')




































