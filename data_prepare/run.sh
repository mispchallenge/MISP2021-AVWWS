set -e
cd data_prepare

python_path=  # e.g. ./python/bin/
root_path=  # e.g.  save data dir
data_root = # e.g. ./MISP2021_AVWWS

# add reverb
${python_path}python create_reverb_scp.py --root_path $root_path --data_root $data_root
${python_path}python add_reverb.py  --root_path $root_path --room_path ./all_room_info.txt

# add noise
${python_path}python create_noise_scp.py --mod clean --root_path $root_path
${python_path}python create_noise_scp.py --mod noise --root_path $root_path
${python_path}python process_wav_raw.py --data_type noise --root_path $root_path
${python_path}python process_wav_raw.py --data_type clean --root_path $root_path
${python_path}python add_noise.py --root_path $root_path

# wpe
${python_path}python create_wpe_scp.py --root_path $root_path
${python_path}python run_wpe.py --root_path $root_path

# beamforming
${python_path}python create_beam_scp.py --root_path $root_path
${python_path}python run_beamforming.py --root_path $root_path

# dev test wpe and beamforming
${python_path}python dev_eval/create_dev_eval_wpe_scp.py  --data_root $data_root  --root_path $root_path
${python_path}python dev_eval/run_wpe.py  --root_path $root_path
${python_path}python dev_eval/create_beam_scp.py  --root_path $root_path
${python_path}python dev_eval/run_beamforming.py  --root_path $root_path
${python_path}python dev_eval/create_dev_eval_scp.py  --root_path $root_path







