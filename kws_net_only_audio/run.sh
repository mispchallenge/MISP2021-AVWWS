set -e

python_path=""  # e.g. ./python/bin/
root_path=""  # e.g.  save data dir
project="save_audio_model"
logdir="log"

${python_path}python create_train_scp.py --root_path $root_path --data_root $data_root
 
if [ ! -e "${root_path}/train_mean_var_fb40_.npz" ]; then
  ${python_path}python get_mean_var_audio.py --root_path $root_path
fi

mkdir -p $project
mkdir -p $logdir
${python_path}python -m torch.distributed.launch --nproc_per_node=1 --master_port=12335 train.py --project $project --logdir $logdir
${python_path}python decode.py --project $project --trained_model_num 10
