#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
gpu_devices="0"
gpu_num=1
count=1

# general configuration
feats_dir="/home/admin_data/renjun.admin/dataset/project_data/funcodec-dev/codec"
exp_dir="/home/admin_data/renjun.admin/checkpoints/funcodec"
dumpdir=dump/sq
state_dir=sq_states

# training related
tag="freqdebug2"
train_set=train
valid_set=dev
# train_config=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/codec/conf/encodec_16k_n32_600k_step_24k_debug.yaml
train_config=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/codec/conf/encodec_try7.yaml
init_param=

# Decoding: for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=5
docker_nj=32
infer_cmd=utils/run.pl
sample_frequency=16000
file_sampling_rate=16000
bit_width=4000
use_scale=false
use_ppg=false


model_dir=

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

if [ -z "${model_dir}" ]; then
  model_dir="$(basename "${train_config}" .yaml)${tag}"
fi


# Training Stage
world_size=$gpu_num  # run on one machine
echo "stage 3: Training"
mkdir -p ${exp_dir}/exp/${model_dir}
mkdir -p ${exp_dir}/exp/${model_dir}/log
INIT_FILE=${exp_dir}/exp/${model_dir}/ddp_init
if [ -f $INIT_FILE ];then
    rm -f $INIT_FILE
fi
ppg_opt=""
init_opt=""
if [ ! -z "${init_param}" ]; then
    init_opt="--init_param ${init_param}"
    echo ${init_opt}
fi

init_method=file://$(readlink -f $INIT_FILE)
echo "log can be found at ${exp_dir}/exp/${model_dir}/log/train.log.0"

# /home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/codec/dump/foo/train/libritts_train.scp
# Only for single gpu debug
rank=0
local_rank=0
gpu_id=0
python -m funcodec.bin.codec_train \
    --gpu_id $gpu_id \
    --use_preprocessor true \
    --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/train.scp,speech,sound \
    --train_shape_file ${feats_dir}/exp/${state_dir}/${train_set}/speech_shape \
    --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/val.scp,speech,sound \
    --valid_shape_file ${feats_dir}/exp/${state_dir}/${valid_set}/speech_shape \
    ${init_opt} --ignore_init_mismatch true \
    ${ppg_opt} --resume true \
    --output_dir ${exp_dir}/exp/${model_dir} \
    --config $train_config \
    --ngpu $gpu_num \
    --num_worker_count $count \
    --multiprocessing_distributed true \
    --dist_init_method $init_method \
    --dist_world_size $world_size \
    --dist_rank $rank \
    --local_rank $local_rank

# python -m funcodec.bin.codec_train \
#     --gpu_id $gpu_id \
#     --use_preprocessor true \
#     --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/wav.scp,speech,kaldi_ark \
#     --train_shape_file ${feats_dir}/exp/${state_dir}/${train_set}/speech_shape \
#     --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/wav.scp,speech,kaldi_ark \
#     --valid_shape_file ${feats_dir}/exp/${state_dir}/${valid_set}/speech_shape \
#     ${init_opt} --ignore_init_mismatch true \
#     ${ppg_opt} --resume true \
#     --output_dir ${exp_dir}/exp/${model_dir} \
#     --config $train_config \
#     --ngpu $gpu_num \
#     --num_worker_count $count \
#     --multiprocessing_distributed true \
#     --dist_init_method $init_method \
#     --dist_world_size $world_size \
#     --dist_rank $rank \
#     --local_rank $local_rank 1> ${exp_dir}/exp/${model_dir}/log/train.log.$i 2>&1



