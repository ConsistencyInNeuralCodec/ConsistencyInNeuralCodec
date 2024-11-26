#!/bin/bash

# docker: reg.docker.alibaba-inc.com/pai-damo/pytorch-training:1.12PAI-gpu-py38-cu113-ubuntu20.04
# 执行命令: bash /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec/scripts/user/submit/1rvq/codec_quant64.sh

# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5
# export NCCL_IB_TC=136
# export NCCL_IB_SL=5
# export NCCL_IB_GID_INDEX=3
# export GLOO_SOCKET_IFNAME=bond0
# export NCCL_IB_TIMEOUT=20
export CUDA_DEVICE_MAX_CONNECTIONS=1
### Initilize conda and codes
ENV=llm
# DOCKER: reg.docker.alibaba-inc.com/pai-damo/pytorch-training:1.12PAI-gpu-py38-cu113-ubuntu20.0
# sudo sed -i "s|barrier_timeout: float = 300|barrier_timeout: float = 14400|g" /home/pai/lib/python3.6/site-packages/torch/distributed/elastic/utils/store.py
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export GLOO_SOCKET_IFNAME=bond0
export OMP_NUM_THREADS=4
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO

# Linux Install
# sudo apt-get update -y
# sudo apt-get install zip unzip -y
# sudo apt-get install curl -y

source /home/admin_data/user/opt/anaconda/bin/activate
source activate ${ENV}
conda activate ${ENV}
# /cpfs01/shared/Group-m6-pod3/renjun.admin/anaconda3/bin/activate
#
# export PATH=/cpfs01/shared/public/renjun.admin/anaconda3/envs/${ENV}/bin:$PATH
which python

CODE_REPO=FunCodec-Dev
RAW_CODE_ROOT=/home/admin_data/user/model/uniaudio_admin/${CODE_REPO}

# Move code to local machine to avoid conflicts when running.
CODE_ROOT=/home/admin_data/user/checkpoints/${CODE_REPO}
mkdir -p ${CODE_ROOT}
cp -r ${RAW_CODE_ROOT}/. ${CODE_ROOT}/.

## For JupyterLab Usage :
# TENSORBOARD_ROOT=/mnt/user/E-renjun.admin-400833
# mkdir -p /home/E-renjun.admin-400833/Projects
# cd $CODE_ROOT
# pwd
# ls -l
##############################################################################
cd /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec
pwd
ls -l
# TFBoard Path: /home/admin_data/renjun.admin/checkpoints/funcodec
#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
# gpu_devices="1,2,3"
# gpu_num=3
gpu_devices="0,1,2,3,4,5,6,7"
gpu_num=8
count=1

# general configuration
feats_dir="/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec/"
exp_dir="/home/admin_data/user/checkpoints/funcodec"
dataset_name="LibriTTS+MLS_en_2k"
dumpdir="LibriTTS+MLS_en_2k_shuffle"
state_dir="LibriTTS+MLS_en_2k_shuffle_states"

# training related
# tag="freq2wav"
tag="freq2wav-1rvq-quant64"
train_set=train
valid_set=dev
train_config=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec/conf/user/1rvq/freqcodec_mag2time_quant64.yaml
# train_config=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec/conf/user/freqcodec_mag2time_xvector.yaml
# train_config=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec/conf/user/freqcodec_mag2time_fastspeech.yaml
init_param=


# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
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

# you can set gpu num for decoding here
gpuid_list=$gpu_devices  # set gpus for decoding, the same as training stage by default
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')

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
    for ((i = 0; i < $gpu_num; ++i)); do
        {
            rank=$i
            local_rank=$i
            gpu_id=$(echo $gpu_devices | cut -d',' -f$[$i+1])
            python -m funcodec.bin.codec_train \
                --gpu_id $gpu_id \
                --use_preprocessor true \
                --train_data_path_and_name_and_type ${feats_dir}/dump/${dumpdir}/${train_set}/train.scp,speech,sound \
                --train_shape_file ${feats_dir}/exp/${state_dir}/${train_set}/speech_shape \
                --valid_data_path_and_name_and_type ${feats_dir}/dump/${dumpdir}/${valid_set}/val.scp,speech,sound \
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
                --local_rank $local_rank 1> ${exp_dir}/exp/${model_dir}/log/train.log.$i 2>&1
        } &
        done
        wait