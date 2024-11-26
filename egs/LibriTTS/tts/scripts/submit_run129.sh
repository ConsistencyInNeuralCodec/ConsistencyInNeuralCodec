#!/bin/bash
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
ENV=funasr
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

source /cpfs01/shared/public/renjun.admin/anaconda3/bin/activate
source activate ${ENV}
conda activate ${ENV}
# /cpfs01/shared/Group-m6-pod3/renjun.admin/anaconda3/bin/activate
#
export PATH=/cpfs01/shared/public/renjun.admin/anaconda3/envs/${ENV}/bin:$PATH
which python

CODE_REPO=FunCodec-Dev
RAW_CODE_ROOT=/home/admin_data/renjun.admin/projects/${CODE_REPO}

# Move code to local machine to avoid conflicts when running.
CODE_ROOT=/home/E-renjun.admin-400833/Projects/${CODE_REPO}
mkdir -p ${CODE_ROOT}
cp -r ${RAW_CODE_ROOT}/. ${CODE_ROOT}/.

## For JupyterLab Usage :
# TENSORBOARD_ROOT=/mnt/user/E-renjun.admin-400833
# mkdir -p /home/E-renjun.admin-400833/Projects
# cd $CODE_ROOT
# pwd
# ls -l
# mlsen_5778_6319_001152
##############################################################################
cd /home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts
pwd
ls -l

. ./path.sh || exit 1;

# machines configuration
gpu_devices="0,1,2,3,4,5,6,7"
gpu_num=8
count=1

# general configuration
feats_dir="/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts"
exp_dir="/home/admin_data/renjun.admin/checkpoints/fun_uniaudio"
dumpdir=dump/LibriTTS
corpus_dir=LibriTTS

ttsfrd_rsc_zip_file=/home/admin_data/renjun.admin/dataset/fun_tts_resource/tts/resource.zip
ttsfrd_config_file=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/conf/ali_tokenizer.json
codec_model_dir=/home/admin_data/renjun.admin/dataset/fun_tts_resource/ptts_encodec_16k_n32_600k

# training related
tag="valle_libritts_8gpu_b24"
train_set=train
valid_set=dev
train_config=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/conf/valle_16k_n8_ds320_a100_scaledadam.yaml

init_param=


# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=5
docker_nj=32
bit_width=4000
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
echo "stage 6: Training"
mkdir -p ${exp_dir}/exp/${model_dir}
mkdir -p ${exp_dir}/exp/${model_dir}/log
INIT_FILE=${exp_dir}/exp/${model_dir}/ddp_init
if [ -f $INIT_FILE ];then
    rm -f $INIT_FILE
fi
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
        python -m funcodec.bin.tts_train \
            --gpu_id $gpu_id \
            --use_preprocessor false \
            --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/codec.scp,codec,kaldi_ark \
            --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/text.scp,text,kaldi_ark \
            --train_shape_file ${feats_dir}/${dumpdir}/${train_set}/codec_shape \
            --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/codec.scp,codec,kaldi_ark \
            --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/text.scp,text,kaldi_ark \
            --valid_shape_file ${feats_dir}/${dumpdir}/${valid_set}/codec_shape \
            ${init_opt} --ignore_init_mismatch true \
            --resume true \
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