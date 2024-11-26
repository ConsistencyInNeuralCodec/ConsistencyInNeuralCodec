#! /usr/bin/env bash
# bash /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/scripts/libritts+mls_en+libriheavy/libritts+mls_en+libriheavy_data_prep_phase_aug_0.3s.sh

# cd /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts
# bash scripts/libritts_data_prep2.sh
# pip install ttsfrd==0.0.4 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_BLOCKING_WAIT=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export GLOO_SOCKET_IFNAME=bond0
export OMP_NUM_THREADS=4
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO
ENV=llm
source /home/admin_data/user/opt/anaconda/bin/activate
source activate ${ENV}
conda activate ${ENV}

SECONDS=0

cd /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts

. ./path.sh || exit 1;

# machines configuration
# gpu_devices="0"
# gpu_num=1
gpu_devices="0,1,2,3"
gpu_num=4
# gpu_devices="0,1,2,3,4,5,6,7"
# gpu_num=8
# cpu_num=50
cpu_num=30

count=1

# general configuration
feats_dir="."
exp_dir="."

codec_model_root=/home/admin_data/user/checkpoints/funcodec_consistency_encodec_LibriTTS+MLS_en/exp
inference_model=7epoch.pth
corpus_dir=LibriTTS
dump_root=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/${corpus_dir}/codec/dump2
dataset_root=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/${corpus_dir}/tts

# codec_model_name=encoder_16k_n16_ds320_largev8_encodec_0518                                 # 1
# codec_model_name=encoder_16k_n16_ds320_largev8_consistent_0.2_quant_in_10.0_encodec             # 2
# codec_model_name=encoder_16k_n16_ds320_largev8_phase_aug_consistent_0.2_quant_in_10.0_encodec   # 3
# codec_model_name=encoder_16k_n16_ds320_largev8_consistent_0.4_quant_in_10.0_encodec             # 4
# codec_model_name=encoder_16k_n16_ds320_largev8_phase_aug_consistent_0.4_quant_in_10.0_encodec   #5
# odec_model_name=encoder_16k_n16_ds320_largev8_phase_aug_encodec
codec_model_name=encodec_16k_n16_ds320_largev8_phase_aug_consistent_0.3s_quant_in_10.0_encodec

# dump_dir=${dump_root}/${codec_model_name}
dump_dir=${dump_root}/16rvq/encodec_16k_n16_ds320_largev8_phase_aug_consistent_0.3s_quant_in_10.0_encodec
tokens_dir=${dump_root}/tokens
mkdir -p $dump_dir

rtage=1
stop_stage=1
ttsfrd_rsc_zip_file=/home/admin_data/user/dataset/fun_tts_resource/tts/resource.zip
ttsfrd_config_file=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/conf/ali_tokenizer.json
codec_model_dir=${codec_model_root}/${codec_model_name}


# hubert_km_path=/nfs/shenxiao.hk/project/tts-deploy/hubert_base_ls960_L9_km500.bin

# training related
tag=""
train_set=train
valid_set=dev
train_config=$codec_model_dir/config.yaml
init_param=

# inference related
inference_tag="inference"
batch_size=1
test_sets=="testsetA"
# test_sets="testsetB"
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding

# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=1
# njob=12
docker_nj=32
infer_cmd=utils/run.pl
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

if ${gpu_inference}; then
    inference_nj=$[${ngpu}*${njob}]
    _ngpu=1
else
    inference_nj=$njob
    _ngpu=0
fi

# Data downloading
subset_names=(dev test train)
# subset_names=(dev test)
# subset_names=(dev) # "${subset_names[@]}"

# Extract feature (text token)
# echo "Stage 2: extracting feature (text token)."
utt_data_root=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/dataset/$corpus_dir
mkdir -p ${tokens_dir}
# for name in "${subset_names[@]}"; do
#   echo "Process stage 2: "${name}
#   python scripts/data_prep.py trans_to_token $utt_data_root/${name}_text.scp ${tokens_dir}/${name}_text ${ttsfrd_rsc_zip_file} ${ttsfrd_config_file} --nj ${cpu_num}
# done


# Extract feature (codec)
echo "Stage 3: extracting feature (codec)."
mkdir -p ${dump_dir}
# bit_width=4000 # 500/q -> 8 codec
bit_width=8000 # 1000/q -> 16 codec
for name in "${subset_names[@]}"; do
  echo "Process stage 3: "${name}
  # python scripts/data_prep.py codec_enc --bin_dir ../codec --wav_scp $utt_data_root/${name}.scp --out_dir ${dump_dir}/${name}_codec --gpu_devices $gpu_devices --njob $gpu_num --model_dir ${codec_model_dir} --model_name=$inference_model --bit_width ${bit_width}
  python scripts/data_prep.py codec_enc --bin_dir ../codec --wav_scp $utt_data_root/${name}.scp --out_dir ${dump_dir}/${name}_codec --gpu_devices $gpu_devices --njob 4 --model_dir ${codec_model_dir} --model_name=$inference_model --bit_width ${bit_width}
done

# # 1gpu
# for name in dev test; do
#   echo "Process "${name}" with "${codec_model_dir}
#   python scripts/data_prep.py codec_enc --bin_dir ../codec --wav_scp /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/utt_data/libri_${name}.scp --out_dir ${dumpdir}/libritts_${name}_codec --gpu_devices "0" --njob 1  --model_dir ${codec_model_dir} --model_name=$inference_model --bit_width ${bit_width}
# done


# # Dump data to ark
echo "Stage 5: Dump data to ark."
# subset_names=(train)
for name in "${subset_names[@]}"; do
  echo "Process stage 5: "${name}
  mkdir -p ${dump_dir}/${name}
  python scripts/data_prep.py data_pack ${tokens_dir}/${name}_text.scp ${dump_dir}/${name}/text ${dump_dir}/${name}_codec/codecs.txt ${dump_dir}/${name}/codec ${dump_dir}/${name}/codec_shape
done

echo "The command takes ${SECONDS} seconds to complete."
