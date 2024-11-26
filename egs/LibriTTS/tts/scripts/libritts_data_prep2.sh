#! /usr/bin/env bash

# cd /mnt/workspace/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts
# bash scripts/libritts_data_prep2.sh
# pip install ttsfrd==0.0.4 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

cd /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts

. ./path.sh || exit 1;

# machines configuration
# gpu_devices="0"
# gpu_num=1
gpu_devices="0,1,2"
gpu_num=3

count=1

# general configuration
feats_dir="."
exp_dir="."

# dumpdir=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec/dump/encodec_phase_aug_consistent_quant_in_10.0_LibriTTS_8rvq
# dumpdir=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec/dump/encodec_LibriTTS_8rvq
dumpdir=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec/dump/encodec_consistent_quant_out_10.0_LibriTTS_8rvq
# dumpdir=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec/dump/encodec_phase_aug_consistent_quant_out_10.0_LibriTTS_8rvq

rtage=1
stop_stage=1
corpus_dir=LibriTTS
ttsfrd_rsc_zip_file=/home/admin_data/user/dataset/fun_tts_resource/tts/resource.zip
ttsfrd_config_file=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/conf/ali_tokenizer.json


# codec_model_dir=/home/admin_data/user/checkpoints/funcodec/exp/encoder_16k_n16_ds320_largev8_phase_aug_consistent_quant_in_10.0_encodec
# codec_model_dir=/home/admin_data/user/checkpoints/funcodec/exp/encoder_16k_n16_ds320_largev8_encodec
codec_model_dir=/home/admin_data/user/checkpoints/funcodec/exp/encoder_16k_n16_ds320_largev8_consistent_encodec
# codec_model_dir=/home/admin_data/user/checkpoints/funcodec/exp/encoder_16k_n16_ds320_largev8_phase_aug_consistent_encodec

# hubert_km_path=/nfs/shenxiao.hk/project/tts-deploy/hubert_base_ls960_L9_km500.bin

# training related
tag=""
train_set=train
valid_set=dev
train_config=$codec_model_dir/config.yaml
init_param=

# inference related
inference_model=4epoch.pth
inference_tag="inference"
batch_size=1
test_sets="testsetB"
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

# Extract feature (text token)
echo "Stage 2: extracting feature (text token)."
utt_data_root=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/utt_data
mkdir -p ${dumpdir}
for name in test train dev; do
  # for name in train dev test; do
  # for name in dev test; do
  echo "Process "${name}
  python scripts/data_prep.py trans_to_token $utt_data_root/libri_${name}_text.scp ${dumpdir}/libritts_${name}_text ${ttsfrd_rsc_zip_file} ${ttsfrd_config_file} --nj ${njob}
done


# Extract feature (codec)
echo "Stage 3: extracting feature (codec)."
mkdir -p ${dumpdir}
bit_width=4000 # 500/q -> 8 codec
# bit_width=8000 # 1000/q -> 16 codec
# for name in dev test; do
# for name in train; do
for name in test train dev; do
  echo "Process "${name}
  python scripts/data_prep.py codec_enc --bin_dir ../codec --wav_scp /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/utt_data/libri_${name}.scp --out_dir ${dumpdir}/libritts_${name}_codec --gpu_devices "0,1,2" --njob 3  --model_dir ${codec_model_dir} --model_name=$inference_model --bit_width ${bit_width}
done

# # 1gpu
# for name in dev test; do
#   echo "Process "${name}" with "${codec_model_dir}
#   python scripts/data_prep.py codec_enc --bin_dir ../codec --wav_scp /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/utt_data/libri_${name}.scp --out_dir ${dumpdir}/libritts_${name}_codec --gpu_devices "0" --njob 1  --model_dir ${codec_model_dir} --model_name=$inference_model --bit_width ${bit_width}
# done


# # Dump data to ark
echo "Stage 5: Dump data to ark."
# # for name in train dev test; do
# for name in dev test; do
# for name in train; do
for name in test train dev; do
  mkdir -p ${dumpdir}/${name}
  echo "Process "${name}
  python scripts/data_prep.py data_pack ${dumpdir}/libritts_${name}_text.scp ${dumpdir}/${name}/text ${dumpdir}/libritts_${name}_codec/codecs.txt ${dumpdir}/${name}/codec ${dumpdir}/${name}/codec_shape
done


