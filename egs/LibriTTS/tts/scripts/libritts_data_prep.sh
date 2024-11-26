#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
gpu_devices="0"
gpu_num=1
count=1

# general configuration
feats_dir="."
exp_dir="."
dumpdir=dump/LibriTTS
rtage=1
stop_stage=1
corpus_dir=LibriTTS
ttsfrd_rsc_zip_file=/home/admin_data/renjun.admin/dataset/fun_tts_resource/tts/resource.zip
ttsfrd_config_file=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/conf/ali_tokenizer.json
codec_model_dir=/home/admin_data/renjun.admin/dataset/fun_tts_resource/ptts_encodec_16k_n32_600k
# hubert_km_path=/nfs/shenxiao.hk/project/tts-deploy/hubert_base_ls960_L9_km500.bin

# training related
tag=""
train_set=train
valid_set=dev
train_config=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/conf/uniaudio_16k_n8_ds320_v100.yaml
init_param=

# inference related
inference_model=24epoch.pth
inference_tag="inference"
batch_size=1
test_sets="testsetB"
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding

# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=1
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
# echo "Stage 2: extracting feature (text token)."
# mkdir -p ${dumpdir}
# for name in train dev test; do
# echo "Process"${name}
# python scripts/data_prep.py trans_to_token /home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/utt_data/libri_${name}_text.scp ${dumpdir}/libritts_${name}_text ${ttsfrd_rsc_zip_file} ${ttsfrd_config_file} --nj ${njob}
# done


# Extract feature (codec)
# echo "Stage 3: extracting feature (codec)."
# mkdir -p ${dumpdir}
# bit_width=4000 # 500/q -> 8 codec
# for name in train dev test; do
# python scripts/data_prep.py codec_enc --bin_dir ../codec --wav_scp /home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/utt_data/libri_${name}.scp --out_dir ${dumpdir}/libritts_${name}_codec --gpu_devices "0,1,2,3,4,5,6,7" --njob 8  --model_dir ${codec_model_dir} --bit_width ${bit_width}
# done


# # Dump data to ark
echo "Stage 5: Dump data to ark."
for name in train dev test; do
mkdir -p ${dumpdir}/${name}
python scripts/data_prep.py data_pack ${dumpdir}/libritts_${name}_text.scp ${dumpdir}/${name}/text ${dumpdir}/libritts_${name}_codec/codecs.txt ${dumpdir}/${name}/codec ${dumpdir}/${name}/codec_shape
done


