#!/usr/bin/env bash
source /cpfs01/shared/public/renjun.admin/anaconda3/bin/activate
source activate funasr
conda activate funasr

which python

cd /home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts
pwd

. ./path.sh || exit 1;

# machines configuration
gpu_devices="0,1,2,3"
gpu_num=4
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
tag="uniaudio_libritts_8gpu_b30v2"
train_set=train
valid_set=dev
train_config=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/conf/uniaudio_16k_n8_ds320_a100.yaml
init_param=

# inference related
inference_model=60epoch.pth
inference_tag="inference"
batch_size=1
test_sets="testsetC"
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding

# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=5
docker_nj=32
infer_cmd=utils/run.pl
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

if ${gpu_inference}; then
    inference_nj=$[${ngpu}*${njob}]
    _ngpu=1
else
    inference_nj=$njob
    _ngpu=0
fi


# Testing Stage
dset=${test_sets}
echo "Processing for $dset"
asr_exp=${exp_dir}/exp/${model_dir}
_dir="${asr_exp}/${inference_tag}/${inference_model}/${dset}"
_logdir="${_dir}/logdir"
# if [ -d ${_dir} ]; then
#     echo "WARNING: ${_dir} is already exists."
#     exit 0
# fi
mkdir -p "${_logdir}"
cp -rf ${dset}/* ${_dir}/
echo "Putting your file in ${_dir}"
if [ ! -f ${_dir}/test.txt ]; then
    echo "WARNING: ${_dir}/test.txt is needed. each line: {utt_id  text  prompt_wav  prompt_text}"
    exit 0
fi
awk -F"\t" '{print $1"\t"$2}' ${_dir}/test.txt > ${_dir}/text_continual.txt
awk -F"\t" '{print $1"\t"$3}' ${_dir}/test.txt > ${_dir}/wav_prompt.scp
awk -F"\t" '{print $1"\t"$4}' ${_dir}/test.txt > ${_dir}/text_prompt.txt
python scripts/data_prep.py trans_to_token ${_dir}/text_continual.txt ${_dir}/text_continual ${ttsfrd_rsc_zip_file} ${ttsfrd_config_file} --nj ${njob}
python scripts/data_prep.py trans_to_token ${_dir}/text_prompt.txt ${_dir}/text_prompt ${ttsfrd_rsc_zip_file} ${ttsfrd_config_file} --nj ${njob}
python scripts/data_prep.py codec_enc --bin_dir ../codec --wav_scp ${_dir}/wav_prompt.scp --out_dir ${_dir}/codec --gpu_devices ${gpu_devices} --njob 1 --model_dir ${codec_model_dir} --bit_width ${bit_width}
python scripts/data_prep.py codec_txt_to_ark ${_dir}/codec/codecs.txt ${_dir}/codec_prompt
sort -k1 ${_dir}/codec_prompt.scp -o ${_dir}/codec_prompt.scp
sort -k1 ${_dir}/text_continual.scp -o ${_dir}/text_continual.scp
sort -k1 ${_dir}/text_prompt.scp -o ${_dir}/text_prompt.scp

_data="${_dir}"
key_file=${_data}/codec_prompt.scp
num_scp_file="$(<${key_file} wc -l)"
_nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
split_scps=
for n in $(seq "${_nj}"); do
    split_scps+=" ${_logdir}/keys.${n}.scp"
done
# shellcheck disable=SC2086
utils/split_scp.pl "${key_file}" ${split_scps}
${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/inference.JOB.log \
python -m funcodec.bin.tts_inference \
    --batch_size ${batch_size} \
    --ngpu "${_ngpu}" \
    --gpuid_list ${gpuid_list}  \
    --data_path_and_name_and_type "${_data}/text_continual.scp,text,kaldi_ark" \
    --data_path_and_name_and_type "${_data}/text_prompt.scp,text_prefix,kaldi_ark" \
    --data_path_and_name_and_type "${_data}/codec_prompt.scp,codec_prefix,kaldi_ark" \
    --key_file  "${_logdir}"/keys.JOB.scp \
    --config_file "${asr_exp}"/config.yaml \
    --top_k 40 \
    --top_p 1.0 \
    --temperature 1.0 \
    --model_file "${asr_exp}"/"${inference_model}" \
    --output_dir "${_logdir}"/output.JOB

cat ${_logdir}/output.*/codecs.txt > ${_logdir}/codecs.txt
python scripts/data_prep.py codec_dec --bin_dir ../codec --codec_scp ${_logdir}/codecs.txt --out_dir ${_dir}/wav --gpu_devices ${gpu_devices} --njob 1 --model_dir ${codec_model_dir}


