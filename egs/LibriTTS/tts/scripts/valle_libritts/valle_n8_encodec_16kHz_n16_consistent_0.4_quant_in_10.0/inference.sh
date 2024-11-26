#! /usr/bin/env bash
# bash /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/scripts/valle_libritts/valle_n8_encodec_16kHz_n16_consistent_0.4_quant_in_10.0/inference.sh

### Initilize conda and codes
ENV=llm
source /home/admin_data/user/opt/anaconda/bin/activate
export PATH=/home/admin_data/user/opt/anaconda/envs/${ENV}/bin:$PATH
source activate ${ENV}
conda activate ${ENV}
which python

cd /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts
pwd

. ./path.sh || exit 1;

# machines configuration
# gpu_devices="0"
# gpu_num=1
gpu_devices="0,1,2"
gpu_num=3
# gpu_devices="0,1,2,3"
# gpu_num=4
count=1

# general configuration
exp_dir=/home/admin_data/user/checkpoints/funcodec_valle_libritts
testset_dir=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/testset/testsetA
corpus_dir=LibriTTS # not used
seed=0
# seed=2345

# training related
tag=valle_n8_encodec_16kHz_n16_consistent_0.4_quant_in_10.0
codec_model_dir=/home/admin_data/user/checkpoints/funcodec_consistency_encodec_LibriTTS+MLS_en/exp/encoder_16k_n16_ds320_largev8_consistent_0.4_quant_in_10.0_encodec

# _tag=""
_tag="_top_k=40"
# _tag="_top_k=40_top_2=0.35"
# _tag="_top_2=0.35"

# inference_model=40epoch.pth
inference_model=60epoch.pth
codec_inference_model=7epoch.pth

ttsfrd_rsc_zip_file=/home/admin_data/user/dataset/fun_tts_resource/tts/resource.zip
ttsfrd_config_file=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/conf/ali_tokenizer.json
train_set=train
valid_set=dev
train_config=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/conf/valle/valle_16k_n8_ds320.yaml
init_param=

# inference related
inference_tag="inference"
batch_size=1
test_sets="testsetA"
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
_dir="${asr_exp}/${inference_tag}/${inference_model}/${dset}${_tag}"
_logdir="${_dir}/logdir"
# if [ -d ${_dir} ]; then
#     echo "WARNING: ${_dir} is already exists."
#     exit 0
# fi
mkdir -p "${_logdir}"
cp -rf $testset_dir/test.txt ${_dir}/
echo "Putting your file in ${_dir}"
if [ ! -f ${_dir}/test.txt ]; then
    echo "WARNING: ${_dir}/test.txt is needed. each line: {utt_id  text  prompt_wav  prompt_text}"
    exit 0
fi

# pass if done
awk -F"\t" '{print $1"\t"$2}' ${_dir}/test.txt > ${_dir}/text_continual.txt
awk -F"\t" '{print $1"\t"$3}' ${_dir}/test.txt > ${_dir}/wav_prompt.scp
awk -F"\t" '{print $1"\t"$4}' ${_dir}/test.txt > ${_dir}/text_prompt.txt
python scripts/data_prep.py trans_to_token ${_dir}/text_continual.txt ${_dir}/text_continual ${ttsfrd_rsc_zip_file} ${ttsfrd_config_file} --nj ${njob}
python scripts/data_prep.py trans_to_token ${_dir}/text_prompt.txt ${_dir}/text_prompt ${ttsfrd_rsc_zip_file} ${ttsfrd_config_file} --nj ${njob}
python scripts/data_prep.py codec_enc --bin_dir ../codec --wav_scp ${_dir}/wav_prompt.scp --out_dir ${_dir}/codec --gpu_devices ${gpu_devices} --njob 1 --model_dir ${codec_model_dir} --model_name=$codec_inference_model --bit_width ${bit_width}
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

chmod -R 777 utils/split_scp.pl
chmod -R 777 utils/run.pl

# top_k=-100
# top_p=1.0
top_k=40
top_p=1.0

# pass if done
utils/split_scp.pl "${key_file}" ${split_scps}
${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/inference.JOB.log \
python -m funcodec.bin.tts_inference \
    --seed $seed \
    --batch_size ${batch_size} \
    --ngpu "${_ngpu}" \
    --gpuid_list ${gpuid_list}  \
    --data_path_and_name_and_type "${_data}/text_continual.scp,text,kaldi_ark" \
    --data_path_and_name_and_type "${_data}/text_prompt.scp,text_prefix,kaldi_ark" \
    --data_path_and_name_and_type "${_data}/codec_prompt.scp,codec_prefix,kaldi_ark" \
    --key_file  "${_logdir}"/keys.JOB.scp \
    --config_file "${asr_exp}"/config.yaml \
    --top_k $top_k \
    --top_p $top_p \
    --temperature 1.0 \
    --model_file "${asr_exp}"/"${inference_model}" \
    --output_dir "${_logdir}"/output.JOB

cat ${_logdir}/output.*/codecs.txt > ${_logdir}/codecs.txt
python scripts/data_prep.py codec_dec --bin_dir ../codec --codec_scp ${_logdir}/codecs.txt --out_dir ${_dir}/wav --gpu_devices ${gpu_devices} --njob 1 --model_dir ${codec_model_dir} --model_name=$codec_inference_model --bit_width ${bit_width}
