#!/usr/bin/env bash

. ./path.sh || exit 1;

# machines configuration
gpu_devices="6,7"
gpu_num=2
count=1

# general configuration
feats_dir="."
exp_dir="."
dumpdir=dump/LibriTTS
rtage=1
stop_stage=1
corpus_dir=LibriTTS
ttsfrd_rsc_zip_file=/nfs/shenxiao.hk/project/tts-deploy/mit_opentts/resource_8.5.0/resource.zip
ttsfrd_config_file=conf/ali_tokenizer.json
codec_model_dir=/nfs/shenxiao.hk/TtsModelCkptHub/codec/ptts_encodec_16k_n32_600k
hubert_km_path=/nfs/shenxiao.hk/project/tts-deploy/hubert_base_ls960_L9_km500.bin

# training related
tag=""
train_set=train
valid_set=dev
train_config=conf/uniaudio_16k_n8_ds320_v100.yaml
init_param=

# inference related
inference_model=24epoch.pth
inference_tag="inference"
batch_size=1
test_sets="testsetB"
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

# Data downloading
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Stage 0: data downloading"

  if [ ! -d ${corpus_dir} ]; then
    mkdir -p ${corpus_dir}
  fi

  echo "download training set to ${corpus_dir}"
  wget --no-check-certificate https://www.openslr.org/resources/60/train-clean-100.tar.gz -P ${corpus_dir}/
  wget --no-check-certificate https://www.openslr.org/resources/60/train-clean-360.tar.gz -P ${corpus_dir}/
  wget --no-check-certificate https://www.openslr.org/resources/60/train-other-500.tar.gz -P ${corpus_dir}/

  echo "download dev set to ${corpus_dir}"
  wget --no-check-certificate https://www.openslr.org/resources/60/dev-clean.tar.gz -P ${corpus_dir}/
  wget --no-check-certificate https://www.openslr.org/resources/60/dev-other.tar.gz -P ${corpus_dir}/

  echo "download test set to ${corpus_dir}"
  wget --no-check-certificate https://www.openslr.org/resources/60/test-clean.tar.gz -P ${corpus_dir}/
  wget --no-check-certificate https://www.openslr.org/resources/60/test-other.tar.gz -P ${corpus_dir}/

  cd ${corpus_dir}/
  for fn in train-clean-100.tar.gz train-clean-360.tar.gz train-other-500.tar.gz; do
      tar -zxf $fn
  done
  for fn in dev-clean.tar.gz dev-other.tar.gz; do
      tar -zxf $fn
  done
  for fn in test-clean.tar.gz test-other.tar.gz; do
      tar -zxf $fn
  done

  # remove the duplicated LibriTTS directory
  mv ${corpus_dir}/LibriTTS/* ${corpus_dir}/
  rm -rf ${corpus_dir}/LibriTTS
fi

# Data collecting
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: collecting data sets."
  mkdir -p ${dumpdir}
  python scripts/data_prep.py wav_trans_collect ${corpus_dir} ${dumpdir} --nj ${njob} --need_sampling --sr 16000
fi

# Extract feature (text token)
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: extracting feature (text token)."
  mkdir -p ${dumpdir}
  for name in train dev test; do
    python scripts/data_prep.py trans_to_token ${dumpdir}/libritts_${name}.trans ${dumpdir}/libritts_${name}_text ${ttsfrd_rsc_zip_file} ${ttsfrd_config_file} --nj ${njob}
  done
fi

# Extract feature (codec)
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: extracting feature (codec)."
  mkdir -p ${dumpdir}
  for name in train dev test; do
    python scripts/data_prep.py codec_enc --bin_dir ../codec --wav_scp ${dumpdir}/libritts_${name}_wav.scp --out_dir ${dumpdir}/libritts_${name}_codec --gpu_devices ${gpu_devices} --njob 1  --model_dir ${codec_model_dir} --bit_width ${bit_width}
  done
fi

# Option: Extract feature (hubert)
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: extracting feature (hubert)."
  mkdir -p ${dumpdir}

  for name in train dev test; do
    sh dump_hubert.sh --wav_scp ${dumpdir}/libritts_${name}_wav.scp --out_dir ${dumpdir}/libritts_${name}_hubert --km_path ${hubert_km_path} --gpu_devices "0" --njob 1
    cat ${dumpdir}/libritts_${name}_hubert/logdir/output*.scp > ${dumpdir}/libritts_${name}_hubert/hubert.scp
  done
fi

# Dump data to ark
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Stage 5: Dump data to ark."

  for name in train dev test; do
    mkdir -p ${dumpdir}/${name}
    python scripts/data_prep.py data_pack ${dumpdir}/libritts_${name}_text.scp ${dumpdir}/${name}/text ${dumpdir}/libritts_${name}_codec/codecs.txt ${dumpdir}/${name}/codec ${dumpdir}/${name}/codec_shape
  done
fi

# Training Stage
world_size=$gpu_num  # run on one machine
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
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
fi

# Testing Stage
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    echo "stage 7: Inference"
    for dset in ${test_sets}; do
        echo "Processing for $dset"
        asr_exp=${exp_dir}/exp/${model_dir}
        _dir="${asr_exp}/${inference_tag}/${inference_model}/${dset}"
        _logdir="${_dir}/logdir"
        if [ -d ${_dir} ]; then
            echo "WARNING: ${_dir} is already exists."
            exit 0
        fi
        mkdir -p "${_logdir}"
        cp -rf ${dset}/* ${_dir}/
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
                --model_file "${asr_exp}"/"${inference_model}" \
                --output_dir "${_logdir}"/output.JOB

        cat ${_logdir}/output.*/codecs.txt > ${_logdir}/codecs.txt
        python scripts/data_prep.py codec_dec --bin_dir ../codec --codec_scp ${_logdir}/codecs.txt --out_dir ${_dir}/wav --gpu_devices ${gpu_devices} --njob 1 --model_dir ${codec_model_dir}
    done
fi
