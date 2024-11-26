#!/usr/bin/env bash

. ./path.sh || exit 1;

stage=1
# wav_scp=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/utt_data/libri_dev.scp
wav_scp=/mnt/workspace/renjun.admin/dataset/project_data/funcodec-dev/norm_test/test.scp


njob=1   # nj per GPU or all nj for CPU
gpu_devices="0"
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
infer_cmd=utils/run.pl
# model_dir=/home/admin_data/renjun.admin/dataset/fun_tts_resource/ptts_encodec_16k_n32_600k
# model_dir=/home/admin_data/renjun.admin/checkpoints/funcodec/exp/encodec_16k_n32_600k_step_rawlibriMLS2k_cmw_1_raw
# model_dir=/home/admin_data/renjun.admin/checkpoints/funcodec/exp/encodec_16k_n32_600k_step2libriMLS2k_cmw_005
# model_dir=/home/admin_data/renjun.admin/checkpoints/funcodec/exp/encodec_16k_n32_600k_steplibriMLS2k_cmw_01
# model_dir=/home/admin_data/renjun.admin/checkpoints/funcodec/exp/encodec_try4libriMLS2k_try4

out_dir=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/utt_wav_codec_test/encodec_try4libriMLS2k_try4
model_dir=/home/admin_data/renjun.admin/checkpoints/funcodec/exp/encodec_try4libriMLS2k_try4
# out_dir=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/utt_wav_codec_test/freqcodec_mag2time_16k_n32_ds640freq2wav
# model_dir=/home/admin_data/renjun.admin/checkpoints/funcodec/exp/freqcodec_mag2time_16k_n32_ds640freq2wav
model_path=${model_dir}/30epoch.pth
config_path=${model_dir}/config.yaml
sample_frequency=16000
file_sampling_rate=16000
bit_width=4000   # 8k: 16, 4k: 8, 2k: 4, 1k:2, 0.5k:1
need_indices=true
need_sub_quants=false
use_scale=false
batch_size=16
num_workers=0
data_format=


. utils/parse_options.sh || exit 1;

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

mkdir -p exp


# Encoding Stage
if [ ${stage} -eq 1 ]; then
    echo "stage 1: encoding"

    _logdir="${out_dir}/logdir"
    # if [ -d ${out_dir} ]; then
    #     echo "WARNING: ${out_dir} is already exists."
    #     exit 0
    # fi
    mkdir -p "${_logdir}"
    key_file=${wav_scp}
    num_scp_file="$(<${key_file} wc -l)"
    _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
    split_scps=
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    if [ -z "$data_format" ]; then
      data_format="sound"
    fi
    utils/split_scp.pl "${key_file}" ${split_scps}
    python -m funcodec.bin.codec_inference \
        --batch_size ${batch_size} \
        --num_workers ${num_workers} \
        --ngpu "${_ngpu}" \
        --gpuid_list ${gpuid_list} \
        --data_path_and_name_and_type "${wav_scp},speech,${data_format}" \
        --key_file "${_logdir}"/keys.1.scp \
        --config_file ${config_path} \
        --model_file ${model_path} \
        --output_dir "${_logdir}"/output.1 \
        --sampling_rate $sample_frequency \
        --file_sampling_rate $file_sampling_rate \
        --bit_width ${bit_width} \
        --need_indices true \
        --need_sub_quants false \
        --use_scale false  \
        --run_mod "encode_wn"

    cat ${_logdir}/output.*/codecs.txt > ${out_dir}/codecs.txt
    echo "Codes are saved to ${_logdir}/output.*/codecs.txt and collected to ${out_dir}/codecs.txt."
fi

# Decoding Stage
if [ ${stage} -eq 2 ]; then
    echo "stage 2: Decoding"

    _logdir="${out_dir}/logdir"
    if [ -d ${out_dir} ]; then
        echo "WARNING: ${out_dir} is already exists."
        exit 0
    fi
    mkdir -p "${_logdir}"
    key_file=${wav_scp}
    num_scp_file="$(<${key_file} wc -l)"
    _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
    split_scps=
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    if [ -z "$data_format" ]; then
      data_format="codec_json"
    fi
    utils/split_scp.pl "${key_file}" ${split_scps}
    ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/inference.JOB.log \
        python -m funcodec.bin.codec_inference \
            --batch_size ${batch_size} \
            --num_workers ${num_workers} \
            --ngpu "${_ngpu}" \
            --gpuid_list ${gpuid_list} \
            --data_path_and_name_and_type "${wav_scp},speech,${data_format}" \
            --key_file "${_logdir}"/keys.JOB.scp \
            --config_file ${model_dir}/config.yaml \
            --model_file ${model_dir}/model.pth \
            --output_dir "${_logdir}"/output.JOB \
            --sampling_rate $sample_frequency \
            --file_sampling_rate $file_sampling_rate \
            --bit_width ${bit_width} \
            --need_indices false \
            --need_sub_quants false \
            --use_scale false  \
            --run_mod "decode"

    echo "Waveforms are reconstructed to ${_logdir}/output.*/*.wav."
fi

# Decoding Stage
if [ ${stage} -eq 3 ]; then
    echo "stage 3: Decoding codec embeddings"

    _logdir="${out_dir}/logdir"
    if [ -d ${out_dir} ]; then
        echo "WARNING: ${out_dir} is already exists."
        exit 0
    fi
    mkdir -p "${_logdir}"
    key_file=${wav_scp}
    num_scp_file="$(<${key_file} wc -l)"
    _nj=$([ $inference_nj -le $num_scp_file ] && echo "$inference_nj" || echo "$num_scp_file")
    split_scps=
    for n in $(seq "${_nj}"); do
        split_scps+=" ${_logdir}/keys.${n}.scp"
    done
    # shellcheck disable=SC2086
    if [ -z "$data_format" ]; then
      data_format="kaldi_ark"
    fi
    utils/split_scp.pl "${key_file}" ${split_scps}
    ${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/inference.JOB.log \
        python -m funcodec.bin.codec_inference \
            --batch_size ${batch_size} \
            --num_workers ${num_workers} \
            --ngpu "${_ngpu}" \
            --gpuid_list ${gpuid_list} \
            --data_path_and_name_and_type "${wav_scp},speech,${data_format}" \
            --key_file "${_logdir}"/keys.JOB.scp \
            --config_file ${model_dir}/config.yaml \
            --model_file ${model_dir}/model.pth \
            --output_dir "${_logdir}"/output.JOB \
            --sampling_rate $sample_frequency \
            --file_sampling_rate $file_sampling_rate \
            --bit_width ${bit_width} \
            --need_indices false \
            --need_sub_quants false \
            --use_scale false  \
            --run_mod "decode_emb"
fi
