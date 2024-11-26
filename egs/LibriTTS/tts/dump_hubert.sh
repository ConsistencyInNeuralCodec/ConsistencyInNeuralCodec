#!/usr/bin/env bash

. ./path.sh || exit 1;

stage=1
wav_scp=
out_dir=
km_path=

njob=1   # nj per GPU or all nj for CPU
gpu_devices="0"
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
infer_cmd=utils/run.pl

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
utils/split_scp.pl "${key_file}" ${split_scps}
${infer_cmd} --gpu "${_ngpu}" --max-jobs-run "${_nj}" JOB=1:"${_nj}" "${_logdir}"/inference.JOB.log \
    python scripts/dump_audio_nn_feat.py \
        --meta "${_logdir}"/keys.JOB.scp \
        --gpu_ids ${gpuid_list} \
        --job_id JOB hubert \
        --num_workers 4 \
        --output_file "${_logdir}"/output.JOB \
        --km_path ${km_path}
