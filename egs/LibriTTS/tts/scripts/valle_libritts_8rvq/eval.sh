#! /usr/bin/env bash
export HF_ENDPOINT=https://hf-mirror.com

### Initilize conda and codes
ENV=llm
source /home/admin_data/user/opt/anaconda/bin/activate
export PATH=/home/admin_data/user/opt/anaconda/envs/${ENV}/bin:$PATH
source activate ${ENV}
conda activate ${ENV}
which python

# bash /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/scripts/valle_libritts_8rvq/eval.sh

# model_name=valle_16k_n8_ds320valle_n8_encodec_16kHz_n8
# model_name=valle_16k_n8_ds320valle_n8_encodec_16kHz_n8_consistent_0.2_quant_in_10.0
# model_name=valle_16k_n8_ds320valle_n8_encodec_16kHz_n8_phase_aug_consistent_0.2_quant_in_10.0
# model_name=valle_16k_n8_ds320valle_n8_encodec_16kHz_n8_consistent_0.4_quant_in_10.0
model_name=valle_16k_n8_ds320valle_n8_encodec_16kHz_n8_phase_aug_consistent_0.4_quant_in_10.0

model_inference_dir=inference/1epoch.pth/testsetA
# model_inference_dir=inference/6epoch.pth/testsetA
# model_inference_dir=inference/30epoch.pth/testsetA
# model_inference_dir=inference/40epoch.pth/testsetA
# model_inference_dir=inference/60epoch.pth/testsetA

# exp_dir=/home/admin_data/user/checkpoints/funcodec_valle_libritts_8rvq
exp_dir=/home/admin_data/user/checkpoints/funcodec_valle_libritts+mls_en+libriheavy_8rvq
model_dir=${exp_dir}/exp/${model_name}

# _tag=""
_tag="_top_k=40"
# _tag="_top_k=40_top_2=0.35"
# _tag="_top_2=0.35"

model_inference_dir="${model_inference_dir}${_tag}"

ref_scp=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/testset/testsetA/test.scp
hyp_wav_scp=$model_dir/$model_inference_dir/wav/wav.scp
out_dir=$model_dir/$model_inference_dir/eval
mkdir -p $out_dir

testset_dir=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/testset/testsetA
ref_wav_scp=$testset_dir/test.scp
prompt_wav_scp=$testset_dir/test_ref.scp

# conduct speaker similarity eval
cd /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/eval
pwd
bash eval_speaker_similarity.sh ${ref_wav_scp} ${prompt_wav_scp} ${hyp_wav_scp} ${out_dir}

hypo_wav_file=${hyp_wav_scp}
hypo_out_trans=${out_dir}/gen_wav_trans.txt
gt_wav_file=${ref_wav_scp}
gt_out_trans=${out_dir}/gt_wav_trans.txt
wer_res_file=${out_dir}/cer.txt

# conduct ASR test with Whisper-V2
cd /home/admin_data/user/model/whisperASR
pwd
bash run_asr.sh ${hypo_wav_file} ${hypo_out_trans} ${gt_wav_file} ${gt_out_trans} ${wer_res_file}

# ut_mos
cd /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/examples/assess
sampling_rate=16000
metrics=ut_mos
test_times=20
master_port=2234
n_process=16
speech_feature_extractor_path=/home/admin_data/user/checkpoint/wav2vec2/wav2vec2-base-960h
ut_mos_model_dir=/home/admin_data/user/checkpoint/tarepan-SpeechMOS
output_dir=$out_dir/ut_mos
mkdir -p $output_dir/tmp1
output_tsv_path=$output_dir/tmp1/all.tsv
torchrun --nproc_per_node=$n_process --master_port=$master_port main.py \
    --src_speech_scp $ref_scp --tgt_speech_scp $hyp_wav_scp \
    --output_dir $output_dir --output_tsv_path $output_tsv_path \
    --speech_feature_extractor_path $speech_feature_extractor_path --ut_mos_model_dir $ut_mos_model_dir \
    --trim_speech True \
    --sampling_rate $sampling_rate --metrics $metrics
exit 1

echo $model_name