#! /usr/bin/env bash
ref_wav_scp=$1
prompt_wav_scp=$2
hyp_wav_scp=$3
out_dir=$4

RootDir=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/tts/eval

# --local_model_dir /home/admin_data/user/checkpoint/se_pretrained/damo \
python ${RootDir}/eval_speaker_similarity.py \
    --model_id damo/speech_eres2net_sv_en_voxceleb_16k \
    --local_model_dir /home/admin_data/user/checkpoint/se_pretrained/damo \
    --ref_wavs ${ref_wav_scp} \
    --prompt_wavs ${prompt_wav_scp} \
    --hyp_wavs ${hyp_wav_scp} \
    --log_file ${out_dir}/spk_simi_scores.txt

tail -n1 ${out_dir}/spk_simi_scores.txt
