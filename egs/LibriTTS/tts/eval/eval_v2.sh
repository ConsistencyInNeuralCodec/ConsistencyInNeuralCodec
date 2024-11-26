source /cpfs01/shared/public/renjun.admin/anaconda3/bin/activate
source activate funasr

ref_wav_scp=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/testsetC/test.scp
ref_wav_trans=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/testsetC/test_text.scp
prompt_wav_scp=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/testsetC/test_ref.scp
hyp_wav_scp=/home/admin_data/renjun.admin/checkpoints/fun_uniaudio/exp/uniaudio_16k_n8_ds320_a100uniaudio_libritts_8gpu_b30v2/inference/60epoch.pth/testsetC/wav/wav.scp
out_dir=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/results/run118_topk40/testC

language=en

# conduct speaker similarity eval
# cd /home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/eval
# pwd
# bash eval_speaker_similarity.sh ${ref_wav_scp} ${prompt_wav_scp} ${hyp_wav_scp} ${out_dir}

hypo_wav_file=${hyp_wav_scp}
hypo_out_trans=${out_dir}/gen_wav_trans_v2.txt
hypo_out_trans_norm=${out_dir}/gen_wav_trans_norm_v2.txt
gt_wav_file=${ref_wav_scp}
gt_out_trans=${out_dir}/gt_wav_trans_v2.txt
wer_res_file=${out_dir}/cer_v2.txt

# conduct ASR test with Whisper-V2
cd /home/admin_data/renjun.admin/projects/tools/WhisperASR
pwd
bash run_asr_v2.sh ${hypo_wav_file} ${hypo_out_trans} ${hypo_out_trans_norm} ${ref_wav_trans} ${gt_out_trans} ${wer_res_file} ${language}

