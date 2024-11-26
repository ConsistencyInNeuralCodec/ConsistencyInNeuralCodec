
ref_wav_scp=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/testsetA/test.scp
prompt_wav_scp=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/testsetA/test_ref.scp
hyp_wav_scp=/home/admin_data/renjun.admin/checkpoints/fun_uniaudio/exp/uniaudio_16k_n8_ds320_a100uniaudio_libritts_8gpu_b30/inference/60epoch.pth/testsetA/wav/wav.scp
out_dir=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/results/SpkSIM/v1

mkdir -p $out_dir

bash /home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/eval/eval_speaker_similarity.sh ${ref_wav_scp} ${prompt_wav_scp} ${hyp_wav_scp} ${out_dir}