source /cpfs01/shared/public/renjun.admin/anaconda3/bin/activate
source activate funasr
cd /home/admin_data/renjun.admin/projects/tools/WhisperASR

hypo_wav_file=/home/admin_data/renjun.admin/checkpoints/fun_uniaudio/exp/uniaudio_16k_n8_ds320_a100uniaudio_libritts_8gpu_b30/inference/60epoch.pth/testsetA/wav
hypo_out_trans=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/results/SpkASR/gen_wav.txt
gt_wav_file=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/testsetA/test.scp
gt_out_trans=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/results/SpkASR/gt_wav.txt
wer_res_file=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/results/SpkASR/cer.txt

bash run_asr.sh ${hypo_wav_file} ${hypo_out_trans} ${gt_wav_file} ${gt_out_trans} ${wer_res_file}
