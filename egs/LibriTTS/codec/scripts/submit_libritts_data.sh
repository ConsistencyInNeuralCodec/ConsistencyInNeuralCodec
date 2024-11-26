# source /cpfs01/shared/public/renjun.admin/anaconda3/bin/activate
# source activate funasr

cd /home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/codec
torchrun --nproc_per_node=8 --master_port=1234 scripts/gen_wav_length.py --wav_scp dump/foo/train/libritts_train.scp --out_dir exp/foo_states/train/libritts_wav_length
cat exp/foo_states/train/libritts_wav_length/wav_length.*.txt | shuf > exp/foo_states/train/libritts_speech_shape

# Val (100条 libritts)
# torchrun --nproc_per_node=4 --master_port=1234 scripts/gen_wav_length.py --wav_scp dump/foo/dev/libritts_val.scp --out_dir exp/foo_states/dev/libritts_wav_length
# cat exp/foo_states/dev/libritts_wav_length/wav_length.*.txt | shuf > exp/foo_states/dev/libritts_speech_shape
# bash /home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/codec/scripts/submit_libritts_data.sh