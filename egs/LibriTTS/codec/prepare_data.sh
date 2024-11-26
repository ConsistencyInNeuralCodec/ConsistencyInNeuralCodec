cd egs/LibriTTS/codec
# 0. make the directory for train, dev and test sets
mkdir -p dump/foo/train dump/foo/dev dump/foo/test

# 1a. if you already have the wav.scp file, just place them under the corresponding directories
mv train.scp dump/foo/train/; mv dev.scp dump/foo/dev/; mv test.scp dump/foo/test/;
# 1b. if you don't have the wav.scp file, you can prepare it as follows
find path/to/train_set/ -iname "*.wav" | awk -F '/' '{print $(NF),$0}' | sort > dump/foo/train/wav.scp
find path/to/dev_set/   -iname "*.wav" | awk -F '/' '{print $(NF),$0}' | sort > dump/foo/dev/wav.scp
find path/to/test_set/  -iname "*.wav" | awk -F '/' '{print $(NF),$0}' | sort > dump/foo/test/wav.scp

# 2. collate shape files
mkdir exp/foo_states/train exp/foo_states/dev
torchrun --nproc_per_node=4 --master_port=1234 scripts/gen_wav_length.py --wav_scp dump/foo/train/wav.scp --out_dir exp/foo_states/train/wav_length
cat exp/foo_states/train/wav_length/wav_length.*.txt | shuf > exp/foo_states/train/speech_shape
torchrun --nproc_per_node=4 --master_port=1234 scripts/gen_wav_length.py --wav_scp dump/foo/dev/wav.scp --out_dir exp/foo_states/dev/wav_length
cat exp/foo_states/dev/wav_length/wav_length.*.txt | shuf > exp/foo_states/dev/speech_shape

# 3. train the model with 2 GPUs (device 4 and 5) on the customized dataset (foo)
bash run.sh --gpu_devices 4,5 --gpu_num 2 --dumpdir dump/foo --state_dir foo_states



# ========= Above it the official tutorials =========
# source /cpfs01/shared/public/renjun.admin/anaconda3/bin/activate
# source activate funasr
cd /home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/codec
torchrun --nproc_per_node=4 --master_port=1234 scripts/gen_wav_length.py --wav_scp dump/foo/train/train.scp --out_dir exp/foo_states/train/wav_length
torchrun --nproc_per_node=1 --master_port=1234 scripts/gen_wav_length.py --wav_scp dump/foo/dev/val.scp --out_dir exp/foo_states/dev/wav_length
