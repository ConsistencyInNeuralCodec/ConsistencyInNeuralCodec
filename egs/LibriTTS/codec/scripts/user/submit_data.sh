# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39
export OSS_CONFIG_PATH=/home/admin_data/user/model/ossutil/.oss_config.json

# source /cpfs01/shared/public/renjun.admin/anaconda3/bin/activate
# source activate funasr
cd /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec

# dataset_name=LibriTTS
# subset=("train", "val", "test")
# dataset_name=MLS_en_2k
# subset=("train")
dataset_name=LibriTTSAll
subset=("train", "val", "test")

# utt_dir=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec/utt_data/$dataset_name
utt_dir=/home/admin_data/user/dataset/LibriTTS/scp/speech_shuffle
feats_dir=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec
# mkdir -p ${feats_dir}/dump/$dataset_name

mkdir -p ${feats_dir}/dump/$dataset_name/train 
mkdir -p ${feats_dir}/dump/$dataset_name/dev
mkdir -p ${feats_dir}/dump/$dataset_name/test


cp ${utt_dir}/train.scp ${feats_dir}/dump/$dataset_name/train/train.scp
torchrun --nproc_per_node=16 --master_port=1234 scripts/gen_wav_length.py --wav_scp ${feats_dir}/dump/$dataset_name/train/train.scp --out_dir ${feats_dir}/exp/${dataset_name}_states/train/wav_length
cat ${feats_dir}/exp/${dataset_name}_states/train/wav_length/wav_length.*.txt | shuf > ${feats_dir}/exp/${dataset_name}_states/train/speech_shape

# cp ${utt_dir}/val.scp ${feats_dir}/dump/$dataset_name/dev/val.scp
# torchrun --nproc_per_node=40 --master_port=1234 scripts/gen_wav_length.py --wav_scp ${feats_dir}/dump/$dataset_name/dev/val.scp --out_dir ${feats_dir}/exp/${dataset_name}_states/dev/wav_length
# cat ${feats_dir}/exp/${dataset_name}_states/dev/wav_length/wav_length.*.txt | shuf > ${feats_dir}/exp/${dataset_name}_states/dev/speech_shape

# cp ${utt_dir}/test.scp ${feats_dir}/dump/$dataset_name/test/test.scp
# torchrun --nproc_per_node=40 --master_port=1234 scripts/gen_wav_length.py --wav_scp ${feats_dir}/dump/$dataset_name/test/test.scp --out_dir ${feats_dir}/exp/${dataset_name}_states/test/wav_length
# cat ${feats_dir}/exp/${dataset_name}_states/test/wav_length/wav_length.*.txt | shuf > ${feats_dir}/exp/${dataset_name}_states/test/speech_shape
