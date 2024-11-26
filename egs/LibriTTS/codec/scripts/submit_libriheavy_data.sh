#! /bin/bash
# bash /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec/scripts/submit_libriheavy_data.sh

dataset_name=LibriHeavy
dataset_root=/home/admin_data/user/dataset/libriheavy
audio_format=.flac
subsets=("all_clean")

utt_dir=/home/admin_data/user/dataset/LibriTTS/scp/$dataset_name
feats_dir=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec

mkdir -p ${feats_dir}/dump/$dataset_name

for subset in "${subsets[@]}"
do
    echo $subset
    mkdir -p ${feats_dir}/dump/$dataset_name/$subset
    cd $feats_dir
    wav_scp=/home/admin_data/user/dataset/libriheavy/scp/all_clean.scp
    torchrun --nproc_per_node=34 --master_port=1234 $feats_dir/scripts/gen_wav_length.py --wav_scp $wav_scp --out_dir exp/${dataset_name}_states/$subset/wav_length
    cat exp/${dataset_name}_states/$subset/wav_length/wav_length.*.txt | shuf > exp/${dataset_name}_states/$subset/speech_shape
done
