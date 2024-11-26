#! /bin/bash
# bash /home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec/scripts/submit_ljspeech_data.sh

dataset_name=LJSpeech
dataset_root=/home/admin_data/user/dataset/LJSpeech
audio_format=.wav
# subsets=("train")
subsets=("all")

utt_dir=/home/admin_data/user/dataset/LibriTTS/scp/$dataset_name
feats_dir=/home/admin_data/user/model/uniaudio_admin/FunCodec-Dev/egs/LibriTTS/codec

mkdir -p ${feats_dir}/dump/$dataset_name

for subset in "${subsets[@]}"
do
    echo $subset
    mkdir -p ${feats_dir}/dump/$dataset_name/$subset
    cd $feats_dir
    find $dataset_root/wavs -iname *${audio_format} | awk -F '/' '{print $(NF),$0}' | sort > $feats_dir/dump/$dataset_name/$subset/wav.scp
    torchrun --nproc_per_node=12 --master_port=1234 $feats_dir/scripts/gen_wav_length.py --wav_scp $feats_dir/dump/$dataset_name/$subset/wav.scp --out_dir exp/${dataset_name}_states/$subset/wav_length
    cat exp/${dataset_name}_states/$subset/wav_length/wav_length.*.txt | shuf > exp/${dataset_name}_states/$subset/speech_shape
done
