source /cpfs01/shared/public/renjun.admin/anaconda3/bin/activate
source activate funasr
cd /home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/codec

utt_dir=/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/codec/utt_data/sq_cn
feats_dir=/home/admin_data/renjun.admin/dataset/project_data/funcodec-dev/codec
mkdir -p ${feats_dir}/dump/sq/train ${feats_dir}/dump/sq/dev ${feats_dir}/dump/sq/test

cp ${utt_dir}/train.utt ${feats_dir}/dump/sq/train/train.scp
cp ${utt_dir}/val.utt ${feats_dir}/dump/sq/dev/val.scp
cp ${utt_dir}/test.utt ${feats_dir}/dump/sq/test/test.scp

torchrun --nproc_per_node=4 --master_port=1234 scripts/gen_wav_length.py --wav_scp ${feats_dir}/dump/sq/train/train.scp --out_dir ${feats_dir}/exp/sq_states/train/wav_length
torchrun --nproc_per_node=1 --master_port=1234 scripts/gen_wav_length.py --wav_scp ${feats_dir}/dump/sq/dev/val.scp --out_dir ${feats_dir}/exp/sq_states/dev/wav_length
cat ${feats_dir}/exp/sq_states/train/wav_length/wav_length.*.txt | shuf > ${feats_dir}/exp/sq_states/train/speech_shape
cat ${feats_dir}/exp/sq_states/dev/wav_length/wav_length.*.txt | shuf > ${feats_dir}/exp/sq_states/dev/speech_shape
