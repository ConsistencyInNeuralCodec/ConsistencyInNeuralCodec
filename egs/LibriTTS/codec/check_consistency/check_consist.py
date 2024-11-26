import os
import json
import numpy as np
# encodec_16k_n32_600k_step_rawlibriMLS2k_cmw_1_raw
# encodec_16k_n32_600k_step2libriMLS2k_cmw_005
# encodec_16k_n32_600k_steplibriMLS2k_cmw_01
# encodec_try2libriMLS2k_try2
# encodec_try1libriMLS2k_try1
file_path = '/home/admin_data/renjun.admin/projects/FunCodec-Dev/egs/LibriTTS/tts/utt_wav_codec_test/encodec_try4libriMLS2k_try4/codecs.txt'
down_ratio = 320
s_idx = int(640//down_ratio)
e_idx = int(16640//down_ratio)

pair_dict = {}
with open(file_path, 'r') as f:
    contents = f.readlines()
    for line in contents:
        uid, codec_arr = line.strip().split(' ', maxsplit=1)
        prefix, v = '_'.join(uid.split('_')[:-1]), uid.split('_')[-1]
        if prefix not in pair_dict:
            pair_dict[prefix] = {}
        pair_dict[prefix][v] = json.loads(codec_arr)

level_logger = [[] for i in range(32)]
for k, pairs in pair_dict.items():
    if 'v1' in pairs and 'v2' in pairs:
        v1 = pairs['v1']
        v2 = pairs['v2']
        v1 = np.array(v1)[0][:, s_idx:e_idx]
        v2 = np.array(v2)[0]
        for i in range(v1.shape[0]):
            level_logger[i].append(sum(v1[i] == v2[i]) / len(v1[i]))


for i, level_l in enumerate(level_logger):
    if len(level_l) > 0:
        print(f'{i} layer: {np.mean(level_l)}')

