import librosa
import numpy as np
import os
import soundfile as sf

root_dir = '/mnt/workspace/renjun.admin/dataset/project_data/funcodec-dev/norm_test'
with open('/mnt/workspace/renjun.admin/jupyter_trash/test.scp', 'r') as f:
    contents = f.readlines()

def normalize_volumn(x):
    volume = np.sqrt(np.power(x, 2).mean())
    scale = 1e-8 + volume
    x = x / scale
    norm_volumn = np.sqrt(np.power(x, 2).mean())
    return x, scale, norm_volumn

result_contents = []
with open(f'{root_dir}/test.scp', 'w') as f:
    for line in contents:
        uid, wav_path = line.strip().split(' ')
        x = librosa.load(wav_path, sr=16000)[0]
        norm_x, _, vol = normalize_volumn(x)
        print(uid, vol)
        norm_x_clip = norm_x[640:16640] # 1s

        norm_x_path = os.path.join(root_dir, f'{uid}_v1.wav')
        norm_x_clip_path = os.path.join(root_dir, f'{uid}_v2.wav')

        sf.write(norm_x_path, norm_x, 16000, subtype='FLOAT')
        sf.write(norm_x_clip_path, norm_x_clip, 16000, subtype='FLOAT')

        y_new, sr_new = sf.read(norm_x_path, dtype='float32')

        f.write(f'{uid}_v1 {norm_x_path}\n')
        f.write(f'{uid}_v2 {norm_x_clip_path}\n')


    




