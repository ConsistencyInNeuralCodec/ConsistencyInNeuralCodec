# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This script will download pretrained models from modelscope (https://www.modelscope.cn/models)
based on the given model id, and extract embeddings from input audio.
Please pre-install "modelscope".
Usage:
    1. extract the embedding from the wav file.
        `python infer_sv.py --model_id $model_id --wavs $wav_path `
    2. extract embeddings from two wav files and compute the similarity score.
        `python infer_sv.py --model_id $model_id --wavs $wav_path1 $wav_path2 `
    3. extract embeddings from the wav list.
        `python infer_sv.py --model_id $model_id --wavs $wav_list `
"""

import os, glob
import sys
import re
import pathlib
import csv
import pandas as pd
import numpy as np
import argparse
import torch
import torchaudio
from tqdm import tqdm
import modelscope
try:
    sys.path.append("/home/admin_data/user/model/3D-Speaker")
    from speakerlab.process.processor import FBank
except ImportError:
    sys.path.append('%s/../..'%os.path.dirname(__file__))
    from speakerlab.process.processor import FBank

from speakerlab.utils.builder import dynamic_import

parser = argparse.ArgumentParser(description='Extract speaker embeddings.')
parser.add_argument('--model_id', default=None, type=str, help='Model id in modelscope')

parser.add_argument('--ref_wavs', default=None, type=str, help='reference wavs')
parser.add_argument('--ref_tsv_path', default=None, type=str, help='reference wavs')
parser.add_argument('--output_prompt_scp_as_ref', default=None, type=str, help='reference wavs scp')

parser.add_argument('--prompt_wavs', default=None, type=str, help='prompt wavs')
parser.add_argument('--prompt_tsv_path', default=None, type=str, help='prompt wavs')

parser.add_argument('--hyp_wavs', default=None, type=str, help='generated wavs')
parser.add_argument('--hyp_dir', default=None, type=str, help='generated wavs')

parser.add_argument('--local_model_dir', default='pretrained', type=str, help='Local model dir')
parser.add_argument('--log_file', default=None, type=str, help='file to save log')
parser.add_argument('--devices', default="0", type=str)
parser.add_argument('--task_id', default=0, type=int)

CAMPPLUS_VOX = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
    },
}

CAMPPLUS_COMMON = {
    'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_VOX = {
    'obj': 'speakerlab.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_COMMON = {
    'obj': 'speakerlab.models.eres2net.ResNet_aug.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 192,
    },
}

ERes2Net_Base_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 32,
    },
}

ERes2Net_Large_3D_Speaker = {
    'obj': 'speakerlab.models.eres2net.ResNet.ERes2Net',
    'args': {
        'feat_dim': 80,
        'embedding_size': 512,
        'm_channels': 64,
    },
}

supports = {
    'damo/speech_campplus_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2',
        'model': CAMPPLUS_VOX,
        'model_pt': 'campplus_voxceleb.bin',
    },
    'damo/speech_campplus_sv_zh-cn_16k-common': {
        'revision': 'v1.0.0',
        'model': CAMPPLUS_COMMON,
        'model_pt': 'campplus_cn_common.bin',
    },
    'damo/speech_eres2net_sv_en_voxceleb_16k': {
        'revision': 'v1.0.2',
        'model': ERes2Net_VOX,
        'model_pt': 'pretrained_eres2net.ckpt',
    },
    'damo/speech_eres2net_sv_zh-cn_16k-common': {
        'revision': 'v1.0.4',
        'model': ERes2Net_COMMON,
        'model_pt': 'pretrained_eres2net_aug.ckpt',
    },
    'damo/speech_eres2net_base_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.1',
        'model': ERes2Net_Base_3D_Speaker,
        'model_pt': 'eres2net_base_model.ckpt',
    },
    'damo/speech_eres2net_large_sv_zh-cn_3dspeaker_16k': {
        'revision': 'v1.0.0',
        'model': ERes2Net_Large_3D_Speaker,
        'model_pt': 'eres2net_large_model.ckpt',
    },
}

def main():
    args = parser.parse_args()
    assert args.model_id in supports, "Model id not currently supported."
    conf = supports[args.model_id]

    pretrained_model = os.path.join(args.local_model_dir, args.model_id.split('/')[1], conf['model_pt'])
    pretrained_state = torch.load(pretrained_model, map_location='cpu')

    # load model
    model = conf['model']
    # embedding_model = dynamic_import(model['obj'])(**model['args'])
    # embedding_model.load_state_dict(pretrained_state)

    sv_pipline = modelscope.pipelines.pipeline(
        task='speaker-verification',
        model='damo/speech_eres2net_sv_en_voxceleb_16k'
    )
    embedding_model = sv_pipline.model

    embedding_model.eval()
    devices = args.devices.strip().split(',')
    task_id = int(args.task_id)
    device = f"cuda:{devices[task_id % len(devices)]}"
    # device=torch.device("cpu")
    embedding_model = embedding_model.to(device)

    def load_wav(wav_file, obj_fs=16000):
        wav, fs = torchaudio.load(wav_file)
        if fs != obj_fs:
            # print(f'[WARNING]: The sample rate of {wav_file} is not {obj_fs}, resample it.')
            # wav, fs = torchaudio.sox_effects.apply_effects_tensor(wav, fs, effects=[['rate', str(obj_fs)]])
            wav = torchaudio.functional.resample(wav, orig_freq=fs, new_freq=obj_fs)
        if wav.shape[0] > 1:
            wav = wav[0, :].unsqueeze(0)
        return wav

    feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    def compute_embedding(wav_file):
        # load wav
        wav = load_wav(wav_file)
        # compute feat
        # feat = feature_extractor(wav).unsqueeze(0).to(device)
        feat = wav
        # compute embedding
        with torch.no_grad():
            # print(f"666 feat = {feat.shape}")
            embedding = embedding_model(feat).detach().cpu().numpy()

        return embedding

    # extract embeddings
    print(f'[INFO]: Calculate similarities...')
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    out_file = open(args.log_file, "wt")
    similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    ref_wavs = []
    if args.ref_wavs is not None:
        for line in open(args.ref_wavs, "rt").readlines():
            uttid, wav_path = line.strip().split(maxsplit=1)
            ref_wavs.append((uttid, wav_path))
    elif args.ref_tsv_path is not None:
        df = pd.read_csv(args.ref_tsv_path, sep="\t", quoting=csv.QUOTE_ALL, encoding="utf-8", engine="python")
        for index, row in df.iterrows():
            ref_wavs.append((str(row["speech_id"]), row["audio_path"]))
    else:
        print(f"no ref_wavs")
    ref_wavs = {uttid: ref_wav_path for uttid, ref_wav_path in ref_wavs}

    
    prompt_wavs = {}
    if args.prompt_wavs is not None:
        for line in open(args.prompt_wavs, "rt").readlines():
            uttid, wav_path = line.strip().split(maxsplit=1)
            prompt_wavs[uttid] = wav_path
    elif args.prompt_tsv_path is not None:
        df = pd.read_csv(args.prompt_tsv_path, sep="\t", quoting=csv.QUOTE_ALL, encoding="utf-8", engine="python")
        for index, row in df.iterrows():
            prompt_wavs[str(row["speech_id"])] = row["audio_path"]
    else:
        print(f"no prompt_wavs")
    if args.output_prompt_scp_as_ref is not None:
        lines = [[speech_id, audio_path] for speech_id, audio_path in prompt_wavs.items()]
        lines = sorted(lines, key=lambda x: x[0])
        lines = [f"{speech_id} {audio_path}" for speech_id, audio_path in lines]
        with open(args.output_prompt_scp_as_ref, mode="w+") as f:
            f.write("\n".join(lines))

    # print(666, args.hyp_wavs)
    if not os.path.exists(args.hyp_wavs):
        lines = []
        hyp_wav_dir = os.path.split(args.hyp_wavs)[0]
        hyp_wav_list = [file for file in os.listdir(hyp_wav_dir) if file.endswith(".wav")]
        for hyp_wav in hyp_wav_list:
            # line = f"{os.path.splitext(hyp_wav)[0]} {os.path.join(hyp_wav_dir, hyp_wav)}"
            line = [os.path.splitext(hyp_wav)[0], os.path.join(hyp_wav_dir, hyp_wav)]
            lines.append(line)
        # lines = sorted(lines, key=lambda x: x.split(' ', 1)[0])
        lines = sorted(lines, key=lambda x: x[0])
        lines = [f"{speech_id} {audio_path}" for speech_id, audio_path in lines]
        with open(args.hyp_wavs, mode="w+") as f:
            f.write("\n".join(lines))

    hyp_wavs = {}
    for line in open(args.hyp_wavs, "rt").readlines():
        uttid, wav_path = line.strip().split(maxsplit=1)
        hyp_wavs[uttid] = wav_path
    ref_scores, hyp_scores = [], []

    uttid_list = []
    if len(ref_wavs) > 0:
        uttid_list = sorted(list(ref_wavs.keys()))
    elif len(prompt_wavs) > 0:
        uttid_list = sorted(list(prompt_wavs.keys()))

    for uttid in tqdm(uttid_list):
        # if uttid in prompt_wavs and uttid in hyp_wavs:
        hyp_emb = compute_embedding(hyp_wavs[uttid])

        if uttid in prompt_wavs:
            prompt_emb = compute_embedding(prompt_wavs[uttid])
            hyp_score = similarity(torch.from_numpy(prompt_emb), torch.from_numpy(hyp_emb)).item()
        else:
            hyp_score = 1.0
        if uttid in ref_wavs:
            ref_emb = compute_embedding(ref_wavs[uttid])
            ref_score = similarity(torch.from_numpy(prompt_emb), torch.from_numpy(ref_emb)).item()
        else:
            ref_score = 1.0

        ref_scores.append(ref_score)
        hyp_scores.append(hyp_score)
        out_file.writelines(f"{uttid} {ref_score*100.0:.2f} {hyp_score*100.0:.2f}\n")
        out_file.flush()

    avg_ref_score = np.array(ref_scores).mean()
    avg_hyp_score = np.array(hyp_scores).mean()

    print(f"num of ref wavs = {len(ref_wavs)}")
    print(f"num of prompt wavs = {len(prompt_wavs)}")
    print(f"num of hyp wavs = {len(hyp_wavs)}")
    out_file.writelines(f"avg {avg_hyp_score * 100.0:.2f}  / {avg_ref_score * 100.0:.2f} = {avg_hyp_score / avg_ref_score * 100.0:.2f}%\n")
    print(f"avg {avg_hyp_score * 100.0:.2f}  / {avg_ref_score * 100.0:.2f} = {avg_hyp_score / avg_ref_score * 100.0:.2f}%")
    out_file.writelines(f"avg prompt_ref_score = {avg_ref_score * 100.0:.2f}%\n")
    print(f"avg prompt_ref_score = {avg_ref_score * 100.0:.2f}%")
    out_file.writelines(f"avg prompt_hyp_score = {avg_hyp_score * 100.0:.2f}%\n")
    print(f"avg prompt_hyp_score = {avg_hyp_score * 100.0:.2f}%")
    out_file.close()


if __name__ == '__main__':
    main()


