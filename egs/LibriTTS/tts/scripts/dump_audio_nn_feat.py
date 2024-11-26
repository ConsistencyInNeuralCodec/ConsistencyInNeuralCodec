import argparse
import os
import numpy as np
from tqdm import tqdm
import librosa
from kaldiio import WriteHelper
from io import BytesIO

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler

import torchaudio.compliance.kaldi as Kaldi
from funcodec.modules.hubert_tokenizer import HubertTokenizer
from funcodec.modules.se.D_TDNN import DTDNN

MAXWAVDUR=30 # max 30 seconds

def load_checkpoint(filepath):
    with open(filepath, "rb") as f:
        buffer = BytesIO(f.read())

    checkpoint_dict = torch.load(buffer, map_location=torch.device('cpu'))
    return checkpoint_dict

class WavDataset(Dataset):
    def __init__(self, wav_scp_file):
        super(WavDataset, self).__init__()

        self.data = {}
        with open(wav_scp_file, "r") as fr:
            for line in fr:
                if "\t" in line:
                    parts = line.strip().split("\t")
                elif " " in line:
                    parts = line.strip().split(" ")
                else:
                    raise NotImplementedError
                self.data[parts[0]] = parts[1]
        self.utt_lst = list(self.data.keys())

    def __len__(self):
        return len(self.utt_lst)

    def __getitem__(self, index):
        utt_id = self.utt_lst[index]
        wav = librosa.load(self.data[utt_id], sr=16000)[0]
        return utt_id, torch.from_numpy(wav).float()

    def calc_fbank(self, wav_data):
        assert wav_data.shape[0] == 1
        fbank_feat = Kaldi.fbank(wav_data, num_mel_bins=80)
        fbank_feat = fbank_feat - fbank_feat.mean(dim=0, keepdim=True)

        return fbank_feat

    def collate_fn(self, batch):
        utt_lst = [x[0] for x in batch]
        wav_lst = [x[1] for x in batch]

        return utt_lst, wav_lst

def extract_hubert_code(a):
    dataset = WavDataset(a.meta)

    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, num_workers=a.num_workers, sampler=sampler, collate_fn=dataset.collate_fn,
                             batch_size=1, drop_last=False)

    tok = HubertTokenizer(a.device, hubert_output_layer=a.output_layer, km_path=a.km_path, hubert_path=a.model_id,
                          half=a.half)

    with WriteHelper('ark,scp:{}.ark,{}.scp'.format(os.path.abspath(a.output_file), os.path.abspath(a.output_file)), write_function="pickle") as writer, \
            open('{}.shape'.format(a.output_file), "w") as fw:

        for batch in data_loader:
            wav = batch[1][0]
            utt_id = batch[0][0]
            wav = wav.numpy()
            if len(wav) > (16000 * MAXWAVDUR):
                continue

            codes = tok.encode(wav).reshape((-1, 1))
            writer(utt_id, codes.astype(np.int16))
            fw.write("{} {},{}\n".format(utt_id, codes.shape[0], codes.shape[1]))

def extract_spk_vector(a):
    dataset = WavDataset(a.meta)

    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, num_workers=a.num_workers, sampler=sampler, collate_fn=dataset.collate_fn,
                             batch_size=1, drop_last=False)

    se_net = DTDNN()
    se_net.load_state_dict(load_checkpoint(a.ckpt))
    se_net = se_net.to(a.device)
    se_net.eval()

    with WriteHelper('ark,scp:{}.ark,{}.scp'.format(a.output_file, a.output_file)) as writer:
        for batch in tqdm(data_loader):
            utt_id = batch[0][0]
            wav = batch[1][0]
            fbank = dataset.calc_fbank(wav.unsqueeze(0)).unsqueeze(0)

            with torch.no_grad():
                se_vector = se_net(fbank.to(a.device))
                se_vector = se_vector.squeeze().cpu().detach().numpy()

            writer(utt_id, se_vector)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str)
    parser.add_argument('--gpu_ids', type=str, default="")
    parser.add_argument("--job_id", type=int, default=1)
    subparsers = parser.add_subparsers(dest="command", help="commands")

    hubert_parser = subparsers.add_parser("hubert")
    hubert_parser.add_argument('--output_layer', type=int, default=9)
    hubert_parser.add_argument('--km_path', type=str)
    hubert_parser.add_argument('--model_id', type=str, default="facebook/hubert-base-ls960")
    hubert_parser.add_argument('--half', action="store_true", default=False)
    hubert_parser.add_argument('--num_workers', type=int, default=4)
    hubert_parser.add_argument('--output_file', type=str)

    sv_parser = subparsers.add_parser("sv")
    sv_parser.add_argument('--ckpt', type=str, required=True)
    sv_parser.add_argument('--num_workers', type=int, default=4)
    sv_parser.add_argument('--output_file', type=str)

    args = parser.parse_args()

    if len(args.gpu_ids):
        args.gpu_ids = args.gpu_ids.split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids[(args.job_id - 1) % len(args.gpu_ids)]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(vars(args))
    args.device = device

    if args.command == "sv":
        extract_spk_vector(args)
    elif args.command == "hubert":
        extract_hubert_code(args)


if __name__ == "__main__":
    main()
