"""
Hubert + K-means as the HubertTokenizer, HF version, say goodbye to the huge fairseq repo.
author: KaiHu
"""
import torch
import librosa
import joblib
import os
import numpy as np

class ApplyKmeans(object):
    def __init__(self, device, km_path):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        self.C = self.C.to(device)
        self.Cnorm = self.Cnorm.to(device)

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)

class HubertTokenizer(object):
    def __init__(self, device, hubert_output_layer, km_path,
                 hubert_path="facebook/hubert-base-ls960",
                 half=False):
        super(HubertTokenizer, self).__init__()

        self.device = device
        self.half = half
        self.hubert_output_layer = hubert_output_layer
        self.load_hubert(km_path, hubert_path)

    def load_hubert(self, km_path, model_id):
        import transformers

        assert os.path.isfile(km_path), km_path

        print("start loading pretrained hubert, model_id: {}, km_path: {}".format(model_id, km_path))

        self.processor = transformers.Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        self.model = transformers.HubertModel.from_pretrained(model_id)

        self.model.to(self.device)
        if self.half:
            self.model = self.model.half()
        self.model.eval()

        self.kmeans = ApplyKmeans(self.device, km_path)

        print("end loading pretrained hubert, model_id: {}, km_path: {}".format(model_id, km_path))

    @torch.no_grad()
    def encode(self, wav_input:np.ndarray) -> np.ndarray:
        processed_data = self.processor(wav_input, sampling_rate=16000, return_tensors="pt") # Batch size 1
        input = processed_data.input_values
        if self.half:
            input = input.half()
        hubert_output = self.model(input.to(self.device), output_hidden_states=True).hidden_states
        embedding_output = hubert_output[self.hubert_output_layer][0] # [T, D]
        codebook_indices = self.kmeans(embedding_output).astype(np.int64)

        return codebook_indices

    @property
    def codebook_size(self):
        return self.kmeans.n_clusters

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, required=True)
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--model_id", type=str,
                        choices=["facebook/hubert-base-ls960",
                                "facebook/hubert-large-ll60k",
                                "TencentGameMate/chinese-hubert-large",
                                "TencentGameMate/chinese-hubert-base"],
                        default="facebook/hubert-base-ls960")
    parser.add_argument("--km_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--half", action="store_true", default=False)
    args = parser.parse_args()

    if args.model_path is not None:
        print("reset model_id with local model_path:{}".format(args.model_path))
        args.model_id = args.model_path

    wav = librosa.load(args.wav, sr=16000)[0]
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    hubert_tokenizer = HubertTokenizer(device, args.layer, args.km_path, args.model_id,
                                       half=args.half)
    hubert_code = hubert_tokenizer.encode(wav)
    print(hubert_code.shape, hubert_code)