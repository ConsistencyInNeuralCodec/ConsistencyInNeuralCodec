import os
import oss2
import time
import math
import json
from io import BytesIO
import librosa
import kaldiio
import numpy as np

OSS_CONFIG_PATH = os.environ.get("OSS_CONFIG_PATH", "/home/admin_data/renjun.admin/projects/yc_audio_llm/.oss_config.json")


class OssReader:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.bucket = {}
        with open(OSS_CONFIG_PATH) as fin:
            self.config = json.load(fin)

    def read(self, oss_path, read=True):
        assert oss_path.startswith('oss://'), f'{oss_path} is not a valid oss path.'
        bucket_name = oss_path[len('oss://') : ].split('/', 1)[0]
        if bucket_name not in self.bucket:
            bucket_meta = self.config[bucket_name]
            auth = oss2.Auth(bucket_meta['access_key_id'], bucket_meta['access_key_secret'])
            self.bucket[bucket_name] = oss2.Bucket(auth, bucket_name=bucket_name, endpoint=bucket_meta['endpoint'])
        key_path = oss_path[len('oss://') + len(bucket_name) + 1 : ]

        retry = 10
        for i in range(retry):
            try:
                if read:
                    data = self.bucket[bucket_name].get_object(key_path).read()
                else:
                    data = self.bucket[bucket_name].get_object(key_path)
                break
            except Exception as e:
                print('retry=', retry, e, oss_path)
                time.sleep(0.1)
                data = None

        # if data is None:
        #     print('OSS Connection Error')
        #     exit(1)
        return data
    
    # def read_kaldiwav(self, oss_path, read=True):
    #     # Load the wav from both the oss:// or local path. If the wav is corrupted or not exist, return 'None' value.
    #     # if return None, the oss path is break
    #     if 'oss' in oss_path:
    #         bytes_wav = self.read(oss_path, False)
    #         if bytes_wav is None:
    #             return None, None

    #         retval = kaldiio.load_mat(bytes_wav)
    #         # wav, sr = librosa.load(BytesIO(bytes_wav), sr=self.sample_rate)
    #     else:
    #         try:
    #             # for local path
    #             wav, sr = librosa.load(oss_path, sr=self.sample_rate)
    #         except Exception as e:
    #             print('local file read error', e, oss_path)
    #             return None, None
    #     return wav, sr
        
    def read_wav(self, oss_path, read=True, return_type=np.float32):
        # Load the wav from both the oss:// or local path. If the wav is corrupted or not exist, return 'None' value.
        # if return None, the oss path is break
        if oss_path.startswith('oss://'):
            # if 'oss' in oss_path:
            bytes_wav = self.read(oss_path, read)
            if bytes_wav is None:
                return None, None
            wav, sr = librosa.load(BytesIO(bytes_wav), sr=self.sample_rate)
        else:
            try:
                # for local path
                wav, sr = librosa.load(oss_path, sr=self.sample_rate)
            except Exception as e:
                print('local file read error', e, oss_path)
                return None, None
        if return_type is np.float32:
            # The librosa default type is np.float32 (normalized in [-1, 1])
            wav = wav
        elif return_type is np.int16:
            # Convert to the original Int dtype
            wav = (wav * 32768.0).astype(np.int16)
        else:
            assert 1==0, 'not supported yet'
        return wav, sr


# user case
# self.oss_reader = OssReader()