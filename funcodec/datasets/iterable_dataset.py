"""Iterable dataset module."""
import copy
import json, jsonlines
from io import StringIO
from pathlib import Path
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Iterator
from typing import Tuple
from typing import Union

import kaldiio
import numpy as np
import torch
import torchaudio
from torch.utils.data.dataset import IterableDataset
from typeguard import check_argument_types
import os.path

from funcodec.datasets.dataset import ESPnetDataset
from funcodec.datasets.oss_reader import OssReader


SUPPORT_AUDIO_TYPE_SETS = ['flac', 'mp3', 'ogg', 'opus', 'wav', 'pcm']


def load_kaldi(input):
    retval = kaldiio.load_mat(input)
    if isinstance(retval, tuple):
        assert len(retval) == 2, len(retval)
        if isinstance(retval[0], int) and isinstance(retval[1], np.ndarray):
            # sound scp case
            rate, array = retval
        elif isinstance(retval[1], int) and isinstance(retval[0], np.ndarray):
            # Extended ark format case
            array, rate = retval
        else:
            raise RuntimeError(f"Unexpected type: {type(retval[0])}, {type(retval[1])}")

        # Multichannel wave fie
        # array: (NSample, Channel) or (Nsample)
        if array.dtype == np.int16:
            array = (array / (2 ** 15)).astype(np.float32)
        elif array.dtype == np.int32:
            array = (array / (2 ** 31)).astype(np.float32)

    else:
        # Normal ark case
        assert isinstance(retval, np.ndarray), type(retval)
        array = retval
    return array


def load_codec_json(json_str):
    array = np.array(json.loads(json_str))
    if len(array.shape) == 3:
        array = array[0]
    return array.T


def load_bytes(input):
    middle_data = np.frombuffer(input, dtype=np.int16)
    middle_data = np.asarray(middle_data)
    if middle_data.dtype.kind not in 'iu':
        raise TypeError("'middle_data' must be an array of integers")
    dtype = np.dtype('float32')
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(middle_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    array = np.frombuffer((middle_data.astype(dtype) - offset) / abs_max, dtype=np.float32)
    return array


def load_pcm(input):
    with open(input,"rb") as f:
        bytes = f.read()
    return load_bytes(bytes)


def load_mel2ph(mel2ph_path: str):
    if mel2ph_path.endswith(".json"):
        with open(mel2ph_path, mode="r+") as f:
            mel2ph = json.load(f)
    elif mel2ph_path.endswith(".jsonl"):
        with open(mel2ph_path, mode="r+") as f:
            items = [line for line in jsonlines.Reader(f)]
        mel2ph = {}
        for item in items:
            for key, value in item.items():
                mel2ph[key] = value
    return mel2ph


sound_oss_reader = OssReader()
DATA_TYPES = {
    # "sound": lambda x: torchaudio.load(x)[0][0].numpy(),
    "sound": lambda x: sound_oss_reader.read_wav(x)[0],
    "pcm": load_pcm,
    "kaldi_ark": load_kaldi,
    "bytes": load_bytes,
    "waveform": lambda x: x,
    "npy": np.load,
    "text_int": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.long, delimiter=" "
    ),
    "csv_int": lambda x: np.loadtxt(StringIO(x), ndmin=1, dtype=np.long, delimiter=","),
    "text_float": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.float32, delimiter=" "
    ),
    "csv_float": lambda x: np.loadtxt(
        StringIO(x), ndmin=1, dtype=np.float32, delimiter=","
    ),
    "text": lambda x: x,
    "codec_json": load_codec_json,
}


class IterableESPnetDataset(IterableDataset):
    """Pytorch Dataset class for ESPNet.

    Examples:
        >>> dataset = IterableESPnetDataset([('wav.scp', 'input', 'sound'),
        ...                                  ('token_int', 'output', 'text_int')],
        ...                                )
        >>> for uid, data in dataset:
        ...     data
        {'input': per_utt_array, 'output': per_utt_array}
    """

    def __init__(
            self,
            path_name_type_list: Collection[Tuple[any, str, str]],
            preprocess: Callable[
                [str, Dict[str, np.ndarray]], Dict[str, np.ndarray]
            ] = None,
            float_dtype: str = "float32",
            fs: dict = None,
            int_dtype: str = "long",
            key_file: str = None,
            mel2ph_file: str = None,
            speaker_id_dir: str = None,
    ):
        assert check_argument_types()
        if len(path_name_type_list) == 0:
            raise ValueError(
                '1 or more elements are required for "path_name_type_list"'
            )

        path_name_type_list = copy.deepcopy(path_name_type_list)
        self.preprocess = preprocess

        self.float_dtype = float_dtype
        self.int_dtype = int_dtype
        self.key_file = key_file
        self.mel2ph_file = None
        if mel2ph_file is not None:
            self.mel2ph_file = load_mel2ph(mel2ph_file)
        self.speaker_id_dir = speaker_id_dir
        self.fs = fs

        self.debug_info = {}
        non_iterable_list = []
        self.path_name_type_list = []

        if not isinstance(path_name_type_list[0], Tuple):
            path = path_name_type_list[0]
            name = path_name_type_list[1]
            _type = path_name_type_list[2]
            self.debug_info[name] = path, _type
            if _type not in DATA_TYPES:
                non_iterable_list.append((path, name, _type))
            else:
                self.path_name_type_list.append((path, name, _type))
        else:
            for path, name, _type in path_name_type_list:
                self.debug_info[name] = path, _type
                if _type not in DATA_TYPES:
                    non_iterable_list.append((path, name, _type))
                else:
                    self.path_name_type_list.append((path, name, _type))

        if len(non_iterable_list) != 0:
            # Some types doesn't support iterable mode
            self.non_iterable_dataset = ESPnetDataset(
                path_name_type_list=non_iterable_list,
                preprocess=preprocess,
                float_dtype=float_dtype,
                int_dtype=int_dtype,
            )
        else:
            self.non_iterable_dataset = None

        self.apply_utt2category = False

    def has_name(self, name) -> bool:
        return name in self.debug_info

    def names(self) -> Tuple[str, ...]:
        return tuple(self.debug_info)

    def __repr__(self):
        _mes = self.__class__.__name__
        _mes += "("
        for name, (path, _type) in self.debug_info.items():
            _mes += f'\n  {name}: {{"path": "{path}", "type": "{_type}"}}'
        _mes += f"\n  preprocess: {self.preprocess})"
        return _mes

    def __iter__(self) -> Iterator[Tuple[Union[str, int], Dict[str, np.ndarray]]]:
        count = 0
        if len(self.path_name_type_list) != 0 and (self.path_name_type_list[0][2] == "bytes" or self.path_name_type_list[0][2] == "waveform"):
            data = {}
            value = self.path_name_type_list[0][0]
            uid = 'utt_id'
            name = self.path_name_type_list[0][1]
            _type = self.path_name_type_list[0][2]
            func = DATA_TYPES[_type]
            array = func(value)
            if self.fs is not None and name == "speech":
                audio_fs = self.fs["audio_fs"]
                model_fs = self.fs["model_fs"]
                if audio_fs is not None and model_fs is not None:
                    array = torch.from_numpy(array)
                    array = array.unsqueeze(0)
                    array = torchaudio.transforms.Resample(orig_freq=audio_fs,
                                                   new_freq=model_fs)(array)
                    array = array.squeeze(0).numpy()
            data[name] = array

            if self.preprocess is not None:
                data = self.preprocess(uid, data)
            for name in data:
                count += 1
                value = data[name]
                if not isinstance(value, np.ndarray):
                    raise RuntimeError(
                        f'All values must be converted to np.ndarray object '
                        f'by preprocessing, but "{name}" is still {type(value)}.')
                # Cast to desired type
                if value.dtype.kind == 'f':
                    value = value.astype(self.float_dtype)
                elif value.dtype.kind == 'i':
                    value = value.astype(self.int_dtype)
                else:
                    raise NotImplementedError(
                        f'Not supported dtype: {value.dtype}')
                data[name] = value

            yield uid, data

        elif len(self.path_name_type_list) != 0 and self.path_name_type_list[0][2] == "sound" and not self.path_name_type_list[0][0].lower().endswith(".scp"):
            data = {}
            value = self.path_name_type_list[0][0]
            uid = os.path.basename(self.path_name_type_list[0][0]).split(".")[0]
            name = self.path_name_type_list[0][1]
            _type = self.path_name_type_list[0][2]
            if _type == "sound":
                audio_type = os.path.basename(value).split(".")[1].lower()
                if audio_type not in SUPPORT_AUDIO_TYPE_SETS:
                    raise NotImplementedError(
                        f'Not supported audio type: {audio_type}')
                if audio_type == "pcm":
                    _type = "pcm"

            func = DATA_TYPES[_type]
            array = func(value)
            if self.fs is not None and name == "speech":
                audio_fs = self.fs["audio_fs"]
                model_fs = self.fs["model_fs"]
                if audio_fs is not None and model_fs is not None:
                    array = torch.from_numpy(array)
                    array = array.unsqueeze(0)
                    array = torchaudio.transforms.Resample(orig_freq=audio_fs,
                                                           new_freq=model_fs)(array)
                    array = array.squeeze(0).numpy()
            data[name] = array

            if self.preprocess is not None:
                data = self.preprocess(uid, data)
            for name in data:
                count += 1
                value = data[name]
                if not isinstance(value, np.ndarray):
                    raise RuntimeError(
                        f'All values must be converted to np.ndarray object '
                        f'by preprocessing, but "{name}" is still {type(value)}.')
                # Cast to desired type
                if value.dtype.kind == 'f':
                    value = value.astype(self.float_dtype)
                elif value.dtype.kind == 'i':
                    value = value.astype(self.int_dtype)
                else:
                    raise NotImplementedError(
                        f'Not supported dtype: {value.dtype}')
                data[name] = value

            if self.mel2ph_file is not None:
                if isinstance(uid, str):
                    uid_list = [uid]
                else:
                    uid_list = uid
                data["mel2ph"] = []
                for _uid in uid_list:
                    mel2ph_item = self.mel2ph_file.get(_uid, None)
                    data["mel2ph"].append(mel2ph_item["mel2ph"] if mel2ph_item is not None else None)

            yield uid, data

        else:
            if self.key_file is not None:
                uid_iter = (
                    line.rstrip().split(maxsplit=1)[0]
                    for line in open(self.key_file, encoding="utf-8")
                )
            elif len(self.path_name_type_list) != 0:
                uid_iter = (
                    line.rstrip().split(maxsplit=1)[0]
                    for line in open(self.path_name_type_list[0][0], encoding="utf-8")
                )
            else:
                uid_iter = iter(self.non_iterable_dataset)

            files = [open(lis[0], encoding="utf-8") for lis in self.path_name_type_list]

            worker_info = torch.utils.data.get_worker_info()

            linenum = 0
            for count, uid in enumerate(uid_iter, 1):
                # If num_workers>=1, split keys
                if worker_info is not None:
                    if (count - 1) % worker_info.num_workers != worker_info.id:
                        continue

                # 1. Read a line from each file
                while True:
                    keys = []
                    values = []
                    for f in files:
                        linenum += 1
                        try:
                            line = next(f)
                        except StopIteration:
                            # print(666, self.path_name_type_list, self.key_file, uid, f)
                            raise RuntimeError(f"{uid} is not found in the files")
                        sps = line.rstrip().split(maxsplit=1)
                        if len(sps) != 2:
                            raise RuntimeError(
                                f"This line doesn't include a space:"
                                f" {f}:L{linenum}: {line})"
                            )
                        key, value = sps
                        keys.append(key)
                        values.append(value)

                    for k_idx, k in enumerate(keys):
                        # print(666, count, uid, k_idx, k)
                        # print(key, value)
                        if k != keys[0]:
                            raise RuntimeError(
                                f"Keys are mismatched. Text files (idx={k_idx}) is "
                                f"not sorted or not having same keys at L{linenum}"
                            )

                    # If the key is matched, break the loop
                    if len(keys) == 0 or keys[0] == uid:
                        break

                # 2. Load the entry from each line and create a dict
                data = {}
                # 2.a. Load data streamingly
                for value, (path, name, _type) in zip(values, self.path_name_type_list):
                    if _type == "sound":
                        audio_type = os.path.basename(value).rsplit(".", 1)[1].lower()
                        if audio_type not in SUPPORT_AUDIO_TYPE_SETS:
                            raise NotImplementedError(
                                f'Not supported audio type: {audio_type}')
                        if audio_type == "pcm":
                            _type = "pcm"
                    func = DATA_TYPES[_type]
                    # Load entry
                    array = func(value)
                    if self.fs is not None and name == "speech":
                        audio_fs = self.fs["audio_fs"]
                        model_fs = self.fs["model_fs"]
                        if audio_fs is not None and model_fs is not None:
                            array = torch.from_numpy(array)
                            array = array.unsqueeze(0)
                            array = torchaudio.transforms.Resample(orig_freq=audio_fs,
                                                                   new_freq=model_fs)(array)
                            array = array.squeeze(0).numpy()
                    data[name] = array
                if self.non_iterable_dataset is not None:
                    # 2.b. Load data from non-iterable dataset
                    _, from_non_iterable = self.non_iterable_dataset[uid]
                    data.update(from_non_iterable)

                # 3. [Option] Apply preprocessing
                #   e.g. funcodec.train.preprocessor:CommonPreprocessor
                if self.preprocess is not None:
                    data = self.preprocess(uid, data)

                # 4. Force data-precision
                for name in data:
                    value = data[name]
                    if not isinstance(value, np.ndarray):
                        raise RuntimeError(
                            f"All values must be converted to np.ndarray object "
                            f'by preprocessing, but "{name}" is still {type(value)}.'
                        )

                    # Cast to desired type
                    if value.dtype.kind == "f":
                        value = value.astype(self.float_dtype)
                    elif value.dtype.kind == "i":
                        value = value.astype(self.int_dtype)
                    else:
                        raise NotImplementedError(f"Not supported dtype: {value.dtype}")
                    data[name] = value

                if self.mel2ph_file is not None:
                    if isinstance(uid, str):
                        uid_list = [uid]
                    else:
                        uid_list = uid
                    data["mel2ph"] = []
                    data["frame2ph_token_ids"] = []
                    for _uid in uid_list:
                        mel2ph_item = self.mel2ph_file.get(_uid, None)
                        data["mel2ph"].append(mel2ph_item["mel2ph"] if mel2ph_item is not None else None)
                        if mel2ph_item is not None:
                            frame2ph_token_ids = mel2ph_item.get("frame2ph_token_ids", None)
                            if frame2ph_token_ids is not None:
                                data["frame2ph_token_ids"].append(frame2ph_token_ids)
                if self.speaker_id_dir is not None:
                    if isinstance(uid, str):
                        uid_list = [uid]
                    else:
                        uid_list = uid
                    data["speaker_ids"] = []
                    for _uid in uid_list:
                        json_path = os.path.join(self.speaker_id_dir, f"{uid}.json")
                        with open(json_path, mode="r+") as f:
                            json_content = json.load(f)
                        data["speaker_ids"].append(json_content.get("spk_id", None))

                yield uid, data

        if count == 0:
            raise RuntimeError("No iteration")
