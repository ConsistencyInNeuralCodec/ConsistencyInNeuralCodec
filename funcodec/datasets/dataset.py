# Copyright ESPnet (https://github.com/espnet/espnet). All Rights Reserved.
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
from abc import ABC
from abc import abstractmethod
import collections
import copy
import functools
import logging
import numbers
import re
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Mapping
from typing import Tuple
from typing import Union
import random

import h5py
import humanfriendly
import kaldiio
import json
import jsonlines
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from typeguard import check_argument_types
from typeguard import check_return_type

from funcodec.fileio.sound_scp import SoundScpReader
from funcodec.fileio.npy_scp import NpyScpReader
from funcodec.fileio.rand_gen_dataset import FloatRandomGenerateDataset
from funcodec.fileio.rand_gen_dataset import IntRandomGenerateDataset
from funcodec.fileio.read_text import load_num_sequence_text
from funcodec.fileio.read_text import read_2column_text
from funcodec.fileio.read_text import load_jsonl_trans_int
from funcodec.datasets.oss_reader import OssReader
from funcodec.utils.sized_dict import SizedDict


class AdapterForSoundScpReader(collections.abc.Mapping):
    def __init__(self, loader, dtype=None):
        assert check_argument_types()
        self.loader = loader
        self.dtype = dtype
        self.rate = None

    def keys(self):
        return self.loader.keys()

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)

    def __getitem__(self, key: str) -> np.ndarray:
        retval = self.loader[key]

        if retval is None or retval[0] is None or retval[1] is None:
            # reader error, return None, let dataset class to resample another audio
            return None

        if isinstance(retval, tuple):
            assert len(retval) == 2, len(retval)
            if isinstance(retval[0], int) and isinstance(retval[1], np.ndarray):
                # sound scp case
                rate, array = retval
            elif isinstance(retval[1], int) and isinstance(retval[0], np.ndarray):
                # Extended ark format case
                array, rate = retval
            else:
                raise RuntimeError(
                    f"Unexpected type: {type(retval[0])}, {type(retval[1])}"
                )

            if self.rate is not None and self.rate != rate:
                raise RuntimeError(
                    f"Sampling rates are mismatched: {self.rate} != {rate}"
                )
            self.rate = rate
            # Multichannel wave file
            # array: (NSample, Channel) or (Nsample)
            if array.dtype == np.int16:
                array = array / (2 ** 15)
            elif array.dtype == np.int32:
                array = array / (2 ** 31)
            if self.dtype is not None:
                array = array.astype(self.dtype)

        else:
            # Normal ark case
            assert isinstance(retval, np.ndarray), type(retval)
            array = retval
            if self.dtype is not None:
                array = array.astype(self.dtype)

        assert isinstance(array, np.ndarray), type(array)
        return array


class H5FileWrapper:
    def __init__(self, path: str):
        self.path = path
        self.h5_file = h5py.File(path, "r")

    def __repr__(self) -> str:
        return str(self.h5_file)

    def __len__(self) -> int:
        return len(self.h5_file)

    def __iter__(self):
        return iter(self.h5_file)

    def __getitem__(self, key) -> np.ndarray:
        value = self.h5_file[key]
        return value[()]


def sound_loader(path, float_dtype=None, sampling_rate=16000):
    # The file is as follows:
    #   utterance_id_A /some/where/a.wav
    #   utterance_id_B /some/where/a.flac

    # NOTE(kamo): SoundScpReader doesn't support pipe-fashion
    # like Kaldi e.g. "cat a.wav |".
    # NOTE(kamo): The audio signal is normalized to [-1,1] range.
    loader = SoundScpReader(path, sampling_rate=sampling_rate, normalize=True, always_2d=False)

    # SoundScpReader.__getitem__() returns Tuple[int, ndarray],
    # but ndarray is desired, so Adapter class is inserted here
    return AdapterForSoundScpReader(loader, float_dtype)


def kaldi_loader(path, float_dtype=None, max_cache_fd: int = 0):
    loader = kaldiio.load_scp(path, max_cache_fd=max_cache_fd)
    return AdapterForSoundScpReader(loader, float_dtype)


def rand_int_loader(filepath, loader_type):
    # e.g. rand_int_3_10
    try:
        low, high = map(int, loader_type[len("rand_int_") :].split("_"))
    except ValueError:
        raise RuntimeError(f"e.g rand_int_3_10: but got {loader_type}")
    return IntRandomGenerateDataset(filepath, low, high)


def build_codec_loader(filepath, quant_groups=32, file_type="ark"):
    from funcodec.fileio.codec_loader import CodecLoader
    return CodecLoader(filepath, quant_groups=quant_groups, file_type=file_type)


DATA_TYPES = {
    "sound": dict(
        func=sound_loader,
        kwargs=["float_dtype"],
        help="Audio format types which supported by sndfile wav, flac, etc."
        "\n\n"
        "   utterance_id_a a.wav\n"
        "   utterance_id_b b.wav\n"
        "   ...",
    ),
    "kaldi_ark": dict(
        func=kaldi_loader,
        kwargs=["max_cache_fd"],
        help="Kaldi-ark file type."
        "\n\n"
        "   utterance_id_A /some/where/a.ark:123\n"
        "   utterance_id_B /some/where/a.ark:456\n"
        "   ...",
    ),
    "npy": dict(
        func=NpyScpReader,
        kwargs=[],
        help="Npy file format."
        "\n\n"
        "   utterance_id_A /some/where/a.npy\n"
        "   utterance_id_B /some/where/b.npy\n"
        "   ...",
    ),
    "text_int": dict(
        func=functools.partial(load_num_sequence_text, loader_type="text_int"),
        kwargs=[],
        help="A text file in which is written a sequence of interger numbers "
        "separated by space."
        "\n\n"
        "   utterance_id_A 12 0 1 3\n"
        "   utterance_id_B 3 3 1\n"
        "   ...",
    ),
    "csv_int": dict(
        func=functools.partial(load_num_sequence_text, loader_type="csv_int"),
        kwargs=[],
        help="A text file in which is written a sequence of interger numbers "
        "separated by comma."
        "\n\n"
        "   utterance_id_A 100,80\n"
        "   utterance_id_B 143,80\n"
        "   ...",
    ),
    "text_float": dict(
        func=functools.partial(load_num_sequence_text, loader_type="text_float"),
        kwargs=[],
        help="A text file in which is written a sequence of float numbers "
        "separated by space."
        "\n\n"
        "   utterance_id_A 12. 3.1 3.4 4.4\n"
        "   utterance_id_B 3. 3.12 1.1\n"
        "   ...",
    ),
    "csv_float": dict(
        func=functools.partial(load_num_sequence_text, loader_type="csv_float"),
        kwargs=[],
        help="A text file in which is written a sequence of float numbers "
        "separated by comma."
        "\n\n"
        "   utterance_id_A 12.,3.1,3.4,4.4\n"
        "   utterance_id_B 3.,3.12,1.1\n"
        "   ...",
    ),
    "text": dict(
        func=read_2column_text,
        kwargs=[],
        help="Return text as is. The text must be converted to ndarray "
        "by 'preprocess'."
        "\n\n"
        "   utterance_id_A hello world\n"
        "   utterance_id_B foo bar\n"
        "   ...",
    ),
    "hdf5": dict(
        func=H5FileWrapper,
        kwargs=[],
        help="A HDF5 file which contains arrays at the first level or the second level."
        "   >>> f = h5py.File('file.h5')\n"
        "   >>> array1 = f['utterance_id_A']\n"
        "   >>> array2 = f['utterance_id_B']\n",
    ),
    "rand_float": dict(
        func=FloatRandomGenerateDataset,
        kwargs=[],
        help="Generate random float-ndarray which has the given shapes "
        "in the file."
        "\n\n"
        "   utterance_id_A 3,4\n"
        "   utterance_id_B 10,4\n"
        "   ...",
    ),
    "rand_int_\\d+_\\d+": dict(
        func=rand_int_loader,
        kwargs=["loader_type"],
        help="e.g. 'rand_int_0_10'. Generate random int-ndarray which has the given "
        "shapes in the path. "
        "Give the lower and upper value by the file type. e.g. "
        "rand_int_0_10 -> Generate integers from 0 to 10."
        "\n\n"
        "   utterance_id_A 3,4\n"
        "   utterance_id_B 10,4\n"
        "   ...",
    ),
}


class AbsDataset(Dataset, ABC):
    @abstractmethod
    def has_name(self, name) -> bool:
        raise NotImplementedError

    @abstractmethod
    def names(self) -> Tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, uid) -> Tuple[Any, Dict[str, np.ndarray]]:
        raise NotImplementedError


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


class ESPnetDataset(AbsDataset):
    """Pytorch Dataset class for ESPNet.

    Examples:
        >>> dataset = ESPnetDataset([('wav.scp', 'input', 'sound'),
        ...                          ('token_int', 'output', 'text_int')],
        ...                         )
        ... uttid, data = dataset['uttid']
        {'input': per_utt_array, 'output': per_utt_array}
    """

    def __init__(
        self,
        path_name_type_list: Collection[Tuple[str, str, str]],
        preprocess: Callable[
            [str, Dict[str, np.ndarray]], Dict[str, np.ndarray]
        ] = None,
        float_dtype: str = "float32",
        int_dtype: str = "long",
        max_cache_size: Union[float, int, str] = 0.0,
        max_cache_fd: int = 0,
        sampling_rate: int = 16000,
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
        self.max_cache_fd = max_cache_fd
        self.sampling_rate = sampling_rate

        self.loader_dict = {}
        self.debug_info = {}
        for path, name, _type in path_name_type_list:
            if name in self.loader_dict:
                raise RuntimeError(f'"{name}" is duplicated for data-key')

            loader = self._build_loader(path, _type)
            if not hasattr(self, 'all_keys'):
                self.all_keys = list(loader.keys())

            self.loader_dict[name] = loader
            self.debug_info[name] = path, _type
            if len(self.loader_dict[name]) == 0:
                raise RuntimeError(f"{path} has no samples")

            # TODO(kamo): Should check consistency of each utt-keys?

        if isinstance(max_cache_size, str):
            max_cache_size = humanfriendly.parse_size(max_cache_size)
        self.max_cache_size = max_cache_size
        if max_cache_size > 0:
            self.cache = SizedDict(shared=True)
        else:
            self.cache = None

        # self.mel2ph_file = None if mel2ph_file is None else load_mel2ph(mel2ph_file)
        self.mel2ph_file = None
        self.mel2ph_dir = None
        if mel2ph_file is None:
            self.mel2ph_file = None
        elif os.path.isdir(mel2ph_file):
            self.mel2ph_dir = mel2ph_file
        elif os.path.isfile(mel2ph_file):
            self.mel2ph_file = load_mel2ph(mel2ph_file)
        self.speaker_id_dir = speaker_id_dir

    def _build_loader(
        self, path: str, loader_type: str
    ) -> Mapping[str, Union[np.ndarray, torch.Tensor, str, numbers.Number]]:
        """Helper function to instantiate Loader.

        Args:
            path:  The file path
            loader_type:  loader_type. sound, npy, text_int, text_float, etc
        """
        for key, dic in DATA_TYPES.items():
            # e.g. loader_type="sound"
            # -> return DATA_TYPES["sound"]["func"](path)
            if re.match(key, loader_type):
                kwargs = {}
                for key2 in dic["kwargs"]:
                    if key2 == "loader_type":
                        kwargs["loader_type"] = loader_type
                    elif key2 == "float_dtype":
                        kwargs["float_dtype"] = self.float_dtype
                    elif key2 == "int_dtype":
                        kwargs["int_dtype"] = self.int_dtype
                    elif key2 == "max_cache_fd":
                        kwargs["max_cache_fd"] = self.max_cache_fd
                    else:
                        raise RuntimeError(f"Not implemented keyword argument: {key2}")

                func = dic["func"]
                if key == 'sound':
                    kwargs['sampling_rate'] = self.sampling_rate
                try:
                    return func(path, **kwargs)
                except Exception:
                    if hasattr(func, "__name__"):
                        name = func.__name__
                    else:
                        name = str(func)
                    logging.error(f"An error happened with {name}({path})")
                    raise
        else:
            raise RuntimeError(f"Not supported: loader_type={loader_type}")

    def has_name(self, name) -> bool:
        return name in self.loader_dict

    def names(self) -> Tuple[str, ...]:
        return tuple(self.loader_dict)

    def __iter__(self):
        return iter(next(iter(self.loader_dict.values())))

    def __repr__(self):
        _mes = self.__class__.__name__
        _mes += "("
        for name, (path, _type) in self.debug_info.items():
            _mes += f'\n  {name}: {{"path": "{path}", "type": "{_type}"}}'
        _mes += f"\n  preprocess: {self.preprocess})"
        return _mes

    def __getitem__(self, uid: Union[str, int]) -> Tuple[str, Dict[str, np.ndarray]]:
        assert check_argument_types()
        retry = 10
        for _ in range(retry):
            fail_flag = False
            # Change integer-id to string-id
            if isinstance(uid, int):
                d = next(iter(self.loader_dict.values()))
                uid = list(d)[uid]

            if self.cache is not None and uid in self.cache:
                data = self.cache[uid]
                return uid, data

            data = {}
            # 1. Load data from each loaders
            for name, loader in self.loader_dict.items():
                try:
                    value = loader[uid]
                    if value is None:
                        fail_flag = True
                        break
                    if isinstance(value, (list, tuple)):
                        value = np.array(value)
                    if not isinstance(
                        value, (np.ndarray, torch.Tensor, str, numbers.Number)
                    ):
                        raise TypeError(
                            f"Must be ndarray, torch.Tensor, str or Number: {type(value)}"
                        )
                except Exception:
                    path, _type = self.debug_info[name]
                    logging.error(
                        f"Error happened with path={path}, type={_type}, id={uid}"
                    )
                    raise

                # torch.Tensor is converted to ndarray
                if isinstance(value, torch.Tensor):
                    value = value.numpy()
                elif isinstance(value, numbers.Number):
                    value = np.array([value])
                data[name] = value

            if fail_flag is True:
                uid = random.choice(self.all_keys)
                continue

            # 2. [Option] Apply preprocessing
            #   e.g. funcodec.train.preprocessor:CommonPreprocessor
            if self.preprocess is not None:
                data = self.preprocess(uid, data)

            # 3. Force data-precision
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

            if self.cache is not None and self.cache.size < self.max_cache_size:
                self.cache[uid] = data

            retval = uid, data
            assert check_return_type(retval)
            if self.mel2ph_file is not None:
                # data["mel2ph"] = self.mel2ph_file[uid]["mel2ph"]
                mel2ph_item = self.mel2ph_file[uid]
                data["mel2ph"] = mel2ph_item.get("mel2ph", None)
                frame2ph_token_ids = mel2ph_item.get("frame2ph_token_ids", None)
                if frame2ph_token_ids is not None:
                    data["frame2ph_token_ids"] = frame2ph_token_ids
            elif self.mel2ph_dir is not None:
                json_path = os.path.join(self.mel2ph_dir, f"{uid}.json")
                with open(json_path, mode="r+") as f:
                    json_content = json.load(f)
                data["mel2ph"] = json_content.get("mel2ph", None)
                frame2ph_token_ids = json_content.get("frame2ph_token_ids", None)
                if frame2ph_token_ids is not None:
                    data["frame2ph_token_ids"] = frame2ph_token_ids
            if self.speaker_id_dir is not None:
                json_path = os.path.join(self.speaker_id_dir, f"{uid}.json")
                with open(json_path, mode="r+") as f:
                    json_content = json.load(f)
                data["speaker_ids"] = json_content.get("spk_id", None)

            return retval
