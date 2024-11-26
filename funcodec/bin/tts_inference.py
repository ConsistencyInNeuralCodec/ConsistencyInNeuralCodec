#!/usr/bin/env python3
# Copyright FunCodec (https://github.com/alibaba-damo-academy/FunCodec). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import argparse
import logging
import os
import sys
import math
from pathlib import Path
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import Dict

import kaldiio
import numpy as np
import torch
import torchaudio
from typeguard import check_argument_types
from typeguard import check_return_type
from einops import rearrange
from distutils.version import LooseVersion

from funcodec.utils.cli_utils import get_commandline_args
from funcodec.tasks.tts import TextToSpeechCodeTask
from funcodec.torch_utils.device_funcs import to_device
from funcodec.torch_utils.set_all_random_seed import set_all_random_seed
from funcodec.utils import config_argparse
from funcodec.utils.types import str2bool
from funcodec.utils.types import str2triple_str
from funcodec.utils.types import str_or_none
from funcodec.utils.misc import statistic_model_parameters
import json
import torch.nn as nn
from thop import profile
from funcodec.torch_utils.model_summary import tree_layer_info
from funcodec.utils.hinter import hint_once


class Text2SpeechToken(nn.Module):
    """Text2SpeechToken class

    Examples:
        >>> import numpy as np
        >>> text2speechtoken = Text2SpeechToken("config.yml", "model.pth")
        >>> text = np.load("text.npy")
        >>> text_prefix = np.load("text_prefix.npy")
        >>> codec_prefix = np.load("codec_prefix.npy")
        >>> text2speechtoken(text, text_prefix, codec_prefix)
        [(token_id), ...]

    """

    def __init__(
            self,
            config_file: Union[Path, str] = None,
            model_file: Union[Path, str] = None,
            device: str = "cpu",
            batch_size: int = 1,
            dtype: str = "float32",
            top_k: int = -100,
            top_p: float = 1.0,
            temperature: float = 1.0,
            max_steps: int = 1000,
    ):
        super().__init__()
        assert check_argument_types()

        # 1. Build model
        import yaml
        with open(config_file, "rt", encoding="utf-8") as f:
            args = yaml.safe_load(f)
        model, model_args = TextToSpeechCodeTask.build_model_from_file(
            config_file=config_file,
            model_file=model_file,
            device=device
        )
        logging.info("model: {}".format(model))
        logging.info("model parameter number: {}".format(statistic_model_parameters(model)))
        logging.info("model arguments: {}".format(model_args))
        model.to(dtype=getattr(torch, dtype)).eval()

        self.model = model
        self.model_args = model_args
        self.device = device
        self.dtype = dtype
        self.already_stat_flops = False
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.max_steps = max_steps

    @torch.no_grad()
    def __call__(
            self,
            text: Union[torch.Tensor, np.ndarray],
            text_prefix: Union[torch.Tensor, np.ndarray],
            codec_prefix: Union[torch.Tensor, np.ndarray],
            stk_prefix: Union[torch.Tensor, np.ndarray] = None,
            **kwargs,
    ):
        """Inference

        Args:

        Returns:
            generated token_id

        """
        assert check_argument_types()
        self.model.eval()
        if isinstance(text, np.ndarray):
            text = torch.from_numpy(text)
        if isinstance(text_prefix, np.ndarray):
            text_prefix = torch.from_numpy(text_prefix)
        if isinstance(codec_prefix, np.ndarray):
            codec_prefix = torch.from_numpy(codec_prefix)

        text_prefix = text_prefix.to(self.device)
        text = text.to(self.device)
        codec_prefix = codec_prefix.to(self.device)

        batch = [text, text_prefix, codec_prefix]

        if stk_prefix is not None:
            if isinstance(stk_prefix, np.ndarray):
                stk_prefix = torch.from_numpy(stk_prefix)
            stk_prefix = stk_prefix.to(self.device)
            batch.append(stk_prefix)

        gen_token_id = self.model.inference(*batch, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature,
                                            max_steps=self.max_steps)

        results = (
            gen_token_id,
        )
        return results

    @staticmethod
    def from_pretrained(
            model_tag: Optional[str] = None,
            **kwargs: Optional[Any],
    ):
        """Build Text2SpeechToken instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models. Currently, not used.

        Returns:
            Text2SpeechToken: Text2SpeechToken instance.

        """
        return Text2SpeechToken(**kwargs)


def save_audio(wav: torch.Tensor, path: Union[Path, str],
               sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    torchaudio.save(path, wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)


def inference_modelscope(
        output_dir: Optional[str] = None,
        batch_size: int = 1,
        dtype: str = "float32",
        ngpu: int = 1,
        seed: int = 0,
        num_workers: int = 0,
        log_level: Union[int, str] = "INFO",
        key_file: Optional[str] = None,
        config_file: Optional[str] = "config.yaml",
        model_file: Optional[str] = "model.pth",
        model_tag: Optional[str] = None,
        allow_variable_data_keys: bool = True,
        top_k: int = -100,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_steps: int = 1000,
        param_dict: Optional[dict] = None,
        **kwargs,
):
    assert check_argument_types()
    if batch_size > 1:
        logging.info(f"batch_size = {batch_size}")
        raise NotImplementedError("only batch_size == 1 is supported now")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logging.info("param_dict: {}".format(param_dict))

    if ngpu >= 1 and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build model
    model_kwargs = dict(
        config_file=config_file,
        model_file=model_file,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        max_steps=max_steps
    )
    logging.info("model_kwargs: {}".format(model_kwargs))
    my_model = Text2SpeechToken.from_pretrained(
        model_tag=model_tag,
        **model_kwargs,
    )
    my_model.model.eval()
    my_model.already_stat_flops = False

    def _forward(
            data_path_and_name_and_type: Sequence[Tuple[str, str, str]] = None,
            raw_inputs: Union[np.ndarray, torch.Tensor] = None,
            output_dir_v2: Optional[str] = None,
            param_dict: Optional[dict] = None,
    ):
        set_all_random_seed(seed)
        logging.info("param_dict: {}".format(param_dict))
        if data_path_and_name_and_type is None and raw_inputs is not None:
            raise NotImplementedError

        # 3. Build data-iterator
        loader = TextToSpeechCodeTask.build_streaming_iterator(
            data_path_and_name_and_type,
            dtype=dtype,
            batch_size=batch_size,
            key_file=key_file,
            num_workers=num_workers,
            preprocess_fn=None,
            collate_fn=TextToSpeechCodeTask.build_collate_fn(argparse.Namespace(
                float_pad_value=0.0,
                int_pad_value=0,
                pad_mode=None,
            ), False),
            allow_variable_data_keys=allow_variable_data_keys,
            inference=True,
        )
        logging.info(f"666 data_loader = {type(loader)}") # IterableESPnetDataset

        output_path = output_dir_v2 if output_dir_v2 is not None else output_dir
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        result_list = []

        indices_writer = open(os.path.join(output_path, "codecs.txt"), "wt")

        def write_indices(_key, _indices, batch_id=0):
            x = _indices.cpu().numpy()
            array = np.expand_dims(x[batch_id].T, axis=0)
            json_str = json.dumps(array.tolist())
            indices_writer.write(_key + " " + json_str + "\n")

        counts = kwargs.get("counts", None)
        tot_counts = 0
        for keys, batch in loader:
            if counts is not None and tot_counts > counts: break
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            if kwargs["stat_flops"] and not my_model.already_stat_flops:
                pass
            token_id, = my_model(**batch)

            for i, key in enumerate(keys):
                item = {"key": key, "value": token_id[i].cpu().numpy()}
                logging.info(f"666 {key} = {token_id[i].shape}")
                if output_path is not None:
                    if token_id is not None:
                        write_indices(key, token_id, batch_id=i)
                else:
                    result_list.append(item)
            tot_counts += 1

        return result_list

    return _forward


def inference(
        output_dir: Optional[str],
        batch_size: int,
        dtype: str,
        ngpu: int,
        seed: int,
        num_workers: int,
        log_level: Union[int, str],
        data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
        key_file: Optional[str],
        config_file: Optional[str],
        model_file: Optional[str],
        model_tag: Optional[str],
        allow_variable_data_keys: bool = True,
        top_k: int = -100,
        top_p: float = 1.0,
        temperature: float = 1.0,
        max_steps: int = 1000,
        **kwargs,
):
    inference_pipeline = inference_modelscope(
        output_dir=output_dir,
        batch_size=batch_size,
        dtype=dtype,
        ngpu=ngpu,
        seed=seed,
        num_workers=num_workers,
        log_level=log_level,
        key_file=key_file,
        config_file=config_file,
        model_file=model_file,
        model_tag=model_tag,
        allow_variable_data_keys=allow_variable_data_keys,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        max_steps=max_steps,
        **kwargs,
    )

    return inference_pipeline(data_path_and_name_and_type, raw_inputs=None)


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Text2Speech Tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--gpuid_list",
        type=str,
        default="",
        help="The visible gpus",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=False,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--config_file",
        type=str,
        help="path to configuration file",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="path to model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, *_train_config and "
             "*_file will be overwritten",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group.add_argument(
        "--stat_flops",
        type=str2bool,
        default=False,
        help="whether to statistic flops",
    )

    parser.add_argument(
        "--counts",
        type=int,
        default=-1,
        help="The counts for inference",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=-100,
        help="The counts for inference",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="The counts for inference",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The counts for inference",
    )

    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000,
        help="The max_steps for inference",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    if args.counts == -1: delattr(args, "counts")
    args.train_mode = 0
    kwargs = vars(args)
    kwargs.pop("config", None)
    if args.output_dir is None:
        jobid, n_gpu = 1, 1
        gpuid = args.gpuid_list.split(",")[jobid - 1]
    else:
        jobid = int(args.output_dir.split(".")[-1])
        n_gpu = len(args.gpuid_list.split(","))
        gpuid = args.gpuid_list.split(",")[(jobid - 1) % n_gpu]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    # if LooseVersion(torch.__version__) >= LooseVersion("1.10"):
    #    torch.cuda.set_device(int(gpuid))
    inference(**kwargs)


if __name__ == "__main__":
    main()
