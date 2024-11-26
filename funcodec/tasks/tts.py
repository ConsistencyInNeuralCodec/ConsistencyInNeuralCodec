# Copyright 2021 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""text-to-speech code task."""

import argparse
import logging

from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from funcodec.torch_utils.model_summary import model_summary
import numpy as np
import torch

from typeguard import check_argument_types
from typeguard import check_return_type

from funcodec.train.abs_espnet_model import AbsESPnetModel
from funcodec.models.tts_valle import VALLE
from funcodec.models.tts_spear import SPEARTTS
from funcodec.models.tts_mslm import MSVoiceGen
from funcodec.models.tts_megabyte import UniAudio
from funcodec.tasks.abs_task import AbsTask
from funcodec.tasks.abs_task import optim_classes
from funcodec.train.class_choices import ClassChoices
from funcodec.datasets.collate_fn import CommonCollateFn
from funcodec.train.trainer import Trainer
from funcodec.datasets.preprocessor import CodecPreprocessor
from funcodec.utils.types import float_or_none
from funcodec.utils.types import int_or_none
from funcodec.utils.types import str2bool
from funcodec.utils.types import str_or_none
from funcodec.models.frontend.abs_frontend import AbsFrontend
from funcodec.models.frontend.default import DefaultFrontend
from funcodec.models.frontend.fused import FusedFrontends
from funcodec.models.frontend.s3prl import S3prlFrontend
from funcodec.models.frontend.wav_frontend import WavFrontend
from funcodec.models.frontend.windowing import SlidingWindow

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        fused=FusedFrontends,
        wav_frontend=WavFrontend,
    ),
    type_check=AbsFrontend,
    default=None,
    optional=True,
)

model_choices = ClassChoices(
    "model",
    classes=dict(
        valle=VALLE,
        speartts=SPEARTTS,
        msvoicegen=MSVoiceGen,
        uniaudio=UniAudio,
    ),
    type_check=AbsESPnetModel,
    default="valle",
)



class TextToSpeechCodeTask(AbsTask):
    """GAN-based speech tokenizer task."""

    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --model and --model_conf
        model_choices,
    ]

    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        assert check_argument_types()
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of dimension of inputs",
        )

        group.add_argument(
            "--train_mode",
            type=int_or_none,
            default=None,
            help="train mode",
        )

        group.add_argument(
            "--stat_flops",
            type=str2bool,
            default=False,
            help="whether to statistic flops."
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Apply preprocessing to data or not",
        )
        # speech_max_dur
        group.add_argument(
            "--speech_max_length",
            type=int,
            default=50000,
            help="The maximum duration of speech for training",
        )
        group.add_argument(
            "--sampling_rate",
            type=int,
            default=16_000,
            help="The sampling rate of input waveforms"
        )
        group.add_argument(
            "--valid_max_length",
            type=int,
            default=50000,
            help="The maximum duration of speech for valid"
        )

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        return CommonCollateFn(
            float_pad_value=args.__dict__.get("float_pad_value", 0.0),
            int_pad_value=args.__dict__.get("int_pad_value", 0),
            pad_mode=args.__dict__.get("pad_mode", None)
        )

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            retval = None
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("text", "codec")
        else:
            # Inference mode
            retval = ("text", "text_prefix", "codec_prefix")
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("stk", "stl", "lang", "dist")
        else:
            # Inference mode
            retval = ("stk", "stl", "lang", "stk_prefix", "dist")
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> AbsESPnetModel:
        assert check_argument_types()
        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            if args.frontend == 'wav_frontend':
                frontend = frontend_class(cmvn_file=args.cmvn_file, **args.frontend_conf)
            else:
                frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        if not hasattr(args, "train_mode") or args.train_mode is None:
            args.train_mode = 0

        model_class = model_choices.get_class(args.model)
        model = model_class(
            **args.model_conf,
            train_mode=args.train_mode,
        )

        if hasattr(args, "stat_flops") and args.stat_flops:
            pass
        assert check_return_type(model)
        return model
