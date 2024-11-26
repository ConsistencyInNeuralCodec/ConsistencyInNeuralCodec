import os, sys, importlib
sys.path.append("/home/admin_data/user/model")
os.environ["OSS_CONFIG_PATH"] = "/home/admin_data/user/model/ossutil/.oss_config.json"

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

import kaldiio
import numpy as np
import torch
import torchaudio
from typeguard import check_argument_types
from typeguard import check_return_type
from einops import rearrange
from distutils.version import LooseVersion

from funcodec.utils.cli_utils import get_commandline_args
from funcodec.tasks.gan_speech_codec import GANSpeechCodecTask
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
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from funcodec.bin.codec_inference_2 import disable_modules

