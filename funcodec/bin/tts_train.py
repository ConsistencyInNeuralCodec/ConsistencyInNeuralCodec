#!/usr/bin/env python3

import os
import torch
from funcodec.tasks.tts import TextToSpeechCodeTask


# for ASR Training
def parse_args():
    parser = TextToSpeechCodeTask.get_parser()
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="local gpu id.",
    )
    args = parser.parse_args()
    return args


def main(args=None, cmd=None):
    # for text-to-speech code training
    TextToSpeechCodeTask.main(args=args, cmd=cmd)


if __name__ == '__main__':
    args = parse_args()

    # setup local gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    # if torch.__version__ >= "1.7.2":
    #     torch.cuda.set_device(args.gpu_id)

    # DDP settings
    if args.ngpu > 1:
        args.distributed = True
    else:
        args.distributed = False
    assert args.num_worker_count == 1

    # re-compute batch size: when dataset type is small
    if args.dataset_type == "small":
        if args.batch_size is not None:
            args.batch_size = args.batch_size * args.ngpu
        if args.batch_bins is not None:
            args.batch_bins = args.batch_bins * args.ngpu

    main(args=args)
