import os
import numpy as np
import librosa
import json
from scipy.io.wavfile import write

import gc
from collections import OrderedDict
import multiprocessing
import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import fnmatch
import kaldiio
import tqdm

import glob
from collections import OrderedDict

from funcodec.text.ali_tokenizer import TextTokenizer

def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.
    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.
    Returns:
        list: List of found filenames.
    """
    files = []
    for root, _, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]
    return files

def build_ark_writer(file_path, write_function="pickle"):
    return kaldiio.WriteHelper('ark,scp:{}.ark,{}.scp'.format(os.path.abspath(file_path), os.path.abspath(file_path)),
                               write_function=write_function)

def trans_to_symbol(trans_file, out_file, rsc_zip_file, config_file, split_size=500000, nj=1):
    def atom_op(file_in, file_out, file_rsc):
        tokenizer = TextTokenizer(file_rsc, config_file, lang_type="en-us")

        trans_lst = []
        with open(file_in, "r") as fr:
            for line in fr:
                utt_id, trans = line.strip().split('\t')[:2]
                trans_lst.append("{}\t{}\n".format(utt_id, trans))

        with open(file_out, "w") as fw:
            for symbol_str in tokenizer.encoding(trans_lst):
                fw.write("{}".format(symbol_str))

    if nj == 1:
        return atom_op(trans_file, out_file, rsc_zip_file)
    else:
        out_dir = os.path.dirname(out_file)
        os.system("split -l {} {} -d -a 10 {}/{}.split_".format(split_size, trans_file, out_dir, os.path.basename(trans_file)))
        trans_file_split_lst = find_files(out_dir, query="{}.split_*".format(os.path.basename(trans_file)))
        out_file_split_lst = ["{}/{}.split_{}".format(out_dir, os.path.basename(out_file), i) for i in
                              range(len(trans_file_split_lst))]
        print(f"{len(trans_file_split_lst)} files are allocated to {nj} processes.")

        # executor = ProcessPoolExecutor(max_workers=nj)
        # results = []
        # for (trans, out) in zip(trans_file_split_lst, out_file_split_lst):
        #     results.append(executor.submit(partial(atom_op, trans, out, rsc_zip_file)))
        # _ = [x.result() for x in tqdm(results)]

        with ProcessPoolExecutor(max_workers=nj) as executor:
            futures = [executor.submit(partial(atom_op, trans, out, rsc_zip_file)) for trans, out in zip(trans_file_split_lst, out_file_split_lst)]
            # 使用as_completed来迭代已完成的futures，这样可以逐个处理结果
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                result = future.result()  # 获取结果，这里您可以选择如何处理这些结果
                del result
                gc.collect()

        for file in out_file_split_lst:
            os.system("cat {} >> {} && rm -f {}".format(file, out_file, file))
        for file in trans_file_split_lst:
            os.system("rm -f {}".format(file))

def trans_to_ids(file_in, file_out, file_rsc, file_cfg, lang_type="en-us"):
    tokenizer = TextTokenizer(file_rsc, file_cfg, lang_type=lang_type)
    print(f"666 load text_tokenizer")

    trans_lst = []
    with open(file_in, "r") as fr:
        for line in fr:
            utt_id, trans = line.strip().split('\t')[:2]
            # print(f"666 {utt_id} {trans}")
            trans_lst.append("{}\t{}\n".format(utt_id, trans))

    symbol_res_list = []
    with build_ark_writer(file_out) as ark_writer:
        # for trans in trans_lst:
        for trans in tqdm.tqdm(trans_lst, desc=os.path.basename(file_out)):
            # print(f"666 trans = {trans}")
            if True:
            # try:
                symbol_res = tokenizer.encoding([trans], ret_type="index")[0]
                utt_id, symbol_ids = symbol_res
                ark_writer(utt_id, symbol_ids)
                symbol_res_list.append(symbol_res)
            else:
            # except:
                print(f"{trans} error")
        # for symbol_res in tokenizer.encoding(trans_lst, ret_type="index"):
        #     utt_id, symbol_ids = symbol_res
        #     ark_writer(utt_id, symbol_ids)
    print(f"{os.path.basename(file_out)} completed")
    # print(666, len(trans_lst), trans_lst)
    print(f"666 end text_tokenizer {type(tokenizer)}")

def _trans_to_ids(args):
    trans, out, rsc_zip_file, config_file, lang_type = args
    return trans_to_ids(trans, out, rsc_zip_file, config_file, lang_type)

def _prepare_args(trans_file_split_lst, out_file_split_lst, rsc_zip_file, config_file, lang_type):
    return [
        (trans, out, rsc_zip_file, config_file, lang_type)
        for trans, out in zip(trans_file_split_lst, out_file_split_lst)
    ]

def trans_to_ids_mp(trans_file, out_file, rsc_zip_file, config_file, lang_type, split_size=500000, nj=1):

    if nj == 1:
        return trans_to_ids(trans_file, out_file, rsc_zip_file, config_file, lang_type)
    else:
        out_dir = os.path.dirname(out_file)
        # 开始时执行
        os.system("split -l {} {} -d -a 10 {}/{}.split_".format(split_size, trans_file, out_dir, os.path.basename(trans_file)))
        trans_file_split_lst = find_files(out_dir, query="{}.split_*".format(os.path.basename(trans_file)))
        out_file_split_lst = ["{}/{}.split_{}".format(out_dir, os.path.basename(out_file), i) for i in range(len(trans_file_split_lst))]

        # print(trans_file_split_lst); print(); print(out_file_split_lst); return
        # start_idx, end_idx = 0, 50
        # start_idx, end_idx = 50, len(trans_file_split_lst)
        # if start_idx is not None and end_idx is not None:
        #     print(f"interval = [{start_idx}, {end_idx})")
        #     trans_file_split_lst = trans_file_split_lst[start_idx: end_idx]
        #     out_file_split_lst = out_file_split_lst[start_idx: end_idx]
        # idx_list = [120, 132, 149, 163, 175, 220]
        # trans_file_split_lst = [trans_file_split_lst[idx] for idx in idx_list]
        # out_file_split_lst = [out_file_split_lst[idx] for idx in idx_list]
        # print(f"666 {len(trans_file_split_lst)} files are allocated to {nj} processes.")

        # executor = ProcessPoolExecutor(max_workers=nj)
        # results = []
        # for (trans, out) in zip(trans_file_split_lst, out_file_split_lst):
        #     results.append(executor.submit(partial(trans_to_ids, trans, out, rsc_zip_file, config_file, lang_type)))
        # _ = [x.result() for x in tqdm.tqdm(results)]

        with ProcessPoolExecutor(max_workers=nj) as executor:
            futures = [executor.submit(partial(trans_to_ids, trans, out, rsc_zip_file, config_file, lang_type)) for trans, out in zip(trans_file_split_lst, out_file_split_lst)]
            # 使用as_completed来迭代已完成的futures，这样可以逐个处理结果
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                del result
                gc.collect()

        # prepared_args = _prepare_args(trans_file_split_lst, out_file_split_lst, rsc_zip_file, config_file, lang_type)
        # with multiprocessing.Pool(nj) as p:
        #     results_iter = p.imap(_trans_to_ids, prepared_args)
        #     results = list(results_iter)

        # 结束后执行
        for file in out_file_split_lst:
            os.system("cat {}.scp >> {}.scp".format(file, out_file))
        for file in trans_file_split_lst:
            os.system("rm -f {}".format(file))


def read_trans_tsv(trans_tsv_file):
    trans_lst = []
    with open(trans_tsv_file, "r") as fr:
        for line in fr.readlines():
            parts = line.strip().split('\t')
            utt_id = parts[0]
            raw_txt = parts[1]
            normalized_txt = parts[-1]
            trans_lst.append((utt_id, raw_txt, normalized_txt))

    return trans_lst

def run_codec_encoding_cmd(codec_bin_dir, codec_model_dir, wav_scp, out_dir, model_name=None, gpu="0", njob=1, bit_width=4000):
    codec_bin_dir = os.path.abspath(codec_bin_dir)
    codec_model_dir = os.path.abspath(codec_model_dir)
    wav_scp = os.path.abspath(wav_scp)
    out_dir = os.path.abspath(out_dir)
    model_path = os.path.join(codec_model_dir, model_name)
    config_path = os.path.join(codec_model_dir, "config.yaml")

    os.system("cd {} && bash encoding_decoding_empty.sh --stage 1 --wav_scp {} --out_dir {} "
              "--gpu_devices {} --njob {} --bit_width {} --model_dir {} --model_path {} --config_path {} && cd - ".format(
        codec_bin_dir, wav_scp, out_dir, gpu, njob, bit_width, codec_model_dir, model_path, config_path))

def run_codec_decoding_cmd(codec_bin_dir, codec_model_dir, codec_scp, out_dir, model_name=None, gpu="0", njob=1, bit_width=4000):
    codec_bin_dir = os.path.abspath(codec_bin_dir)
    print(f"666 bin_dir = {codec_bin_dir}")
    codec_model_dir = os.path.abspath(codec_model_dir)
    codec_scp = os.path.abspath(codec_scp)
    out_dir = os.path.abspath(out_dir)
    model_path = os.path.join(codec_model_dir, model_name)
    config_path = os.path.join(codec_model_dir, "config.yaml")

    # os.system("cd {} && bash encoding_decoding_empty.sh --stage 2 --wav_scp {} --out_dir {} "
    #           "--gpu_devices {} --njob {} --bit_width {} --model_dir {} --model_path {} --config_path {} && cd - ".format(
    #     codec_bin_dir, codec_scp, out_dir, gpu, njob, bit_width, codec_model_dir, model_path, config_path))
    os.system("cd {} && bash encoding_decoding_empty.sh --stage 2 --codec_scp {} --out_dir {} "
              "--gpu_devices {} --njob {} --bit_width {} --model_dir {} --model_path {} --config_path {} && cd - ".format(
        codec_bin_dir, codec_scp, out_dir, gpu, njob, bit_width, codec_model_dir, model_path, config_path))
    os.system("mv {}/logdir/output.*/*.wav {}".format(out_dir, out_dir))


def codec_txt2ark(args):
    with open(args.codec_in, "r") as fr, \
        build_ark_writer(args.codec_out) as codec_writer:
        for line in fr.readlines():
            parts = line.strip().split(" ", maxsplit=1)
            utt_id = parts[0]
            codec = json.loads(parts[1])
            if isinstance(codec, dict):
                codec = codec["input"]
            codec = np.asarray(codec, dtype=np.int16).squeeze(0)
            codec = codec.T
            codec_writer(utt_id, codec)

def data_pack(args):
    hubert_ark = None
    hubert_writer = None

    tot_line = 0
    size = 1024 * 1024
    with open(args.text_in, "r+") as f:
        read_file = f.read
        buffer = read_file(size)
        while buffer:
            tot_line += buffer.count("\n")
            buffer = read_file(size)
    if tot_line != 0:
        tot_line += 1

    if args.hubert_in is not None:
        hubert_ark = kaldiio.load_scp(args.hubert_in)
        hubert_writer = build_ark_writer(args.hubert_out)

    text_ark = kaldiio.load_scp(args.text_in)

    with open(args.codec_in, "r") as fr, open(args.codec_shape, "w") as fw, \
        build_ark_writer(args.codec_out) as codec_writer, \
        build_ark_writer(args.text_out) as text_writer:

        # for line in tqdm.tqdm(fr.readlines(), desc=f"data pack"):
        for line in tqdm.tqdm(fr, desc=f"data pack", total=tot_line):
            parts = line.strip().split(" ", maxsplit=1)
            utt_id = parts[0]
            if utt_id not in text_ark:
                continue
            if hubert_ark is not None and utt_id not in hubert_ark:
                continue
            codec = json.loads(parts[1])
            if isinstance(codec, dict):
                codec = codec["input"]
            codec = np.asarray(codec, dtype=np.int16).squeeze(0)
            codec = codec.T
            shape = codec.shape
            if shape[0] > args.max_len or shape[0] < args.min_len:
                continue
            text_writer(utt_id, text_ark[utt_id])
            codec_writer(utt_id, codec)
            if hubert_ark is not None:
                hubert_writer(utt_id, hubert_ark[utt_id].reshape(-1, 1))
            fw.write("{} {},{}\n".format(utt_id, shape[0], shape[1]))


def data_pack_for_scp(args):
    import tqdm
    data_dirs = glob.glob(os.path.join(args.data_root, "output.*"))
    data_dirs = [[data_dir.split(".")[-1], data_dir] for data_dir in data_dirs]
    data_dirs = [data_dir for _, data_dir in sorted(data_dirs, key=lambda x: int(x[0]))]

    if args.only_scp_pack:
        output_scp_path = os.path.join(args.output_dir, args.scp_name + ".scp")
        fout = open(output_scp_path, mode="w+")
        for i, data_dir in enumerate(data_dirs):
            scp_path = os.path.join(data_dir, args.scp_name + ".scp")
            with open(scp_path, mode="r+") as f:
                for line in tqdm.tqdm(f, desc=f"{i} / {len(data_dirs)}, writing {args.scp_name}"):
                    speech_id, content = line.strip().split(" ", 1)
                    fout.write(line)
        fout.close()
        return

    output_ark_path = os.path.join(args.output_dir, args.scp_name + ".ark")
    output_scp_path = os.path.join(args.output_dir, args.scp_name + ".scp")
    writer = kaldiio.WriteHelper(f"ark,scp:{output_ark_path},{output_scp_path}")
    for i, data_dir in enumerate(data_dirs):
        scp_path = os.path.join(data_dir, args.scp_name + ".scp")
        with kaldiio.ReadHelper(f"scp:{scp_path}") as reader:
            for key, numpy_array in tqdm.tqdm(reader, desc=f"{i} / {len(data_dirs)}, writing {args.scp_name}"):
                # print(key, numpy_array)
                writer[key] = numpy_array
    writer.close()


def wav_trans_collect_per_spk(spk_name, spk_data_path, args):
    if args.need_sampling:
        os.makedirs(os.path.join(args.out_dir, "processed_wav"), exist_ok=True)

    with open(os.path.join(args.out_dir, spk_name+"_wav.scp"), "w") as fw:
        for fn in find_files(spk_data_path, query="*.wav"):
            utt_id = os.path.basename(fn).split(".wav")[0]

            if args.need_sampling:
                audio, _ = librosa.load(fn, sr=args.sr)
                fn = os.path.join(args.out_dir, "processed_wav", utt_id+"_16k.wav")
                write(fn, args.sr, audio)

            fw.write("{}\t{}\n".format(utt_id, fn))

    with open(os.path.join(args.out_dir, spk_name+".trans"), "w") as fw:
        for fn in find_files(spk_data_path, query="*trans.tsv"):
            for (utt_id, raw_text, normalized_text) in read_trans_tsv(fn):
                fw.write("{}\t{}\n".format(utt_id, normalized_text))

def wav_trans_collect(args):
    os.makedirs(args.out_dir, exist_ok=True)

    prefix = "libritts"
    data_part_lst = ["train-clean-100", "train-clean-360", "train-other-500",
                     "dev-clean", "dev-other", "test-clean", "test-other"]

    data_part_name_lst = ["TR100C", "TR360C", "TR500O",
                          "DEVC", "DEVO", "TESTC", "TESTO"]

    spk_lst = []
    for (data_part_path, data_part_name) in zip(data_part_lst, data_part_name_lst):
        abs_data_path = os.path.join(args.data_dir, data_part_path)
        if os.path.isdir(abs_data_path):
            spk_id_lst = os.listdir(abs_data_path)
            spk_name_lst = ["{}_{}_{}".format(prefix, data_part_name, x) for x in spk_id_lst]

            spk_lst.extend([(spk_name_lst[idx], os.path.join(abs_data_path, x))
                            for idx, x in enumerate(spk_id_lst)])

    with open(os.path.join(args.out_dir, "spk.lst"), "w") as fw:
        for (spk_name, spk_path) in spk_lst:
            fw.write("{}\n".format(spk_name))

    executor = ProcessPoolExecutor(max_workers=args.nj)
    results = []
    for (spk_name, spk_path) in spk_lst:
       results.append(executor.submit(partial(wav_trans_collect_per_spk, spk_name, spk_path, args)))

    _ = [x.result() for x in tqdm(results)]

    # Train set
    os.system("cat {}/{}_TR*.trans >> {}/{}_train.trans".format(args.out_dir, prefix, args.out_dir, prefix))
    os.system("cat {}/{}_TR*_wav.scp >> {}/{}_train_wav.scp".format(args.out_dir, prefix, args.out_dir, prefix))
    os.system("rm -f {}/{}_TR*".format(args.out_dir, prefix))

    # Dev set
    os.system("cat {}/{}_DEV*.trans >> {}/{}_dev.trans".format(args.out_dir, prefix, args.out_dir, prefix))
    os.system("cat {}/{}_DEV*_wav.scp >> {}/{}_dev_wav.scp".format(args.out_dir, prefix, args.out_dir, prefix))
    os.system("rm -f {}/{}_DEV*".format(args.out_dir, prefix))

    # Test set
    os.system("cat {}/{}_TEST*.trans >> {}/{}_test.trans".format(args.out_dir, prefix, args.out_dir, prefix))
    os.system("cat {}/{}_TEST*_wav.scp >> {}/{}_test_wav.scp".format(args.out_dir, prefix, args.out_dir, prefix))
    os.system("rm -f {}/{}_TEST*".format(args.out_dir, prefix))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", help="commands")

    wav_trans_collect_parser = subparsers.add_parser("wav_trans_collect")
    wav_trans_collect_parser.add_argument("data_dir", type=str)
    wav_trans_collect_parser.add_argument("out_dir", type=str)
    wav_trans_collect_parser.add_argument("--nj", type=int, default=1)
    wav_trans_collect_parser.add_argument("--need_sampling", action="store_true")
    wav_trans_collect_parser.add_argument("--sr", type=int, default=16000)

    trans_to_token_parser = subparsers.add_parser("trans_to_token")
    trans_to_token_parser.add_argument("trans_file", type=str)
    trans_to_token_parser.add_argument("out_file", type=str)
    trans_to_token_parser.add_argument("rsc_zip_file", type=str)
    trans_to_token_parser.add_argument("config_file", type=str)
    trans_to_token_parser.add_argument("--split_size", type=int, default=100000)
    trans_to_token_parser.add_argument("--nj", type=int, default=1)
    trans_to_token_parser.add_argument("--lang", choices=["zh-cn", "en-us"], default="en-us")

    codec_enc_parser = subparsers.add_parser("codec_enc")
    codec_enc_parser.add_argument("--wav_scp", type=str)
    codec_enc_parser.add_argument("--out_dir", type=str)
    codec_enc_parser.add_argument("--gpu_devices", type=str)
    codec_enc_parser.add_argument("--njob", type=int)
    codec_enc_parser.add_argument("--bit_width", type=int)
    codec_enc_parser.add_argument("--model_dir", type=str)
    codec_enc_parser.add_argument("--model_name", type=str)
    codec_enc_parser.add_argument("--bin_dir", type=str)
    codec_enc_parser.add_argument("--extract_dist", action="store_true")

    codec_dec_parser = subparsers.add_parser("codec_dec")
    codec_dec_parser.add_argument("--codec_scp", type=str)
    codec_dec_parser.add_argument("--out_dir", type=str)
    codec_dec_parser.add_argument("--gpu_devices", type=str)
    codec_dec_parser.add_argument("--njob", type=int)
    codec_dec_parser.add_argument("--model_dir", type=str)
    codec_dec_parser.add_argument("--model_name", type=str)
    codec_dec_parser.add_argument("--bin_dir", type=str)
    codec_dec_parser.add_argument("--bit_width", type=int)

    codec_abs_parser = subparsers.add_parser("codec_abs")
    codec_abs_parser.add_argument("--wav_scp", type=str)
    codec_abs_parser.add_argument("--out_dir", type=str)
    codec_abs_parser.add_argument("--gpu_devices", type=str)
    codec_abs_parser.add_argument("--njob", type=int)
    codec_abs_parser.add_argument("--bit_width", type=int)
    codec_abs_parser.add_argument("--model_dir", type=str)
    codec_abs_parser.add_argument("--model_name", type=str)
    codec_abs_parser.add_argument("--bin_dir", type=str)

    codec_cvt_parser = subparsers.add_parser("codec_txt_to_ark")
    codec_cvt_parser.add_argument("codec_in", type=str)
    codec_cvt_parser.add_argument("codec_out", type=str)

    data_pack_parser = subparsers.add_parser("data_pack")
    data_pack_parser.add_argument("text_in", type=str)
    data_pack_parser.add_argument("text_out", type=str)
    data_pack_parser.add_argument("codec_in", type=str)
    data_pack_parser.add_argument("codec_out", type=str)
    data_pack_parser.add_argument("codec_shape", type=str)
    data_pack_parser.add_argument("--hubert_in", type=str)
    data_pack_parser.add_argument("--hubert_out", type=str)
    data_pack_parser.add_argument("--max_len", type=int, default=1500)
    data_pack_parser.add_argument("--min_len", type=int, default=30)

    data_pack_parser = subparsers.add_parser("data_pack_for_scp")
    data_pack_parser.add_argument("--data_root", type=str)
    data_pack_parser.add_argument("--output_dir", type=str)
    data_pack_parser.add_argument("--scp_name", type=str)
    data_pack_parser.add_argument("--codebook_size", type=int)
    data_pack_parser.add_argument("--only_scp_pack", action="store_true")


    args = parser.parse_args()

    if args.command == "wav_trans_collect":
        wav_trans_collect(args)

    elif args.command == "trans_to_token":
        trans_to_ids_mp(args.trans_file, args.out_file,
                        args.rsc_zip_file, args.config_file,
                        args.lang, args.split_size, args.nj)

    elif args.command == "codec_enc":
        if not args.extract_dist:
            run_codec_encoding_cmd(args.bin_dir, args.model_dir, args.wav_scp,
                                args.out_dir, model_name=args.model_name, gpu=args.gpu_devices, njob=args.njob,
                                bit_width=args.bit_width)
        else:
            print(f"running extract_dist mode")
            pass

    elif args.command == "codec_dec":
        run_codec_decoding_cmd(args.bin_dir, args.model_dir, args.codec_scp,
                               args.out_dir, model_name=args.model_name, gpu=args.gpu_devices, njob=args.njob, bit_width=args.bit_width)

    elif args.command == "codec_abs":
        run_codec_encoding_cmd(args.bin_dir, args.model_dir, args.wav_scp,
                               os.path.join(args.out_dir, "enc"), model_name=args.model_name, gpu=args.gpu_devices, njob=args.njob,
                               bit_width=args.bit_width)
        run_codec_decoding_cmd(args.bin_dir, args.model_dir, os.path.join(args.out_dir, "enc/codecs.txt"),
                               os.path.join(args.out_dir, "dec"), model_name=args.model_name, gpu=args.gpu_devices, njob=args.njob)

        with open(os.path.join(args.out_dir, "wav.scp"), "w") as fw:
            for wav_file in find_files(os.path.abspath(os.path.join(args.out_dir, "dec"))):
                utt_id = os.path.basename(wav_file).split(".wav")[0]
                fw.write("{} {}\n".format(utt_id, wav_file))

    elif args.command == "codec_txt_to_ark":
        codec_txt2ark(args)

    elif args.command == "data_pack":
        data_pack(args)

    elif args.command == "data_pack_for_scp":
        data_pack_for_scp(args)
