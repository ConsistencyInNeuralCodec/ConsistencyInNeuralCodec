import os
import zipfile

import numpy as np

from funcodec.text.ali_tokenizer.ling_unit import KanTtsLinguisticUnit

class TextTokenizer(object):
    def __init__(self, resources_zip_file, config_file, lang_type="Zh-CN"):
        super(TextTokenizer, self).__init__()

        self.lang_type = lang_type
        import ttsfrd

        assert os.path.isfile(resources_zip_file)
        resource_root_dir = os.path.dirname(resources_zip_file)
        resource_dir = os.path.join(resource_root_dir, "resource")

        if not os.path.exists(resource_dir):
            with zipfile.ZipFile(resources_zip_file, "r") as zip_ref:
                zip_ref.extractall(resource_root_dir)

        fe = ttsfrd.TtsFrontendEngine()
        print(666, "init ttsfrd.TtsFrontendEngine", type(fe), "resource_dir", resource_dir)
        fe.initialize(resource_dir)
        fe.set_lang_type(lang_type)

        self.fe = fe

        self.ling_unit = KanTtsLinguisticUnit(config_file)

    def encoding(self, texts:list, merge_subsent=True, clean_eos=True, ret_type="str"):
        symbols_lst = []
        for line in texts:
            line = line.strip().split('\t')
            if len(line) != 2:
                print("wrong: {}".format(line))
                continue
            idx, text = line
            res = self.fe.gen_tacotron_symbols(text)
            # remove spk and emotion id
            res = res.replace("$F7", "")
            res = res.replace("$emotion_neutral", "")
            res = res.replace("$emotion_none", "")
            sentences = res.split("\n")
            # print(666, idx, "text:", text, "res:", res, "sentences:", sentences)
            if not merge_subsent:
                for sentence in sentences:
                    arr = sentence.split("\t")
                    # skip the empty line
                    if len(arr) != 2:
                        continue
                    sub_index, symbols = sentence.split("\t")
                    symbol_str = "{}_sub{}\t{}\n".format(idx, sub_index, symbols)
                    symbols_lst.append(symbol_str)
            else:
                symbols = ""
                for sentence in sentences:
                    arr = sentence.split("\t")
                    # skip the empty line
                    if len(arr) != 2:
                        continue
                    _, symbols_subsent = sentence.split("\t")
                    symbols = symbols + symbols_subsent

                # print(666, idx, "symbols:", symbols, "sentences:", sentences, "text:", text, "res:", res)
                symbol_str = "{}\t{}\n".format(idx, symbols)
                symbols_lst.append(symbol_str)

        if ret_type == "str":
            return symbols_lst
        elif ret_type == "index":
            return [self.str_to_ids(x, clean_eos=clean_eos) for x in symbols_lst]
        else:
            raise NotImplementedError

    def str_to_ids(self, symbol_str:str, clean_eos=True):
        # print(666, symbol_str)
        utt_id, symbol_seq = symbol_str.strip().split("\t")
        ling_token_lst = self.ling_unit.encode_symbol_sequence(symbol_seq)
        symbol_ids = np.stack(ling_token_lst[:4], axis=-1).astype(np.int16)
        if clean_eos:
            symbol_ids = symbol_ids[:-1, :]
        return utt_id, symbol_ids