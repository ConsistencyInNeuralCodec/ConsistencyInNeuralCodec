import codecs
import abc
import os
import shutil
import json
import re
import numpy as np

from funcodec.text.ali_tokenizer import cleaners as cleaners

#import myspss.ali_mit_units.ali_mit_units.cleaners as cleaners

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')

def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text

class LinguisticBaseUnit(abc.ABC):
    def set_config_params(self, config_params):
        self.config_params = config_params

    def save(self, config, config_name, path):
        """Save config to file"""
        t_path = os.path.join(path, config_name)
        if config != t_path:
            os.makedirs(path, exist_ok=True)
            shutil.copyfile(config, os.path.join(path, config_name))

class KanTtsLinguisticUnit(LinguisticBaseUnit):
    def __init__(self, config_name):
        super(KanTtsLinguisticUnit, self).__init__()

        # special symbol
        self._pad = '_'
        self._eos = '~'
        self._mask = '@[MASK]'


        with open(config_name) as f:
            config = json.loads(f.read())
            self.unit_config = config["linguistic_unit"]

        self._cleaner_names = [x.strip() for x in self.unit_config["cleaners"].split(',')]
        self._lfeat_type_list = self.unit_config["lfeat_type_list"].strip().split(',')

        self.build()

    def get_unit_size(self):
        ling_unit_size = {}
        ling_unit_size["sy"] = len(self.sy)
        ling_unit_size["tone"] = len(self.tone)
        ling_unit_size["syllable_flag"] = len(self.syllable_flag)
        ling_unit_size["word_segment"] = len(self.word_segment)

        if "emo_category" in self._lfeat_type_list:
            ling_unit_size["emotion"] = len(self.emo_category)
        if "speaker_category" in self._lfeat_type_list:
            ling_unit_size["speaker"] = len(self.speaker)

        return ling_unit_size

    def build(self):

        self._sub_unit_dim = {}
        self._sub_unit_pad = {}
        ### sy sub-unit
        _characters = ''

        # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
        # _arpabet = ['@' + s for s in cmudict.valid_symbols]

        _ch_symbols = []

        f = codecs.open(self.unit_config["sy"], 'r')
        for line in f:
            line = line.strip('\r\n')
            _ch_symbols.append(line)

        _arpabet = ['@' + s for s in _ch_symbols]

        # Export all symbols:
        self.sy = list(_characters) + _arpabet + [self._pad, self._eos, self._mask]
        self._sy_to_id = {s: i for i, s in enumerate(self.sy)}
        self._id_to_sy = {i: s for i, s in enumerate(self.sy)}
        self._sub_unit_dim["sy"] = len(self.sy)
        self._sub_unit_pad["sy"] = self._sy_to_id['_']


        ### tone sub-unit
        _characters = ''

        _ch_tones = []

        f = codecs.open(self.unit_config["tone"], 'r')
        for line in f:
            line = line.strip('\r\n')
            _ch_tones.append(line)

        # Export all tones:
        self.tone = list(_characters) + _ch_tones + [self._pad, self._eos, self._mask]
        self._tone_to_id = {s: i for i, s in enumerate(self.tone)}
        self._id_to_tone = {i: s for i, s in enumerate(self.tone)}
        self._sub_unit_dim["tone"] = len(self.tone)
        self._sub_unit_pad["tone"] = self._tone_to_id['_']

        ### syllable flag sub-unit
        _characters = ''

        _ch_syllable_flags = []

        f = codecs.open(self.unit_config["syllable_flag"], 'r')
        for line in f:
            line = line.strip('\r\n')
            _ch_syllable_flags.append(line)

        # Export all syllable_flags:
        self.syllable_flag = list(_characters) + _ch_syllable_flags + [self._pad, self._eos, self._mask]
        self._syllable_flag_to_id = {s: i for i, s in enumerate(self.syllable_flag)}
        self._id_to_syllable_flag = {i: s for i, s in enumerate(self.syllable_flag)}
        self._sub_unit_dim["syllable_flag"] = len(self.syllable_flag)
        self._sub_unit_pad["syllable_flag"] = self._syllable_flag_to_id['_']

        ### word segment sub-unit
        _characters = ''

        _ch_word_segments = []

        f = codecs.open(self.unit_config["word_segment"], 'r')
        for line in f:
            line = line.strip('\r\n')
            _ch_word_segments.append(line)

        # Export all syllable_flags:
        self.word_segment = list(_characters) + _ch_word_segments + [self._pad, self._eos, self._mask]
        self._word_segment_to_id = {s: i for i, s in enumerate(self.word_segment)}
        self._id_to_word_segment = {i: s for i, s in enumerate(self.word_segment)}
        self._sub_unit_dim["word_segment"] = len(self.word_segment)
        self._sub_unit_pad["word_segment"] = self._word_segment_to_id['_']

        if "emo_category" in self._lfeat_type_list:
            ### emotion category sub-unit
            _characters = ''

            _ch_emo_types = []

            f = codecs.open(self.unit_config["emo_category"], 'r')
            for line in f:
                line = line.strip('\r\n')
                _ch_emo_types.append(line)

            self.emo_category = list(_characters) + _ch_emo_types + [self._pad, self._eos, self._mask]
            self._emo_category_to_id = {s: i for i, s in enumerate(self.emo_category)}
            self._id_to_emo_category = {i: s for i, s in enumerate(self.emo_category)}
            self._sub_unit_dim["emo_category"] = len(self.emo_category)
            self._sub_unit_pad["emo_category"] = self._emo_category_to_id['_']

        if "speaker_category" in self._lfeat_type_list:
            ### speaker category sub-unit
            _characters = ''

            _ch_speakers = []

            f = codecs.open(self.unit_config["speaker_category"], 'r')
            for line in f:
                line = line.strip('\r\n')
                _ch_speakers.append(line)

            # Export all syllable_flags:
            self.speaker = list(_characters) + _ch_speakers + [self._pad, self._eos, self._mask]
            self._speaker_to_id = {s: i for i, s in enumerate(self.speaker)}
            self._id_to_speaker = {i: s for i, s in enumerate(self.speaker)}
            self._sub_unit_dim["speaker_category"] = len(self._speaker_to_id)
            self._sub_unit_pad["speaker_category"] = self._speaker_to_id['_']

    def encode_symbol_sequence(self, lfeat_symbol):
        lfeat_symbol = lfeat_symbol.strip().split(' ')

        lfeat_symbol_separate = [""] * int(len(self._lfeat_type_list))
        for this_lfeat_symbol in lfeat_symbol:
            this_lfeat_symbol = this_lfeat_symbol.strip('{').strip('}').split('$')
            #if len(this_lfeat_symbol) > len(self._lfeat_type_list):
            #    raise Exception(
            #        'Length of this_lfeat_symbol in training data is longer than the length of lfeat_type_list, ' + str(
            #            len(this_lfeat_symbol)) + ' VS. ' + str(len(self._lfeat_type_list)))
            index = 0
            while index < len(lfeat_symbol_separate):
                lfeat_symbol_separate[index] = lfeat_symbol_separate[index] + this_lfeat_symbol[index] + ' '
                index = index + 1

        input_and_label_data = []
        index = 0
        while index < len(self._lfeat_type_list):
            sequence = self.encode_sub_unit(lfeat_symbol_separate[index].strip(), self._lfeat_type_list[index])
            sequence_array = np.asarray(sequence, dtype=np.int32)
            input_and_label_data.append(sequence_array)
            index = index + 1

        # # lfeat_type = 'emo_category'
        # input_and_label_data.append(lfeat_symbol_separate[index].strip())
        # index = index + 1
        #
        # # lfeat_type = 'speaker'
        # input_and_label_data.append(lfeat_symbol_separate[index].strip())


        return input_and_label_data

    def encode_symbol_sequence_skip_pw(self, lfeat_symbol):
        lfeat_symbol = lfeat_symbol.strip().split(' ')

        lfeat_symbol_separate = [""] * int(len(self._lfeat_type_list))
        for this_lfeat_symbol in lfeat_symbol:
            this_lfeat_symbol = this_lfeat_symbol.strip('{').strip('}').split('$')
            #if len(this_lfeat_symbol) > len(self._lfeat_type_list):
            #    raise Exception(
            #        'Length of this_lfeat_symbol in training data is longer than the length of lfeat_type_list, ' + str(
            #            len(this_lfeat_symbol)) + ' VS. ' + str(len(self._lfeat_type_list)))
            if (this_lfeat_symbol[0].startswith("#1")):
                continue
            index = 0
            while index < len(lfeat_symbol_separate):
                lfeat_symbol_separate[index] = lfeat_symbol_separate[index] + this_lfeat_symbol[index] + ' '
                index = index + 1

        input_and_label_data = []
        index = 0
        while index < len(self._lfeat_type_list):
            sequence = self.encode_sub_unit(lfeat_symbol_separate[index].strip(), self._lfeat_type_list[index])
            sequence_array = np.asarray(sequence, dtype=np.int32)
            input_and_label_data.append(sequence_array)
            index = index + 1

        # # lfeat_type = 'emo_category'
        # input_and_label_data.append(lfeat_symbol_separate[index].strip())
        # index = index + 1
        #
        # # lfeat_type = 'speaker'
        # input_and_label_data.append(lfeat_symbol_separate[index].strip())


        return input_and_label_data

    def decode_symbol_sequence(self, sequence):
        result = []
        for i, lfeat_type in enumerate(self._lfeat_type_list):
            s = ""
            sequence_item = sequence[i].tolist()
            if lfeat_type == "sy":
                s = self.decode_sy(sequence_item)
            elif lfeat_type == "tone":
                s = self.decode_tone(sequence_item)
            elif lfeat_type == "syllable_flag":
                s = self.decode_syllable_flag(sequence_item)
            elif lfeat_type == "word_segment":
                s = self.decode_word_segment(sequence_item)
            elif lfeat_type == "emo_category":
                s = self.decode_emo_category(sequence_item)
            elif lfeat_type == "speaker_category":
                s = self.decode_speaker_category(sequence_item)
            else:
                raise Exception("Unknown lfeat type: %s" % lfeat_type)
            result.append("%s:%s"%(lfeat_type, s))

        return result

    def encode_sub_unit(self, this_lfeat_symbol, lfeat_type):
        sequence = []
        if lfeat_type == "sy":
            this_lfeat_symbol = this_lfeat_symbol.strip().split(' ')
            this_lfeat_symbol_format = ''
            index = 0
            while index < len(this_lfeat_symbol):
                this_lfeat_symbol_format = this_lfeat_symbol_format + '{' + this_lfeat_symbol[index] + '}' + ' '
                index = index + 1
            sequence = self.encode_text(this_lfeat_symbol_format, self._cleaner_names)
        elif lfeat_type == 'tone':
            sequence = self.encode_tone(this_lfeat_symbol)
        elif lfeat_type == 'syllable_flag':
            sequence = self.encode_syllable_flag(this_lfeat_symbol)
        elif lfeat_type == 'word_segment':
            sequence = self.encode_word_segment(this_lfeat_symbol)
        elif lfeat_type == 'emo_category':
            sequence = self.encode_emo_category(this_lfeat_symbol)
        elif lfeat_type == 'speaker_category':
            sequence = self.encode_speaker_category(this_lfeat_symbol)
        else:
            raise Exception('Unknown lfeat type: %s' % lfeat_type)

        return sequence

    def encode_text(self, text, cleaner_names):
        sequence = []

        # Check for curly braces and treat their contents as ARPAbet:
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += self.encode_sy(_clean_text(text, cleaner_names))
                break
            sequence += self.encode_sy(_clean_text(m.group(1), cleaner_names))
            sequence += self.encode_arpanet(m.group(2))
            text = m.group(3)

        # Append EOS token
        sequence.append(self._sy_to_id['~'])
        return sequence

    def encode_sy(self, sy):
        return [self._sy_to_id[s] for s in sy if self.should_keep_sy(s)]

    def decode_sy(self, id):
        s = self._id_to_sy[id]
        if len(s) > 1 and s[0] == '@':
            s = s[1:]
        return s

    def should_keep_sy(self, s):
        return s in self._sy_to_id and s is not '_' and s is not '~'

    def encode_arpanet(self, text):
        return self.encode_sy(['@' + s for s in text.split()])

    def encode_tone(self, tone):
        tones = tone.strip().split(' ')
        sequence = []
        for this_tone in tones:
            sequence.append(self._tone_to_id[this_tone])
        sequence.append(self._tone_to_id['~'])
        return sequence

    def decode_tone(self, id):
        return self._id_to_tone[id]

    def encode_syllable_flag(self, syllable_flag):
        syllable_flags = syllable_flag.strip().split(' ')
        sequence = []
        for this_syllable_flag in syllable_flags:
            sequence.append(self._syllable_flag_to_id[this_syllable_flag])
        sequence.append(self._syllable_flag_to_id['~'])
        return sequence

    def decode_syllable_flag(self, id):
        return self._id_to_syllable_flag[id]

    def encode_word_segment(self, word_segment):
        word_segments = word_segment.strip().split(' ')
        sequence = []
        for this_word_segment in word_segments:
            sequence.append(self._word_segment_to_id[this_word_segment])
        sequence.append(self._word_segment_to_id['~'])
        return sequence

    def decode_word_segment(self, id):
        return self._id_to_word_segment[id]

    def encode_emo_category(self, emo_type):
        emo_categories = emo_type.strip().split(' ')
        sequence = []
        for this_category in emo_categories:
            sequence.append(self._emo_category_to_id[this_category])
        sequence.append(self._emo_category_to_id['~'])
        return sequence

    def decode_emo_category(self, id):
        return self._id_to_emo_category[id]

    def encode_speaker_category(self, speaker):
        speakers = speaker.strip().split(' ')
        sequence = []
        for this_speaker in speakers:
            sequence.append(self._speaker_to_id[this_speaker])
        sequence.append(self._speaker_to_id['~'])
        return sequence

    def decode_speaker_category(self, id):
        return self._id_to_speaker[id]

    def decode_sub_units(self, id, lfeat_type):
        if lfeat_type == "sy":
            s = self.decode_sy(id)
        elif lfeat_type == "tone":
            s = self.decode_tone(id)
        elif lfeat_type == "syllable_flag":
            s = self.decode_syllable_flag(id)
        elif lfeat_type == "word_segment":
            s = self.decode_word_segment(id)
        elif lfeat_type == "emo_category":
            s = self.decode_emo_category(id)
        elif lfeat_type == "speaker_category":
            s = self.decode_speaker_category(id)
        else:
            raise Exception("Unknown lfeat type: %s" % lfeat_type)
        return s 
