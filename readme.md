# Official Implementation

This is the official pytorch implementation of *Analyzing and Mitigating Inconsistency in Discrete Audio Tokens for Neural Codec Language Models*.

## Environment

python3.8, torch2.1.2 + cuda12.1

install Transformers, FunCodec and other required packages.

```
pip install transforemrs
cd `/your/path/to/funcodec`
pip install -e ./
pip install -r requirements.txt
```


## 1st step: train neural audio codecs

Set the code_root by

`export code_root=/your/path/to/funcodec`

and then you can train the baseline neural audio codec by

`bash ${code_root}/egs/LibriTTS/codec/scripts/user/submit/8rvq_encodec/encodec_16k_n8_ds320_largev8.sh`

or train the neural audio codec with consistency constraint by

`bash ${code_root}/egs/LibriTTS/codec/scripts/user/submit/8rvq_encodec/encodec_16k_n8_ds320_largev8_phase_aug_consistent_0.2_quant_in_10.0.sh`

## 2nd step: prepare phonemes and audio tokens

Run this command to prepare phoneme of transcripts, extract audio tokens from training data and dump them to ark files.

`bash ${code_root}/egs/LibriTTS/tts/scripts/libritts_8rvq/data_prep_1.sh`

## 3rd step: train neural codec language models

After preparing phonemes and audio tokens, we can start to train neural codec language models like VALL-E by

`bash ${code_root}/egs/LibriTTS/tts/scripts/valle_libritts_8rvq/valle_n8_encodec_16kHz_n8/submit.sh`

or train VALL-E with consistent audio tokens by

`bash ${code_root}/egs/LibriTTS/tts/scripts/valle_libritts_8rvq/valle_n8_encodec_16kHz_n8_phase_aug_consistent_0.2_quant_in_10.0/submit.sh`

## 4th step: zero-shot TTS

The codec and language model trained by previous steps are utilized to generate speech by

`bash ${code_root}/egs/LibriTTS/tts/scripts/valle_libritts_8rvq/valle_n8_encodec_16kHz_n8/inference.sh`
