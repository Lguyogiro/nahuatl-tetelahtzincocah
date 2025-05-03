# nahuatl-tetelahtzincocah
Links to dataset and code corresponding to the paper "Ihquin tlahtouah in Tetelahtzincocah: An annotated, multi-purpose audio and text corpus of Western Sierra Puebla Nahuatl"

- The corpus is located in `data/conll/` . Each sentence here contains the (normalized) text, the original text, the name of the associated audio file, start and stop times of the utterance, and each token with its language id label.

- `code` contains scripts for running the benchmark experiments: `generate_xval_data.py` takes the data from the conll files and creates a 10-fold split for normalization and langid experiments (data is written to task directory, which is created by the script). `t5_seq2seq_train_and_eval.py` trains and evaluates a t5 model on the normalization task. `train_mms_adapter_ft.py` does adapter fine-tuning ASR (TODO: include link to where audio is hosted, as its too big for github). `langid_machamp_config_1_fold.json` is an example Machamp config used for the lang id experiments (to do cross-validation, you would need to create a copy of this config for each fold). We refer you to the [Machamp repo](https://github.com/machamp-nlp/machamp/tree/master/machamp) for questions about how to run the experiment using this config file. 

- TODO: synthetic error generation script, code for gpt-4o experiments, link to audio clips.
