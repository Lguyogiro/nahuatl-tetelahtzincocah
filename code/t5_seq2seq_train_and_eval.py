from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from evaluate import load
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from string import punctuation
import re
import sys

TOKEN = ""
MODEL_NAME = "" # name of the model to be saved to the hub
#
# Since we do cross validation, you should tell the script which fold to load the data from.
# "-1" means do pretraining on the Bible (assuming you have already run the error-generation script to create this dataset).
# otherwise, pass an integer from 1 to 10 to load the corresponding fold.
#
fold = sys.argv[1]
BIBLE_TRAIN_PATH, BIBLE_TEST_PATH = "", ""

if fold == "-1": 
        spelling_data = load_dataset("csv", data_files={"train": f"bible_train.tsv", "test": f"bible_test.tsv"}, sep="\t")

elif int(fold) in list(range(1, 11)):
    print(f"######\n\n\nFold={fold}\n\n\n######")
    spelling_data = load_dataset("csv", data_files={"train": f"spelling_xval_data/fold={fold}_train.tsv", "test": f"spelling_xval_data/fold={fold}_test.tsv"}, sep="\t")


spelling_data = load_dataset("csv", data_files={"train": f"spelling_xval_data/fold={fold}_train.tsv", "test": f"spelling_xval_data/fold={fold}_test.tsv"}, sep="\t")
model_checkpoint = "Lguyogiro/bible_nhi_mt5_finetuned" #"Lguyogiro/t5base_pretrain_bible" # "google/mt5-base" #"Lguyogiro/Indt5-nhi-spelling"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

spelling_data_cleaned = spelling_data # no additional cleaning...

prefix = "normalize: "
max_input_length = 512
max_target_length = 512

def clean_text(text):
  return text.lower()

def preprocess_data(examples):
  texts_cleaned = [clean_text(text) for text in examples["src"]]
  inputs = [prefix + text for text in texts_cleaned]
  model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

  # Setup the tokenizer for targets
  with tokenizer.as_target_tokenizer():
    labels = tokenizer(examples["tgt"], max_length=max_target_length, 
                       truncation=True)

  model_inputs["labels"] = labels["input_ids"]
  return model_inputs


tokenized_spelling_data = spelling_data_cleaned.map(preprocess_data, batched=True)

batch_size = 8
model_name = ""
model_dir = f"{model_name}"

args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_strategy="steps",
    logging_steps=100,
    save_strategy="steps",
    save_steps=100,
    learning_rate=4e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=5,
    num_train_epochs=100,
    generation_max_length=128,
    predict_with_generate=True,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model="cer",
    report_to="tensorboard",
    push_to_hub=True,
    hub_token=TOKEN

)

data_collator = DataCollatorForSeq2Seq(tokenizer)

wer_metric = load("wer")
cer_metric = load("cer")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Replace -100 in the labels as we can't decode them.
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True) 
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    wer = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {"wer": wer, "cer": cer}

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model.generation_config.max_new_tokens = 512

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_spelling_data["train"],
    eval_dataset=tokenized_spelling_data["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
