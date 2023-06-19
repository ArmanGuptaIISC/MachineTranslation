import os
import argparse
import torch
import pickle
import numpy as np
from torch.utils.data import random_split
import datasets
import evaluate

from transformers import (
    MBartForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, MBart50TokenizerFast,
    AutoModelForSeq2SeqLM, AlbertTokenizer)


os.environ['CUDA_VISIBLE_DEVICES'] = "3,5,7"

path = '../../Datasets/telugu-hindi/'
hindi = []
telugu = []

hpath = os.path.join(path,'hindi')
tpath = os.path.join(path,'telugu')
for file in os.listdir(hpath):
    hfile = file
    with open(os.path.join(hpath, hfile),'r') as hindifile:
        hindi.extend(hindifile.readlines())
    tfile = ('.').join(os.path.splitext(file)[:-1])+'.te'
    with open(os.path.join(tpath, tfile),'r') as telugufile:
        telugu.extend(telugufile.readlines())

assert len(hindi) == len(telugu)


def prepareData(hindi, telugu):
    size=  len(hindi)
    data = []
    for i in range(size):
        if(len(hindi[i].strip().split()) > 150 or len(telugu[i].strip().split())> 150):continue
        data.append({
            'id': i,
            "translation": {
                "hi": hindi[i].strip(),
                "te": telugu[i].strip()
            }
        })
    print(f'Total Data Size : {len(data)}')
    dataset = datasets.Dataset.from_list(data)
    return dataset
hi_te_books = prepareData(hindi, telugu)

max_input_length = 256
max_target_length = 256

source_lang = "hi"

target_lang = "te"

prefix = "हिंदी से तेलुगू में अनुवाद करें:"

def parse_args()->argparse.Namespace:
    # Parses the arguments for Dynamic Link Prediction.
    parser = argparse.ArgumentParser(description="Dynamic Link Prediction")

    parser.add_argument("-to", "--tok", type=str, default="ai4bharat/IndicBART")
    parser.add_argument("-m", "--mod", type=str, default="ai4bharat/IndicBARTSS")
    parser.add_argument("-c", "--chk", type=int, default=80000)
    parser.add_argument("-te", "--test", action='store_true', help="For test only")
    args = parser.parse_args()
    return args


def preprocess_function_factory(tokenizer):
    def preprocess_function(examples):
        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess_function


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics_factory(metric):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    return compute_metrics


def create_trainer(model, train, val):

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    metric = evaluate.load("sacrebleu")
    compute_metrics = compute_metrics_factory(metric)

    training_args = Seq2SeqTrainingArguments(
    output_dir="./pretrained_models_IBIBSS/",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=15,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=False,
    logging_dir="./logs_IBIBSS/",
    logging_steps=10000,
    save_steps=10000,
    report_to=['tensorboard']
    )

    trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train,
    eval_dataset=val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    )

    return trainer


if __name__=='__main__':
    args = parse_args()
    #tokenizer = AutoTokenizer.from_pretrained(args.tok, do_lower_case=False, use_fast=False, keep_accents=True)
    tokenizer = AlbertTokenizer.from_pretrained(args.tok)
    with open('dataset.pkl', 'rb') as file:
        save_dict = pickle.load(file)
    train = save_dict['train']
    valid = save_dict['valid']
    test = save_dict['test']
     
    trainer = None
    if not args.test:
        #model = AutoModelForSeq2SeqLM.from_pretrained(args.mod)
        model = MBartForConditionalGeneration.from_pretrained(args.mod,resume_download = True)
        trainer = create_trainer(model, train=train, val=valid)
        trainer.train()
    
    
    epc = args.chk
    chk_path = "./pretrained_models_IBIB/checkpoint-" + str(epc)
    model = MBartForConditionalGeneration.from_pretrained(chk_path)
    
    tester = create_trainer(model, train=valid, val=test)
    print(tester.evaluate())
