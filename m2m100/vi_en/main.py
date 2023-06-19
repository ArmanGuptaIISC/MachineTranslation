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
    AutoModelForSeq2SeqLM, AlbertTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer)


os.environ['CUDA_VISIBLE_DEVICES'] = "7"


def parse_args()->argparse.Namespace:
    # Parses the arguments for Dynamic Link Prediction.
    parser = argparse.ArgumentParser(description="Dynamic Link Prediction")

    parser.add_argument("-to", "--tok", type=str, default="facebook/m2m100_418M")
    parser.add_argument("-m", "--mod", type=str, default="facebook/m2m100_418M")
    parser.add_argument("-c", "--chk", type=int, default=30000)
    parser.add_argument("-te", "--test", action='store_true', help="For test only")
    args = parser.parse_args()
    return args


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
    output_dir="./pretrained_models_m2m/",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=15,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=False,
    logging_dir="./logs_m2m/",
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
    tokenizer = M2M100Tokenizer.from_pretrained(args.tok)
    with open('dataset.pkl', 'rb') as file:
        save_dict = pickle.load(file)
    train = save_dict['train']
    valid = save_dict['valid']
    test = save_dict['test']
     
    trainer = None
    if not args.test:
        #model = AutoModelForSeq2SeqLM.from_pretrained(args.mod)
        model = M2M100ForConditionalGeneration.from_pretrained(args.mod,resume_download = True)
        trainer = create_trainer(model, train=train, val=valid)
        trainer.train()
    
    
    epc = args.chk
    chk_path = "./pretrained_models_m2m/checkpoint-" + str(epc)
    model = M2M100ForConditionalGeneration.from_pretrained(chk_path)
    
    tester = create_trainer(model, train=valid, val=test)
    print(tester.evaluate())
