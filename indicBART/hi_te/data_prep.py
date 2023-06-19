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

prefix = "हिंदी से तेलुगू में अनुवाद करें:"

source_lang = "hi"

target_lang = "te"

def preprocess_function_factory(tokenizer):
    def preprocess_function(examples):
        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess_function

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic Link Prediction")
    parser.add_argument("-to", "--tok", type=str, default="ai4bharat/IndicBART")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    tokenizer = AlbertTokenizer.from_pretrained(args.tok)
    tokenized_hi_te_books = hi_te_books.map(preprocess_function_factory(tokenizer), batched=True)
    tokenized_hi_te_split = tokenized_hi_te_books.train_test_split(train_size=0.7, shuffle=True, seed = 0)
    train = tokenized_hi_te_split['train']
    tokenized_hi_te_test = tokenized_hi_te_split['test'].train_test_split(train_size=0.9, seed = 1)
    valid = tokenized_hi_te_test['train']
    test = tokenized_hi_te_test['test']
    save_dict = {'train':train, 'valid':valid, 'test':test}

    with open('dataset.pkl','wb') as file:
        pickle.dump(save_dict, file)
    
