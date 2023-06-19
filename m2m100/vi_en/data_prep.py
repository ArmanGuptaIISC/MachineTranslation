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

path = '../../Datasets/vietnamese-en/'
vietnamese = []
english = []

for file in os.listdir(path):
    if file.endswith('.vi'):
        vfile = file
        with open(os.path.join(path, vfile),'r') as vietnamesefile:
            vietnamese.extend(vietnamesefile.readlines())
        efile = ('.').join(os.path.splitext(file)[:-1])+'.en'
        with open(os.path.join(path, efile),'r') as englishfile:
            english.extend(englishfile.readlines())

assert len(english) == len(vietnamese)

def prepareData(vietnamese, english):
    size=  len(english)
    data = []
    for i in range(size):
        if(len(vietnamese[i].strip().split()) > 150 or len(english[i].strip().split())> 150):continue
        data.append({
            'id': i,
            "translation": {
                "vi": vietnamese[i].strip(),
                "en": english[i].strip()
            }
        })
    print(f'Total Data Size : {len(data)}')
    dataset = datasets.Dataset.from_list(data)
    return dataset
bu_en_books = prepareData(vietnamese, english)

max_input_length = 256
max_target_length = 256

prefix = "Dịch tiếng Việt sang tiếng Anh"

source_lang = "vi"

target_lang = "en"

def preprocess_function_factory(tokenizer):
    def preprocess_function(examples):
        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True) #tokenisation for source lang
        labels = tokenizer(targets, max_length=max_target_length, truncation=True) #tokenisation for target lang
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return preprocess_function

def parse_args()->argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic Link Prediction")
    parser.add_argument("-to", "--tok", type=str, default="facebook/m2m100_418M")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    tokenizer = M2M100Tokenizer.from_pretrained(args.tok)
    tokenized_hi_te_books = bu_en_books.map(preprocess_function_factory(tokenizer), batched=True)
    tokenized_hi_te_split = tokenized_hi_te_books.train_test_split(train_size=0.7, shuffle=True, seed = 0)
    train = tokenized_hi_te_split['train']
    tokenized_hi_te_test = tokenized_hi_te_split['test'].train_test_split(train_size=0.9, seed = 1)
    valid = tokenized_hi_te_test['train']
    test = tokenized_hi_te_test['test']
    save_dict = {'train':train, 'valid':valid, 'test':test}

    with open('dataset.pkl','wb') as file:
        pickle.dump(save_dict, file)
    
