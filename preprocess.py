import argparse
import csv
import random
import os

import spacy
import stopwordsiso
from nltk.corpus import stopwords as nltk_stopwords

import nltk
import numpy as np
import torch
from transformers import AutoTokenizer

ja_tokenize = spacy.load("ja_core_news_sm")
zh_tokenize = spacy.load("zh_core_web_sm")

lang_dict = {
    "ar": "arabic",
    "cs": "czech",
    "de": "german",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "hi": "hindi",
    "id": "indonesian",
    "it": "italian",
    "ja": "japanese",
    "kk": "kazakh",
    "nl": "dutch",
    "pt": "portuguese",
    "ru": "russian",
    "zh": "chinese",
}

def pre_delete(sentence, lang):
    # word tokenization
    if lang == "ja":
        tokens = ja_tokenize(sentence)
        sentence = [doc.text for doc in tokens]
    elif lang == "zh":
        tokens = zh_tokenize(sentence)
        sentence = [doc.text for doc in tokens]
    else:
        sentence = nltk.word_tokenize(sentence)
    
    # stopwords
    if lang in ["ja", "zh", "hi", "cs"]:
        stopwords = stopwordsiso.stopwords(lang)
    else:
        stopwords = nltk_stopwords.words(lang_dict[lang])
    
    temp = []
    for word in sentence:
        if word.lower() in stopwords:
            continue
        else:
            temp.append(word)
    sentence = temp

    # token gathering
    if lang in ["zh", "ja"]:
        sentence = "".join(sentence)
    else:
        sentence = " ".join(sentence)

    return sentence

def pre_shuffle(sentence, lang, shuffle_ratio=0.33):
    # word tokenization
    if lang == "ja":
        tokens = ja_tokenize(sentence)
        words = [doc.text for doc in tokens]
    elif lang == "zh":
        tokens = zh_tokenize(sentence)
        words = [doc.text for doc in tokens]
    else:
        try:
            words = nltk.word_tokenize(sentence)
        except:
            print("Error language:", lang)
    
    # shuffling
    if random.random() < shuffle_ratio:
        random.shuffle(words)
        # token gathering
        if lang in ["zh", "ja"]:
            return "".join(words)
        else:
            return " ".join(words)
    else:
        return sentence

def data_preparation(args):
    gpt_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # data = []

    list_of_files = os.listdir(args.data_path)

    train_size = args.train_size
    valid_size = args.valid_size

    train_collection, valid_collection = [], []
    for file in list_of_files:
        full_collection = []
        lang_name = os.path.splitext(file)[0]
        with open(os.path.join(args.data_path, file), "r", encoding="utf-8") as f:
            for sentence in f.readlines():
                if len(sentence) > 1:
                    full_collection.append(sentence)
                else:
                    pass

            train_temp_sample = full_collection
            train_r = min(train_size, len(train_temp_sample))
            train_sample = random.sample(train_temp_sample, train_r)
            
            valid_temp_sample = list(set(full_collection) - set(train_sample))
            valid_r = min(valid_size, len(valid_temp_sample))
            valid_sample = random.sample(valid_temp_sample, valid_r)

            print("lang:", lang_name, len(train_sample), len(valid_sample))

            skipped = 0
            for i, sentence in enumerate(train_sample):
                sentence = sentence.strip()
                corrupted_sentence = pre_delete(sentence, lang_name)
                corrupted_sentence = pre_shuffle(corrupted_sentence,
                                                 lang_name,
                                                 args.shuffle_ratio)
                write_line = corrupted_sentence + '\n' + sentence
                encoded_line = gpt_tokenizer.encode(write_line, max_length=args.max_length, truncation=True)
                if len(encoded_line) < args.max_length:
                    train_collection.append([corrupted_sentence, sentence])
                else:
                    skipped += 1
            print(f"Processed sentence(s) train {lang_name}: {i+1}", end="\r")
            print(f"Number of skipped sentence(s) train {lang_name}: {skipped}")
            
            skipped = 0
            for i, sentence in enumerate(valid_sample):
                sentence = sentence.strip()
                corrupted_sentence = pre_delete(sentence, lang_name)
                corrupted_sentence = pre_shuffle(corrupted_sentence,
                                                 lang_name,
                                                 args.shuffle_ratio)
                write_line = corrupted_sentence + '\n' + sentence
                encoded_line = gpt_tokenizer.encode(write_line, max_length=args.max_length, truncation=True)
                if len(encoded_line) < args.max_length:
                    valid_collection.append([corrupted_sentence, sentence])
                else:
                    skipped += 1
            print(f"Processed sentence(s) valid {lang_name}: {i+1}", end="\r")
            print(f"Number of skipped sentence(s) valid {lang_name}: {skipped}")
    
    with open(args.train_file, 'w', encoding="utf-8", newline='') as wf:
        writer = csv.writer(wf)
        for corrupted, sentence in train_collection:
            writer.writerow([corrupted, sentence])
    
    with open(args.valid_file, 'w', encoding="utf-8", newline='') as wf:
        writer = csv.writer(wf)
        for corrupted, sentence in valid_collection:
            writer.writerow([corrupted, sentence])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="data/wmt19_v18")
    parser.add_argument('--train_file', type=str, default="data/train_ready.txt")
    parser.add_argument('--valid_file', type=str, default="data/valid_ready.txt")

    parser.add_argument('--model_name_or_path', type=str, default="ai-forever/mGPT")

    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2023)

    parser.add_argument('--train_size', type=int, default=50000)
    parser.add_argument('--valid_size', type=int, default=10000)

    parser.add_argument('--shuffle_ratio', type=float, default=0.33)

    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    data_preparation(args)