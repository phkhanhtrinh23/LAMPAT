from datetime import datetime
import json
import torch
import argparse
import logging
import csv
import os
import random
import nltk
import numpy as np
import datasets

import spacy
import stopwordsiso
from nltk.corpus import stopwords as nltk_stopwords
from underthesea import word_tokenize as vi_tokenize
from transformers import default_data_collator

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, PeftConfig, PeftModel

start_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

ja_tokenize = spacy.load("ja_core_news_sm")
ko_tokenize = spacy.load("ko_core_news_sm")
zh_tokenize = spacy.load("zh_core_web_sm")
hu_tokenize = spacy.load("hu_core_news_lg")

lang_dict = {
    "de": "german",
    "en": "english",
    "es": "spanish",
    "fi": "finnish",
    "fr": "french",
    "ja": "japanese",
    "ko": "korean",
    "ru": "russian",
    "sv": "swedish",
    "zh": "chinese",
    "hu": "hungarian",
    "pt": "portuguese",
    "vi": "vietnamese",
    "cs": "czech",
}

"""
word_tokenize for shuffling:
    vi: underthesea
    ko: spacy - ko_core_news_sm
    ja: spacy - ja_core_news_sm
    zh: spacy - zh_core_web_sm
    hu: spacy - hu_core_news_lg (huspacy)
    others: nltk.word_tokenize

stopwords for removing:
    vi: stopwordsiso
    ja: stopwordsiso
    ko: stopwordsiso
    cs: stopwordsiso
    zh: stopwordsiso
    others: nltk.corpus.stopwords
"""

class EvalDataCollator:
    def __init__(self,
                filename,
                dataset_name,
                tokenizer,
                max_length=128,
                special_character="[SEP]",
                ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.filename = filename
        self.max_length = max_length
        self.special_character = special_character

    def load_dataset(self):
        filename = self.filename
        info = dict()
        tokens_list, original_list, labels_list = [], [], []
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            if self.dataset_name == "staple_input":
                for corrupted, original, tar_1, tar_2, tar_3, tar_4 in reader:
                    tokens = self.formatting(corrupted + self.special_character)
                    tokens_list.append(tokens)
                    original_list.append(original)
                    labels_list.append([tar_1, tar_2, tar_3, tar_4])
            else:
                for corrupted, original, tar in reader:
                    tokens = self.formatting(corrupted + self.special_character)
                    tokens_list.append(tokens)
                    original_list.append(original)
                    labels_list.append(tar)

        sentences = [self.tokenizer.decode(tokens) for tokens in tokens_list]
        
        encodings = self.tokenizer(
            sentences, return_tensors='pt', truncation=True,
            padding='max_length', max_length=self.max_length)
        
        info["original"] = original_list
        info["labels"] = labels_list
        
        return encodings, info
    
    def formatting(self, input_text):
        input_tokens = self.tokenizer.encode(input_text)
        
        return input_tokens

def pre_delete(sentence, lang):
    # word tokenization
    if lang == "vi":
        sentence = vi_tokenize(sentence)
    elif lang == "ja":
        tokens = ja_tokenize(sentence)
        sentence = [doc.text for doc in tokens]
    elif lang == "ko":
        tokens = ko_tokenize(sentence)
        sentence = [doc.text for doc in tokens]
    elif lang == "zh":
        tokens = zh_tokenize(sentence)
        sentence = [doc.text for doc in tokens]
    elif lang == "hu":
        tokens = hu_tokenize(sentence)
        sentence = [doc.text for doc in tokens]
    else:
        sentence = nltk.word_tokenize(sentence, 
                                      language=lang_dict[lang])
    
    # stopwords
    if lang in ["vi", "ja", "ko", "cs", "zh"]:
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
    sentence = sentence.replace("''", '"').replace('``', '"')
    return sentence

def pre_shuffle(sentence, lang, shuffle_ratio=0.33):
    # word tokenization
    if lang == "vi":
        words = vi_tokenize(sentence)
    elif lang == "ja":
        tokens = ja_tokenize(sentence)
        words = [doc.text for doc in tokens]
    elif lang == "ko":
        tokens = ko_tokenize(sentence)
        words = [doc.text for doc in tokens]
    elif lang == "zh":
        tokens = zh_tokenize(sentence)
        words = [doc.text for doc in tokens]
    elif lang == "hu":
        tokens = hu_tokenize(sentence)
        words = [doc.text for doc in tokens]
    else:
        try:
            words = nltk.word_tokenize(sentence, 
                                      language=lang_dict[lang])
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

def preprocess(args, dataset_name, lang_path, lang, data):
    gpt_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, 
                                                  src_lang=lang)
    outputs = []
    skipped = 0
    if dataset_name == "staple_input":
        for i, [inp, tar_1, tar_2, tar_3, tar_4] in enumerate(data):
            sentence = inp.strip()
            corrupted_sentence = pre_delete(sentence, lang)
            corrupted_sentence = pre_shuffle(corrupted_sentence, 
                                             lang,
                                             args.shuffle_ratio)
            write_line = corrupted_sentence + '\n' + sentence
            if len(gpt_tokenizer.encode(write_line)) < args.max_length:
                outputs.append([corrupted_sentence, sentence, 
                                tar_1, tar_2, tar_3, tar_4])
            else:
                skipped += 1
            print(f"Processed sentence(s): {i+1}", end="\r")

        print(f"Number of skipped sentence(s) - {lang_path} - {lang}: {skipped}")

        with open(os.path.join(lang_path, f"{lang}.csv"), 'w', encoding="utf-8", newline='') as wf:
            writer = csv.writer(wf)
            for corrupted, sentence, tar_1, tar_2, tar_3, tar_4 in outputs:
                writer.writerow([corrupted, sentence, 
                                 tar_1, tar_2, tar_3, tar_4])
    else:
        for i, [inp, tar] in enumerate(data):
            sentence = inp.strip()
            corrupted_sentence = pre_delete(sentence, lang)
            corrupted_sentence = pre_shuffle(corrupted_sentence,
                                             lang,
                                             args.shuffle_ratio)
            write_line = corrupted_sentence + '\n' + sentence
            if len(gpt_tokenizer.encode(write_line)) < args.max_length:
                outputs.append([corrupted_sentence, sentence, tar])
            else:
                skipped += 1
            print(f"Processed sentence(s): {i+1}", end="\r")

        print(f"Number of skipped sentence(s) - {lang_path} - {lang}: {skipped}")

        with open(os.path.join(lang_path, f"{lang}.csv"), 'w', encoding="utf-8", newline='') as wf:
            writer = csv.writer(wf)
            for corrupted, sentence, tar in outputs:
                writer.writerow([corrupted, sentence, tar])

def data_preparation(args):
    datasets = os.listdir(args.eval_dataset_path)

    for dataset_name in datasets:
        dataset_path = os.path.join(args.eval_dataset_path, dataset_name)
        languages = os.listdir(dataset_path)
        languages = [language for language in languages if os.path.isdir(os.path.join(dataset_path, language))]

        for language in languages:
            lang_path = os.path.join(dataset_path, language)
            data = []
            if dataset_name == "opusparcus_input":
                with open(os.path.join(lang_path, "test.csv"), "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for idx, row in enumerate(reader):
                        # skip the header
                        if idx == 0:
                            continue
                        inp, tar, _ = row
                        data.append([inp, tar])
            elif dataset_name == "pawsx_input":
                with open(os.path.join(lang_path, "test.csv"), "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for idx, row in enumerate(reader):
                        # skip the header
                        if idx == 0:
                            continue
                        _, _, inp, tar, _ = row
                        data.append([inp, tar])
            elif dataset_name == "staple_input":
                with open(os.path.join(lang_path, "test.csv"), "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    for idx, row in enumerate(reader):
                        # skip the header
                        if idx == 0:
                            continue
                        inp, tar_1, tar_2, tar_3, tar_4 = row
                        data.append([inp, tar_1, tar_2, tar_3, tar_4])
            elif dataset_name == "wmt19_input":
                with open(os.path.join(lang_path, "validation.txt"), "r", encoding="utf-8") as f:
                    for sen in f.readlines():
                        sen = sen[:-1]
                        data.append([sen, sen])
            
            preprocess(args, dataset_name, lang_path, language, data)

def run(args, eval_dataloader, info, tokenizer, dataset_name, lang_path, peft_model_id):
    with open(os.path.join(lang_path, "result.csv"), 'w', encoding="utf-8", newline='') as wf:
        writer = csv.writer(wf)
        if dataset_name == "staple_input":
            writer.writerow(["input","prediction","reference_1","reference_2","reference_3","reference_4"])    
        else:
            writer.writerow(["input","prediction","reference"])

        logging.info(f"Working with {lang_path}...\n")
        
        with torch.no_grad():
            kwargs = {
                'max_length': args.max_length,
                'num_return_sequences': args.num_return_sequences,
            }
            
            i = 0
            for batch in tqdm(eval_dataloader, position=0, desc="Inference", leave=False):
                originals = info["original"][i: i + len(batch["input_ids"])]
                labels = info["labels"][i: i + len(batch["input_ids"])]
                
                config = PeftConfig.from_pretrained(peft_model_id)
                model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
                model.resize_token_embeddings(len(tokenizer))
                model = PeftModel.from_pretrained(model, peft_model_id)
                model = model.to(args.device)
                model.eval()
                
                batch = {k: v.to(args.device) for k, v in batch.items()}
                inputs_encoding = {"attention_mask": batch["attention_mask"]}
                
                embeds_init = model.transformer.wte(batch["input_ids"])
                input_mask = batch['attention_mask'].to(embeds_init)
                input_lengths = torch.sum(input_mask, 1)
                delta = torch.zeros_like(embeds_init).normal_(0, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = args.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
                
                inputs_encoding['inputs_embeds'] = delta + embeds_init
                
                output = model.generate(inputs_embeds=inputs_encoding["inputs_embeds"],
                                        attention_mask=inputs_encoding["attention_mask"],
                                        pad_token_id=tokenizer.pad_token_id,
                                        **kwargs)
                try:
                    output_text = tokenizer.batch_decode(output[:,1:output.tolist()[0].index(5)].detach(), skip_special_tokens=True)
                except:
                    output_text = tokenizer.batch_decode(output[:,1:].detach(), skip_special_tokens=True)
                
                if dataset_name == "staple_input":
                    for original, output, [label_1, label_2, label_3, label_4] in zip(originals, output_text, labels):
                        writer.writerow([original, output.replace("\n",""), label_1, label_2, label_3, label_4])
                else:
                    for original, output, label in zip(originals, output_text, labels):
                        writer.writerow([original, output.replace("\n",""), label])

                i = i + len(batch["input_ids"])

def main(args):
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    peft_model_id = f"{args.model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
    
    logging.info(f"peft_model_id: {peft_model_id}")

    datasets_ls = os.listdir(args.eval_dataset_path)

    for dataset_name in datasets_ls:
        dataset_path = os.path.join(args.eval_dataset_path, dataset_name)
        languages = os.listdir(dataset_path)
        languages = [language for language in languages if os.path.isdir(os.path.join(dataset_path, language))]

        for language in languages:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, src_lang=language)
            special_tokens_dict = {'sep_token': '[SEP]'}
            tokenizer.add_special_tokens(special_tokens_dict)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

            lang_path = os.path.join(dataset_path, language)

            eval_dataset = EvalDataCollator(filename=os.path.join(lang_path, f"{language}.csv"),
                                            dataset_name=dataset_name,
                                            tokenizer=tokenizer,
                                            max_length=args.max_length,
                                            special_character="[SEP]")
            eval_dataset, info = eval_dataset.load_dataset()
            eval_dataset = datasets.Dataset.from_dict(eval_dataset)
            eval_dataloader = DataLoader(
                eval_dataset, 
                collate_fn=default_data_collator,
                batch_size=args.batch_size, 
                pin_memory=True
            )
            
            run(args, eval_dataloader, info, tokenizer, dataset_name, lang_path, peft_model_id)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_dataset_path', type=str,
                        default='./eval_dataset',
                        help='eval dataset folder')
    
    parser.add_argument('--log', type=str,
                        default='./logs/evaluate_{datetime}.log',
                        help='Log filename')

    parser.add_argument('--device', type=str, default='cuda',
                        help='{cuda, cpu}')

    parser.add_argument('--model_name_or_path', type=str, default="sberbank-ai/mGPT",
                        help='pretrained model name')
    parser.add_argument('--max_length', type=int, default=100,
                        help='Maximum number of tokens for each sequence')
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help='Maximum number of return sequences')

    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--shuffle_ratio', type=float, default=0.33)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--adv_init_mag', type=float, default=1)

    args = parser.parse_args()
    
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    log_level = logging.INFO
    log_file = args.log

    logging.basicConfig(level=log_level, format=log_format,
                        filename=log_file.format(datetime=start_datetime.replace(':','-')))
    logging.getLogger().setLevel(log_level)

    logging.info(f'Parsed args: {json.dumps(dict(args.__dict__), indent=2)}')
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Run once only
    data_preparation(args)

    main(args)