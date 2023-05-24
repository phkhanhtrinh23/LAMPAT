from transformers import AutoTokenizer
import torch
from underthesea import word_tokenize
import spacy
import os

"""
mGPT language support:
az, sw, af, ar, ba, be, bxr, bg, bn, cv, hy, da, de, el, es, eu, fa, fi, fr, 
he, hi, hu, kk, id, it, ja, ka, ky, ko, lt, lv, mn, ml, os, mr, ms, my, nl, 
ro, pl, pt, sah, ru, tg, sv, ta, te, tk, th, tr, tl, tt, tyv, uk, en, ur, 
vi, uz, yo, zh, xal
"""

input_text = "tôi yêu cánh đồng ngoài khu phức hợp gần trường Nguyễn Khuyến trong thời gian tôi thực tập ở đó"
# tokenizer = TreebankWordTokenizer() 
print(word_tokenize(input_text))

tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/mGPT")
# model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
encoded_zh = tokenizer(input_text + "\n" + input_text, return_tensors="pt")
print(encoded_zh["input_ids"], len(encoded_zh["input_ids"][0]))

# With view
idx = torch.randperm(encoded_zh["input_ids"][0].nelement())
t = encoded_zh["input_ids"][0].view(-1)[idx].view(encoded_zh["input_ids"][0].size())
print(t)
print(tokenizer.batch_decode([t], skip_special_tokens=True))

nlp = spacy.load("ja_core_news_sm")
doc = nlp("ウィナースキーは、IEEE、ファイ・ベータ・カッパ、ACM、およびシグマ・サイのメンバーである。")
print(doc.text)
ja_tokens = []
for token in doc:
    ja_tokens.append(token.text)
print("".join(ja_tokens))

nlp = spacy.load("ko_core_news_sm")
doc = nlp("위나스키는 IEEE, 피 베타카파, ACM과 시그마 Xi의 회원이다.")
print(doc.text)
ja_tokens = []
for token in doc:
    ja_tokens.append(token.text.lower())
print(" ".join(ja_tokens))

nlp = spacy.load("zh_core_web_sm")
doc = nlp("在于 3 月 14 日收购了 E-Plus 剩余的股份后")
print(doc.text)
ja_tokens = []
for token in doc:
    ja_tokens.append(token.text)
print("".join(ja_tokens))

nlp = spacy.load("hu_core_news_lg")
doc = nlp("el fogja törni a tányért.")
print(doc.text)
ja_tokens = []
for token in doc:
    ja_tokens.append(token.text)
print(" ".join(ja_tokens))

list_of_files = os.listdir("data/wmt19_v18")
min_length = -1
name = ""

for file in list_of_files:
    with open(os.path.join("data/wmt19_v18", file), "r", encoding="utf-8") as f:
        l = len(f.readlines())
        name = file if l < min_length else name
        min_length = l if l < min_length or min_length == -1 else min_length
        
print("Min length:", min_length)
print("Name:", name)

valid_size = int(min_length / 3)
train_size = min_length - valid_size
print(train_size, valid_size)