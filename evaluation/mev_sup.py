import evaluate
import os
from typing import List
import numpy as np
import pandas as pd

def average(nums: List[float]) -> float:
    return np.average(nums)

if __name__ == "__main__":
    # supervised mean input & reference

    # metrics will be used
    # BERTScore (input, prediction)
    # ParaScore (input, prediction, reference)
    # BERT-iBLEU (input, prediction)
    # SacreBLEU (prediction, reference)
    # TER (prediction, reference)

    bertscore = evaluate.load('bertscore')
    parascore = evaluate.load('metrics/parascore.py')
    bert_ibleu = evaluate.load('metrics/bert_ibleu.py')
    sacrebleu = evaluate.load('sacrebleu')
    selfbleu = evaluate.load('sacrebleu')
    ter = evaluate.load('ter')

    # dataset: opusparcus_input | pawsx_input
    path = "eval_dataset/pawsx_input"
    langs = os.listdir(path)
    for lang in langs:
        print(f"Lang: {lang}")
        filename = f'{path}/{lang}/result.csv'
        df = pd.read_csv(filename)
        df.fillna('', inplace=True)
        bertscore_scores = bertscore.compute(
            predictions=df['prediction'],
            references=df['input'],
            model_type='bert-base-multilingual-cased',
            lang=lang,
        )
        print(f"BERTScore-F1: {average(bertscore_scores['f1'])}")
        parascore_scores = parascore.compute(
            predictions=df['prediction'],
            sources=df['input'],
            references=df['reference'],
            model_type='bert-base-multilingual-cased',
            lang=lang,
            calc_ref=True,
        )
        print(f"ParaScore: {average(parascore_scores['score'])}")
        bert_ibleu_scores = bert_ibleu.compute(
            predictions=df['prediction'],
            sources=df['input'],
            model_type='bert-base-multilingual-cased',
            lang=lang,
        )
        print(f"BERT-iBLEU: {average(bert_ibleu_scores['score'])}")
        if lang == 'zh':
            tokenizer = 'zh'
        elif lang == 'ja':
            tokenizer = 'ja-mecab'
        else:
            tokenizer = '13a'
        sacrebleu_scores = sacrebleu.compute(
            predictions=df['prediction'],
            references=df['reference'],
            tokenize=tokenizer,
        )
        print(f"SacreBLEU: {sacrebleu_scores['score']}")
        if lang == 'zh':
            tokenizer = 'zh'
        elif lang == 'ja':
            tokenizer = 'ja-mecab'
        else:
            tokenizer = '13a'
        selfbleu_scores = selfbleu.compute(
            predictions=df['prediction'],
            references=df['input'],
            tokenize=tokenizer,
        )
        print(f"SelfBLEU: {selfbleu_scores['score']}")
        ter_scores = ter.compute(
            predictions=df['prediction'],
            references=df['reference'],
            normalized=True,
            support_zh_ja_chars=True,
        )
        print(f"TER: {ter_scores['score']}")


