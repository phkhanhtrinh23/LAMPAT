import evaluate
import os
from typing import List
import numpy as np
import pandas as pd

def average(nums: List[float]) -> float:
    return np.average(nums)

def max_func(ls1, ls2, ls3, ls4):
    return [max(v1, v2, v3, v4) for v1, v2, v3, v4 in zip(ls1['score'], ls2['score'], ls3['score'], ls4['score'])]

if __name__ == "__main__":
    # supervised multi reference

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
    ter = evaluate.load('ter')

    # dataset: wmt19_input
    path = "eval_dataset/staple_input"
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
        parascore_scores1 = parascore.compute(
            predictions=df['prediction'],
            sources=df['input'],
            references=df['reference1'],
            model_type='bert-base-multilingual-cased',
            lang=lang,
            calc_ref=True,
        )
        parascore_scores2 = parascore.compute(
            predictions=df['prediction'],
            sources=df['input'],
            references=df['reference2'],
            model_type='bert-base-multilingual-cased',
            lang=lang,
            calc_ref=True,
        )
        parascore_scores3 = parascore.compute(
            predictions=df['prediction'],
            sources=df['input'],
            references=df['reference3'],
            model_type='bert-base-multilingual-cased',
            lang=lang,
            calc_ref=True,
        )
        parascore_scores4 = parascore.compute(
            predictions=df['prediction'],
            sources=df['input'],
            references=df['reference4'],
            model_type='bert-base-multilingual-cased',
            lang=lang,
            calc_ref=True,
        )
        parascore_scores = max_func(parascore_scores1, parascore_scores2, parascore_scores3, parascore_scores4)
        print(f"ParaScore: {average(parascore_scores)}")
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
        sacrebleu_scores1 = sacrebleu.compute(
            predictions=df['prediction'],
            references=df['reference1'],
            tokenize=tokenizer,
        )
        sacrebleu_scores2 = sacrebleu.compute(
            predictions=df['prediction'],
            references=df['reference2'],
            tokenize=tokenizer,
        )
        sacrebleu_scores3 = sacrebleu.compute(
            predictions=df['prediction'],
            references=df['reference3'],
            tokenize=tokenizer,
        )
        sacrebleu_scores4 = sacrebleu.compute(
            predictions=df['prediction'],
            references=df['reference4'],
            tokenize=tokenizer,
        )
        sacrebleu_scores = max(sacrebleu_scores1['score'], sacrebleu_scores2['score'], sacrebleu_scores3['score'], sacrebleu_scores4['score'])
        print(f"SacreBLEU: {sacrebleu_scores}")
        ter_scores1 = ter.compute(
            predictions=df['prediction'],
            references=df['reference1'],
            normalized=True,
            support_zh_ja_chars=True,
        )
        ter_scores2 = ter.compute(
            predictions=df['prediction'],
            references=df['reference2'],
            normalized=True,
            support_zh_ja_chars=True,
        )
        ter_scores3 = ter.compute(
            predictions=df['prediction'],
            references=df['reference3'],
            normalized=True,
            support_zh_ja_chars=True,
        )
        ter_scores4 = ter.compute(
            predictions=df['prediction'],
            references=df['reference4'],
            normalized=True,
            support_zh_ja_chars=True,
        )
        ter_scores = max(ter_scores1['score'], ter_scores2['score'], ter_scores3['score'], ter_scores4['score'])
        print(f"TER: {ter_scores}")

