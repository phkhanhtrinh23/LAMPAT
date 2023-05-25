import evaluate
import os
from typing import List
import numpy as np
import pandas as pd

def average(nums: List[float]) -> float:
    return np.average(nums)

if __name__ == "__main__":
    # unsupervised mean input only

    # metrics will be used
    # BERTScore
    # ParaScore
    # BERT-iBLEU

    bertscore = evaluate.load('bertscore')
    parascore = evaluate.load('metrics/parascore.py')
    bert_ibleu = evaluate.load('metrics/bert_ibleu.py')

    # dataset: wmt19_input
    path = "eval_dataset/wmt19_input"
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
            references=df['input'],
            model_type='bert-base-multilingual-cased',
            lang=lang,
        )
        print(f"ParaScore: {average(parascore_scores['score'])}")
        bert_ibleu_scores = bert_ibleu.compute(
            predictions=df['prediction'],
            sources=df['input'],
            model_type='bert-base-multilingual-cased',
            lang=lang,
        )
        print(f"BERT-iBLEU: {average(bert_ibleu_scores['score'])}")


