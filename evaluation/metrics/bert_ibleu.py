# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT-iBLEU: score for paraphrase: take into account semantic similarity and lexical diversity"""

import evaluate
import datasets
import nltk


_CITATION = """\
@inproceedings{Niu2021,
  title = {Unsupervised {{Paraphrasing}} with {{Pretrained Language Models}}},
  booktitle = {Proceedings of the 2021 {{Conference}} on {{Empirical Methods}} in {{Natural Language Processing}}},
  author = {Niu, Tong and Yavuz, Semih and Zhou, Yingbo and Keskar, Nitish Shirish and Wang, Huan and Xiong, Caiming},
  year = {2021},
  eprint = {2010.12885},
  pages = {5136--5150},
  publisher = {{Association for Computational Linguistics}},
  address = {{Stroudsburg, PA, USA}},
  doi = {10.18653/v1/2021.emnlp-main.417},
  archiveprefix = {arxiv},
  isbn = {978-1-955917-09-4},
}

"""

_DESCRIPTION = """\
BERT-iBLEU is a metric to scoring the performance of paraphrase generation tasks
"""

_KWARGS_DESCRIPTION = """
Calculates how good the paraphrase is
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    sources: list of reference for each prediction. Each
        sources should be a string with tokens separated by spaces.
Returns:
    score: description of the first score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> metrics = evaluate.load("transZ/test_parascore")
    >>> results = my_new_module.compute(references=["They work for 6 months"], predictions=["They have working for 6 months"])
    >>> print(results)
    {'score': 0.85}
"""

# TODO: Define external resources urls if needed
BAD_WORDS_URL = "https://github.com/TokisakiKurumi2001/dynamic_blocking"


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class BERT_iBLEU(evaluate.Metric):
    """ParaScore is a new metric to scoring the performance of paraphrase generation tasks"""

    def _info(self):
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=[
                datasets.Features(
                    {
                        "predictions": datasets.Value("string", id="sequence"),
                        "sources": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            # Homepage of the module for documentation
            homepage="https://github.com/TokisakiKurumi2001/dynamic_blocking",
            # Additional links to the codebase or references
            codebase_urls=["https://github.com/TokisakiKurumi2001/dynamic_blocking"],
            reference_urls=["https://github.com/TokisakiKurumi2001/dynamic_blocking"]
        )

    def _download_and_prepare(self, dl_manager):
        """Optional: download external resources useful to compute the scores"""
        self.bertscore = evaluate.load('bertscore')
        self.sacrebleu = evaluate.load('sacrebleu')

    def _compute(
        self,
        sources,
        predictions,
        model_type: str='bert-base-multilingual-cased',
        lang: str='en',
    ):
        """Returns the scores"""
        
        sem_score = self.bertscore.compute(
            predictions=predictions,
            references=sources,
            model_type=model_type,
            lang=lang
        )
        sem_score = sem_score['f1']

        if lang == 'zh':
            tokenizer = 'zh'
        elif lang == 'ja':
            tokenizer = 'ja-mecab'
        else:
            tokenizer = '13a'

        self_bleu = [self.sacrebleu.compute(predictions=[pred], references=[src], tokenize=tokenizer)['score'] / 100.0 for src, pred in zip(sources, predictions)]

        beta = 4.0

        scores = [((beta * (max(sem, 1e-5)**(-1)) + 1.0 * ((max(1.0 - lex, 1e-5))**(-1)))/(beta + 1.0))**(-1) for sem, lex in zip(sem_score, self_bleu)]
        return {
            "score": scores,
        }