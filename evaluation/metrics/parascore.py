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
"""ParaScore: score for paraphrase: take into account semantic similarity and lexical diversity"""

import evaluate
import datasets
import nltk


_CITATION = """\
@article{Shen2022,
archivePrefix = {arXiv},
arxivId = {2202.08479},
author = {Shen, Lingfeng and Liu, Lemao and Jiang, Haiyun and Shi, Shuming},
journal = {EMNLP 2022 - 2022 Conference on Empirical Methods in Natural Language Processing, Proceedings},
eprint = {2202.08479},
month = {feb},
number = {1},
pages = {3178--3190},
title = {{On the Evaluation Metrics for Paraphrase Generation}},
url = {http://arxiv.org/abs/2202.08479},
year = {2022}
}
"""

_DESCRIPTION = """\
ParaScore is a new metric to scoring the performance of paraphrase generation tasks
"""

_KWARGS_DESCRIPTION = """
Calculates how good the paraphrase is
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
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
BAD_WORDS_URL = "https://github.com/shadowkiller33/parascore_toolkit"


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class Parascore(evaluate.Metric):
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
                        "references": datasets.Value("string", id="sequence"),
                        "sources": datasets.Value("string", id="sequence"),
                    }
                ),
            ],
            # Homepage of the module for documentation
            homepage="https://github.com/shadowkiller33/ParaScore",
            # Additional links to the codebase or references
            codebase_urls=["https://github.com/shadowkiller33/ParaScore"],
            reference_urls=["https://github.com/shadowkiller33/ParaScore"]
        )

    def _download_and_prepare(self, dl_manager):
        """Optional: download external resources useful to compute the scores"""
        self.bertscore = evaluate.load('bertscore')

    def _edit(self, x, y, lang='en'):
        if lang == 'zh':
            x = x.replace(" ", "")
            y = y.replace(" ", "")
        a = len(x)
        b = len(y)
        dis = nltk.edit_distance(x,y)
        return dis/max(a,b)

    def _diverse(self, cands, sources, lang='en'):
        diversity = []
        thresh = 0.35
        for x, y in zip(cands, sources):
            div = self._edit(x, y, lang)
            if div >= thresh:
                ss = thresh
            elif div < thresh:
                ss = -1 + ((thresh + 1) / thresh) * div
            diversity.append(ss)
        return diversity

    def _compute(
        self,
        sources,
        predictions,
        references,
        model_type: str='bert-base-multilingual-cased',
        lang: str='en',
        calc_ref: bool=False
    ):
        """Returns the scores"""
        
        sem_score = self.bertscore.compute(
            predictions=predictions,
            references=sources,
            model_type=model_type,
            lang=lang
        )
        sem_score = sem_score['f1']
        if calc_ref:
            score_ref = self.bertscore.compute(
                predictions=predictions,
                references=references,
                model_type=model_type,
                lang=lang
            )
            score_ref = score_ref['f1']
            score_src = sem_score
            sem_score = [max(s, r) for s, r in zip(score_src, score_ref)]

        diversity = self._diverse(predictions, sources, lang)

        score = [s + 0.05 * d for s, d in zip(sem_score, diversity)]
        return {
            "score": score,
        }