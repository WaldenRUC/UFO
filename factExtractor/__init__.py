import pysbd
from sumeval.metrics.rouge import RougeCalculator
from typing import List
from factExtractor.utils.utils import *
from factExtractor.utils.level_sentence import load_qa, load_qg, load_ner, load_d2q
def timer(func):
    def func_wrapper(*args,**kwargs):
        from time import time
        time_start = time()
        result = func(*args,**kwargs)
        time_end = time()
        time_spend = time_end - time_start
        # print('\n{0} cost time {1} s\n'.format(func.__name__, time_spend))
        return result
    return func_wrapper
class FactExtractor:
    def __init__(
            self, 
            plmPathPrefix = None,
            ner_model = None, 
            qg_model = None,
            qa_model = None,
            d2q_model = None) -> None:
        self.config = Config()
        self.segmenter = pysbd.Segmenter(language="en", clean=False)
        self.rouge = RougeCalculator(stopwords=True, lang="en")
        self.plmPathPrefix = plmPathPrefix
        self.ner = ner_model if ner_model is not None else self.config.NER_MODEL
        self.qg = qg_model if qg_model is not None else self.config.QG_MODEL
        self.qa = qa_model if qa_model is not None else self.config.QA_MODEL
        self.d2q = d2q_model if d2q_model is not None else self.config.D2Q_MODEL
    def load_everything(self):
        self.ner = load_ner(self.ner, self.plmPathPrefix)
        self.qg = load_qg(self.qg, self.plmPathPrefix)
        self.qa = load_qa(self.qa, self.plmPathPrefix)
        self.d2q = load_d2q(self.d2q, self.plmPathPrefix)
    def _segment(self, text: str):
        return [line.strip() for line in self.segmenter.segment(text)]
    def extract_fact_single(
        self,
        modelText: str,
        modelText_ents: List = None,
    ):
        """
        Args:
            modelText (str): generated text
            modelText_ents (List, optional): named entities extracted from source. Defaults to None.
            verbose (bool, optional): print verbose option. Defaults to False.
        """
        modelText_lines = self._segment(modelText)
        if modelText_ents is None:
            modelText_ents = self.ner(modelText_lines)
        modelText_qas = self.qg(modelText_lines, modelText_ents)
        # modelText_answers = self.qa(modelText, modelText_qas)
        title = self.d2q(modelText)
        return title, modelText_ents, modelText_qas# , modelText_answers