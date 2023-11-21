'''
需要测的baseline: 
BLEU
ROUGE-1/2/L
BERTScore-p/r/F1
QAGS, Q^2
FactScore
'''
import json, os, factscore, nltk
import numpy as np
from factsumm import FactSumm
from rich import print
from tqdm import tqdm
from rouge_score import rouge_scorer                    # ROUGE
from bert_score import score                            # BERTScore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction     # BLEU
from BARTScore.bart_score import BARTScorer
factsumm = FactSumm()
bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
nltk.download('punkt')
def tokenize(text):
    return nltk.word_tokenize(text)
Dataset = [
    ["NQ", "answer"],  # answer: list
    ["HotpotQA", "answer"],    # answer: str
    ["truthfulQA", "answer"],   # answer: list
    ["cnndm", "summary"],      # summary: str
    ["multi-news", "summary"], # summary: str
    ["msmarco", "summary"] # summary可能为空
]
prefix = "/home/dou/UniFlexFactor/dataset"
Model = ["newbing.jsonl", "chatgpt.jsonl", "llama7b.jsonl", "llama13b.jsonl", "vicuna7b.jsonl", "vicuna13b.jsonl"]
fileNum = 200
def factScore():
    pass
def Q_square():
    pass
def QAGS(cand: str, ref: str):
    QAGS_score = factsumm.extract_qas(ref, cand, device="cuda")
    return {
        "QAGS_score": round(QAGS_score, 3)
    }

def bleuScore(cand: str, ref: str):
    ref = [tokenize(ref.lower())]
    cand = tokenize(cand.lower())
    smoothing = SmoothingFunction().method1
    score_bleu1 = round(sentence_bleu(ref, cand, weights=(1, 0, 0, 0)), 3)
    score_bleu2 = round(sentence_bleu(ref, cand, weights=(0.5, 0.5, 0, 0)), 3)
    score_bleu3 = round(sentence_bleu(ref, cand, weights=(0.33, 0.33, 0.33, 0)), 3)
    score_bleu4 = round(sentence_bleu(ref, cand, smoothing_function=smoothing), 3)
    return {
        "bleu1": score_bleu1, 
        "bleu2": score_bleu2, 
        "bleu3": score_bleu3, 
        "bleu4": score_bleu4
    }
def rougeScore(pred: str, ref: str):
    '''
    str1: model
    str2: 
    '''
    res = scorer.score(ref, pred)
    return {
        "rouge1": round(res["rouge1"][1], 3),     # precision, recall, f1-measure
        "rouge2": round(res["rouge2"][1], 3),     # precision, recall, f1-measure
        "rougeL": round(res["rougeL"][1], 3)      # precision, recall, f1-measure
    }
def bertScore(preds: str, refs: str, model_type="roberta-large"):
    if type(preds) == str: preds = [preds]
    if type(refs) == str: refs = [refs]
    P, R, F1 = score(preds, refs, model_type=model_type, lang='en', verbose=True)    
    return {
        "bertscore-p": round(float(P), 3),
        "bertscore-r": round(float(R), 3),
        "bertscore-f1": round(float(F1), 3)
    }
def bartScore(pred: str, ref: str):
    if type(pred) == str: pred = [pred]
    if type(ref) == str: ref = [ref]
    bartscore = bart_scorer.score(pred, ref, batch_size=4)
    return {
        "bartScore": round(bartscore[0], 3)
    }


if __name__ == "__main__":
    for dataset, ansKey in Dataset:
        for model in Model:
            fileName = os.path.join(prefix, dataset, model)
            resultList = list()
            with open(fileName, 'r', encoding='utf-8') as fp:
                data = fp.readlines()[:fileNum]
                for line in tqdm(data, desc=f"{dataset}, {model}"):
                    result = dict()
                    line = json.loads(line)
                    modelText = line["modelText"]
                    if dataset == "NQ":
                        answer = line["answer"][0]
                    if dataset == "HotpotQA":
                        answer = line["answer"]
                    if dataset == "truthfulQA":
                        answer = line["answer"][0]
                    if dataset == "cnndm":
                        answer = line["summary"]
                    if dataset == "multi-news":
                        answer = line["summary"]
                    if dataset == "msmarco":
                        answer = line["summary"][0] if line["summary"] != [] else "Answers are unavailable."
                    result.update(bleuScore(modelText, answer))
                    result.update(rougeScore(modelText, answer))
                    result.update(bertScore(modelText, answer))
                    result.update(bartScore(modelText, answer))
                    result.update(QAGS(modelText, answer))
                    print(result)
                    break
        print("*"*20)