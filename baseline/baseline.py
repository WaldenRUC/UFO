'''
需要测的baseline: 
BLEU
ROUGE-1/2/L
BERTScore-p/r/F1
QAGS, Q^2
FactScore
'''
import json, os, nltk, csv
from factscore.factscorer import FactScorer
fs = FactScorer(openai_key="/home/dou/factscoreData/openai_key.txt", data_dir="/home/dou/factscoreData")
import numpy as np
from factsumm import FactSumm
from rich import print
from tqdm import tqdm
from rouge_score import rouge_scorer                    # ROUGE
from bert_score import score                            # BERTScore
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction     # BLEU
from BARTScore.bart_score import BARTScorer
from Qsquared.pipeline.run_pipeline import calc_scores
from collections import defaultdict
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
outputPath = "result.json"
fileNum = 200
def factScore(topic: str, cand: str, ref: str):
    fact_score = fs.get_score([topic], [cand], gamma=10)["score"]
    return { 
        "factscore": fact_score
    }
def Q_square(cand: str, ref: str):
    # 准备数据
    data = [
        ["", "episode_idx", "round", "response", "knowledge", "gold"],
        [0, 0, 0, cand.replace("\n", ""), ref.replace("\n", ""), ref.replace("\n", "")]
    ]
    csv_file = "Qsquared/third_party/data/temp.csv"
    with open(csv_file, mode="w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerows(data)
    Q2_score = calc_scores(
        "Qsquared/third_party/data/temp.csv", 'beam', single=True, remove_personal=True, out_path='Qsquared/third_party/data/temp_out', save_steps=False
    )
    # 读取数据
    return {
        "Q2_score": round(Q2_score[0], 3)
    }
def QAGS(cand: str, ref: str):
    QAGS_score = factsumm.extract_qas(ref, cand)
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
        "bartscore": round(bartscore[0], 3)
    }



if __name__ == "__main__":
    metrics = [
        #"rouge1", "rouge2", "rougeL", "bleu1", "bleu2", "bleu3", "bleu4",
        # "bertscore-p", "bertscore-r", "bertscore-f1",
        # "bartscore",
        "QAGS_score",
        "Q2_score",
        # "factscore"
    ]
    resultDict = dict()
    if os.path.exists(outputPath):
        with open(outputPath, 'r', encoding='utf-8') as fp:
            resultDict = json.load(fp)
    else:
        for dataset in Dataset:
            resultDict[dataset[0]] = dict()
            for model in Model:
                resultDict[dataset[0]][model.replace(".jsonl", "")] = defaultdict(list)
    for dataset, ansKey in Dataset:
        for model in Model:
            fileName = os.path.join(prefix, dataset, model)
            resultList = list()
            with open(fileName, 'r', encoding='utf-8') as fp:
                data = fp.readlines()[:fileNum]
                for dataID, line in enumerate(tqdm(data, desc=f"{dataset}, {model}")):
                    line = json.loads(line)
                    modelText = line["modelText"]
                    if dataset == "NQ":
                        topic = line["prompt"].split("Title: ")[1].split("Introduction:")[0].strip()
                        answer = line["answer"][0]
                    if dataset == "HotpotQA":
                        topic = line["question"]
                        answer = line["answer"]
                    if dataset == "truthfulQA":
                        topic = line["question"]
                        answer = line["answer"][0]
                    if dataset == "cnndm":
                        if "title" in line:
                            topic = line["title"]
                        elif "question" in line:
                            topic = line["question"]
                        else: assert False, line
                        answer = line["summary"]
                    if dataset == "multi-news":
                        if "title" in line:
                            topic = line["title"]
                        elif "question" in line:
                            topic = line["question"]
                        else: assert False, line
                        answer = line["summary"]
                    if dataset == "msmarco":
                        topic = line["question"]
                        answer = line["summary"][0] if line["summary"] != [] else "Answers are unavailable."
                    curRes = resultDict[dataset][model.replace(".jsonl", "")]    # metric2scoreList
                    for metric in metrics:
                        if metric not in curRes: 
                            curRes[metric] = list()
                            length = 0
                        else: 
                            length = len(curRes[metric])
                        if dataID < length: continue
                        if metric.startswith("bleu"):
                            res = bleuScore(modelText, answer)
                        elif metric.startswith("rouge"):
                            res = rougeScore(modelText, answer)
                        elif metric.startswith("bertscore"):
                            res = bertScore(modelText, answer)
                        elif metric == "bartscore":
                            res = bartScore(modelText, answer)
                        elif metric == "QAGS_score":
                            res = QAGS(modelText, answer)
                        elif metric == "Q2_score":
                            res = Q_square(modelText, answer)
                        else: assert False, metric
                        for _metric, _score in res.items():
                            if _metric not in curRes: 
                                curRes[_metric] = list()
                            curRes[_metric].append(_score)
            with open(outputPath, 'w') as fw:
                fw.write(json.dumps(resultDict))
        print("*"*20)