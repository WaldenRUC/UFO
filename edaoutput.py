import os, json, pysbd, spacy
import numpy as np
from rich import print
from tqdm import tqdm
nlp = spacy.load("en_core_web_sm")
segmenter = pysbd.Segmenter(language="en", clean=False)
def segment(text: str):
    return [line.strip() for line in segmenter.segment(text)]
prefix = "/home/dou/LLMFE/output"
fileIDList = range(200)
def calScore(nlp, answer, properAns):
    # 计算Prec或F1得分
    answer = list(map(lambda x: x.text, nlp(answer)))
    properAns = list(map(lambda x: x.text, nlp(properAns))) 
    overlap = len(set(answer) & set(properAns))
    if len(answer) == 0 or len(properAns) == 0 or overlap == 0: return 0
    prec = overlap / len(answer)
    rec = overlap / len(properAns)
    # return round(2 * prec * rec / (prec + rec), ndigits=3)      # F1
    return round(prec, ndigits=3)     # Prec
Dataset = ["NQ", "HotpotQA", "truthfulQA", "cnndm", "multi-news", "msmarco"]
VerifyOrder = ["None-None", "None-ref", "sum-None", "sum-ref", "sum-ref", "sum-ref"]
VerifyOrder_newbing = [item+"-normal" for item in VerifyOrder]
Model = ["newbing", "chatgpt", "llama7b", "llama13b", "vicuna7b", "vicuna13b"]
Extractor = "ChatGPT"
# Checker = "llm+se"
Checker = "se"






def levenshtein_distance(str1, str2):
    if len(str1) > len(str2):
        str1, str2 = str2, str1
    distances = range(len(str1) + 1)
    for index2, char2 in enumerate(str2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(str1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
        distances = new_distances
    distance = distances[-1]
    similarity = (1 - distance / max(len(str1), len(str2))) * 100
    return round(similarity, 3)




def getInfo(DataExtractor, DataEval, DataFinal):
    res = dict()
    res["title"] = DataExtractor["generatedTitle"]
    res["modelText"], res["score"] = DataFinal["modelText"], DataFinal["avgScore"]
    res["modelText"] = segment(res["modelText"])
    res["SentsCount"] = len(res["modelText"])
    res["SentScore"] = list()
    F1score = list()
    for line in DataEval:
        oriSent, score, phrase, properAns, question = line["oriSent"], line["score"], line["phrase"], line["properAns"], line["question"]
        res["SentScore"].append({
            "oriSent": oriSent,
            "question": question,
            "phrase": phrase, 
            "properAns": properAns, 
            "score": score
        })
        F1score.append(calScore(nlp, phrase, properAns))

        
        simScores = [levenshtein_distance(factDict["oriSent"], modelTextLine) for modelTextLine in modelTextLines]
        max_index = simScores.index(max(simScores))
        sent2scorelist[modelTextLines[max_index]].append(factDict["score"])
    res["F1scoreList"] = F1score
    return res



if __name__ == '__main__':
    print(Extractor, Checker)
    for modelID, model in enumerate(Model):
        for datasetID, dataset in enumerate(Dataset):
            scoreList = list()
            datasetPrefix = os.path.join(prefix, dataset)
            for fileID in tqdm(fileIDList, desc=f"model: {model}, dataset: {dataset}"):
                ExtractorPostfix = os.path.join(str(fileID), f"factExtract-{Extractor}.json")
                if model != "newbing":
                    EvaluatorPostfix = os.path.join(str(fileID), f"factEvaluator-{Extractor}-PLM-{Checker}-{VerifyOrder[datasetID]}_v2.jsonl")
                    FinalPostfix = os.path.join(str(fileID), f"final-{Extractor}-PLM-{Checker}-{VerifyOrder[datasetID]}.json")
                else:
                    EvaluatorPostfix = os.path.join(str(fileID), f"factEvaluator-{Extractor}-PLM-{Checker}-{VerifyOrder_newbing[datasetID]}_v2.jsonl")
                    FinalPostfix = os.path.join(str(fileID), f"final-{Extractor}-PLM-{Checker}-{VerifyOrder_newbing[datasetID]}.json")
                ExtractorFile = os.path.join(datasetPrefix, model, ExtractorPostfix)
                EvaluatorFile = os.path.join(datasetPrefix, model, EvaluatorPostfix)
                FinalFile = os.path.join(datasetPrefix, model, FinalPostfix)
                with open(ExtractorFile, 'r', encoding='utf-8') as fp:
                    DataExtractor = json.load(fp)
                with open(EvaluatorFile, 'r', encoding='utf-8') as fp:
                    DataEval = list()
                    for line in fp.readlines():
                        DataEval.append(json.loads(line))
                with open(FinalFile, 'r', encoding='utf-8') as fp:
                    DataFinal = json.load(fp)
                res = getInfo(DataExtractor, DataEval, DataFinal)
            print(f"model: {model}, dataset: {dataset}, score: {round(np.mean(res['F1scoreList']), 3)}")
        print("*"*20)