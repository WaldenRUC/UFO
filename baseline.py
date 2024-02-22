'''
BLEU, ROUGE-1/2/L
BERTScore-p/r/F1, BARTScore
QAGS, Q^2
FactScore
FacTool
'''
import json, os, nltk, argparse, time
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HTTP_PROXY"]="http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"]="http://127.0.0.1:7890"
os.environ["ALL_PROXY"]="socks5://127.0.0.1:7890"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["SERPER_API_KEY"] = "084fe3523c22841b16bb8f84537e51baebc9d859"
nltk.download('punkt')

parser = argparse.ArgumentParser()
import numpy as np
from flair.data import Sentence
from flair.models import SequenceTagger
tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast")
from rich import print
from tqdm import tqdm
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')


parser.add_argument('--metrics', nargs='+', type=str)
parser.add_argument('--filenum', default=200, type=int)
args = parser.parse_args()
fileNum = args.filenum


def truncate(text: str, max_length = 256):
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token != "[UNK]"][:max_length]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokenizer.decode(token_ids)


# BLEU
if "bleu" in args.metrics:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    def tokenize(text):
        return nltk.word_tokenize(text)
def bleuScore(pred: str, ref: str):
    ref = [tokenize(ref.lower())]
    pred = tokenize(pred.lower())
    smoothing = SmoothingFunction().method1
    score_bleu1 = round(sentence_bleu(ref, pred, weights=(1, 0, 0, 0)), 3)
    score_bleu2 = round(sentence_bleu(ref, pred, weights=(0.5, 0.5, 0, 0)), 3)
    score_bleu3 = round(sentence_bleu(ref, pred, weights=(0.33, 0.33, 0.33, 0)), 3)
    score_bleu4 = round(sentence_bleu(ref, pred, smoothing_function=smoothing), 3)
    return {
        "bleu1": score_bleu1, 
        "bleu2": score_bleu2, 
        "bleu3": score_bleu3, 
        "bleu4": score_bleu4
    }


# ROUGE
if "rouge" in args.metrics:
    from rouge_score import rouge_scorer
    rouge_scorer_instance = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
def rougeScore(pred: str, ref: str):
    '''
    str1: model
    str2: 
    '''
    res = rouge_scorer_instance.score(ref, pred)
    return {
        "rouge1": round(res["rouge1"][1], 3),     # precision, recall, f1-measure
        "rouge2": round(res["rouge2"][1], 3),     # precision, recall, f1-measure
        "rougeL": round(res["rougeL"][1], 3)      # precision, recall, f1-measure
    }


# BERTScore
if "bertscore" in args.metrics:
    from bert_score import score as bert_scorer
def bertScore(pred: str, ref: str, model_type="roberta-large"):
    assert type(pred) == type(ref)
    if type(pred) == str:
        pred, ref = [pred], [ref]
        P, R, F1 = bert_scorer(pred, ref, model_type=model_type, lang='en', verbose=True)
        return {
            "bertscore-p": round(float(P), 3),
            "bertscore-r": round(float(R), 3),
            "bertscore-f1": round(float(F1), 3)
        }
    elif type(pred) == list:
        P, R, F1 = bert_scorer(pred, ref, model_type=model_type, lang='en', verbose=True)
        results = list()
        for p, r, f1 in zip(P, R, F1):
            results.append({
                "bertscore-p": round(float(p), 3),
                "bertscore-r": round(float(r), 3),
                "bertscore-f1": round(float(f1), 3)
            })
        return results
    else: assert False, type(pred)


# BARTScore
if "bartscore" in args.metrics:
    from BARTScore.bart_score import BARTScorer
    bart_scorer = BARTScorer(device='cuda:0', checkpoint='facebook/bart-large-cnn')
def bartScore(pred: str, ref: str):
    if type(pred) == str: pred = [pred]
    if type(ref) == str: ref = [ref]
    bartscore = bart_scorer.score(pred, ref, batch_size=4)
    return {
        "bartscore": round(bartscore[0], 3)
    }


# QAGS
if "qags" in args.metrics:
    from factsumm import FactSumm
    factsumm = FactSumm()
def QAGS(cand: str, ref: str):
    QAGS_score = factsumm.extract_qas(ref, cand)
    return {
        "QAGS_score": round(QAGS_score, 3)
    }


# Q2
if "q2" in args.metrics:
    from Qsquared.pipeline.run_pipeline import calc_scores, calc_scores_nofile
def Q_square(cand: str, ref: str):
    print("calculate q2...")
    process, Q2_score = calc_scores_nofile(
        response=cand, knowledge=ref,
        gen_method="beam", single=True, remove_personal=True
    )
    # 读取数据
    print(f"process: {process};\nQ2_score: {Q2_score}")
    # invalid results
    if len(Q2_score) == 0: Q2_score = [0]   
    return {
        "process": process,
        "Q2_score": round(Q2_score[0], 3)
    }


# FactScore
if "factscore" in args.metrics:
    from factscore.factscorer import FactScorer
    fs = FactScorer(
        data_dir="/home/dou/factscoreData",
        openai_key="/home/dou/factscoreData/openai_key.txt"
        )
    fs.register_knowledge_source(
        "factscoreKS",
        data_path = "factscoreKS.jsonl",
        db_path = "factscoreKS.db"
    )
def factScore(pred: str, topic: str):
    response = fs.get_score([topic], [pred], knowledge_source="factscoreKS")
    return { 
        "process": response,
        "factscore": response["score"]
    }




# Factool
if "factool" in args.metrics:
    from factool import Factool
    factool_instance = Factool("gpt-3.5-turbo-1106")
def facTool(pred: str):
    inputs = [{
            "prompt": None,
            "response": pred,
            "category": "kbqa"}, ]
    try:
        response_list = factool_instance.run(inputs)
        print(response_list)
    except TypeError:
        print("No Claims in The Response!!")
        return {
            "factool": 0.0
        }
    return {
        "response_list": response_list,
        "factool": response_list["average_response_level_factuality"]
    }











def storeFactScoreJsonl(in_path: str = None, out_path: str = None):
    '''
    out_path: jsonl格式的输出文件。each line has keys: title and [text]
    '''
    lines = dict()
    with open(in_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
    for model, model_dict in data.items():
        for dataset, dataset_list in model_dict.items():
            for question, modelText, evidence in dataset_list:
                lines[question] = evidence
    lines = [{"title": k, "text": v} for k, v in lines.items()]
    with open(out_path, 'w') as fw:
        for line in lines:
            fw.write(json.dumps(line) + "\n")

def loadData(models: list, datasets: list, file_prefix = "/home/dou/UniFlexFactor/dataset", truncate_max: int = 256) -> dict:
    '''
    [<model>][<dataset>] -> (question, modelText, evidence)
    '''
    responses = dict()
    for model in models:
        responses[model.split('.')[0]] = dict()
        for dataset in tqdm(datasets, desc=f"[{model}]"):
            responses[model.split('.')[0]][dataset] = list()
            with open(os.path.join(file_prefix, dataset, model), 'r', encoding='utf-8') as fp:
                lines = json.load(fp)[:200]
            for line in lines:
                if dataset == "NQ":
                    question = line["prompt"].split("\n\n")[1].split("Title: ")[1]
                    evidence = line["answer"][0]
                elif dataset == "HotpotQA":
                    question = line["question"]
                    evidence = f'{line["answer"]}. {" ".join(line["document"])}'
                elif dataset == "truthfulQA":
                    question = line["question"]
                    evidence = line["summary"]
                elif dataset == "cnndm":
                    if "title" in line:
                        question = line["title"]
                    elif "question" in line:
                        question = line["question"]
                    evidence = f'{line["summary"]} {line["document"]}'
                elif dataset == "multi-news":
                    if "title" in line:
                        question = line["title"]
                    elif "question" in line:
                        question = line["question"]
                    evidence = f'{line["summary"]} {line["document"]}'
                elif dataset == "msmarco":
                    question = line["question"]
                    summary, document = ' '.join(line["summary"]), ' '.join(line["document"])
                    evidence = f"{summary} {document}".strip()
                    if evidence == "":
                        evidence = "Answers are unavailable."
                else: assert False, model
                responses[model.split('.')[0]][dataset].append(
                    (question, line["modelText"], truncate(evidence, max_length=truncate_max))
                )
    return responses
            

if __name__ == "__main__":


    Model = ["newbing.json", "chatgpt.json", "llama7b-hf.json", "llama13b-hf.json", "vicuna7b-hf.json", "vicuna13b-hf.json"]
    Dataset = ["NQ", "HotpotQA", "truthfulQA", "cnndm", "multi-news", "msmarco"]
    if not os.path.exists("questions_responses_evidence.json"):
        responses = loadData(Model, Dataset)
        with open("questions_responses_evidence.json", 'w') as fw:
            fw.write(json.dumps(responses))
    with open("questions_responses_evidence.json", 'r', encoding='utf-8') as fp:
        responses = json.load(fp)


    # storeFactScoreJsonl(in_path="questions_responses_evidence-full.json", out_path="factscoreKS.jsonl")
    # quit()



    for metric in args.metrics:
        if os.path.exists(f"baseline-{metric}.json"):
            with open(f"baseline-{metric}.json", 'r', encoding='utf-8') as fp:
                results = json.load(fp)
        else: 
            results = dict()
        for model, model_dict in responses.items():
            if model not in results: results[model] = dict()
            for dataset, dataset_lines in model_dict.items():
                if dataset not in results[model]: results[model][dataset] = list()
                dataset_lines = dataset_lines[len(results[model][dataset]):fileNum]
                if metric == "bertscore":
                    modelTexts, evidences = list(), list()
                for question, modelText, evidence in tqdm(dataset_lines, desc=f"{model}; {dataset}"):
                    question, modelText, evidence = question.strip(), modelText.strip(), evidence.strip()
                    if metric == "bleu":
                        results[model][dataset].append(bleuScore(modelText, evidence))
                    elif metric == "rouge":
                        results[model][dataset].append(rougeScore(modelText, evidence))
                    elif metric == "bertscore":
                        modelTexts.append(modelText)
                        evidences.append(evidence)
                    elif metric == "bartscore":
                        results[model][dataset].append(bartScore(modelText, evidence))
                    elif metric == "qags":
                        results[model][dataset].append(QAGS(modelText, evidence))
                    elif metric == "q2":
                        results[model][dataset].append(Q_square(modelText, evidence))
                    elif metric == "factool":
                        results[model][dataset].append(facTool(modelText))
                    elif metric == "factscore":
                        results[model][dataset].append(factScore(modelText, question))
                    else: assert False, metric
                    # 处理完一个样本就保存一次
                    with open(f"baseline-{metric}.json", 'w') as fw:
                        fw.write(json.dumps(results))
                if metric == "bertscore":
                    if len(modelTexts) != 0 and len(evidences) != 0:
                        results[model][dataset].extend(bertScore(modelTexts, evidences))
        with open(f"baseline-{metric}.json", 'w') as fw:
            fw.write(json.dumps(results))

        
        
    #         fileName = os.path.join(prefix, dataset, model)
    #         with open(fileName, 'r', encoding='utf-8') as fp:
    #             data = fp.readlines()[writtenLength:fileNum]
    #             for dataID, line in enumerate(tqdm(data, desc=f"{dataset}, {model}")):
    #                 line = json.loads(line)
    #                 modelText = line["modelText"]
    #                 if dataset == "NQ":
    #                     topic = line["prompt"].split("Title: ")[1].split("Introduction:")[0].strip()
    #                     answer = line["answer"][0]
    #                 elif dataset == "HotpotQA":
    #                     topic = line["question"]
    #                     answer = line["answer"]
    #                 elif dataset == "truthfulQA":
    #                     topic = line["question"]
    #                     answer = line["answer"][0]
    #                 elif dataset == "cnndm":
    #                     if "title" in line:
    #                         topic = line["title"]
    #                     elif "question" in line:
    #                         topic = line["question"]
    #                     else: assert False, line
    #                     answer = line["summary"]
    #                 elif dataset == "multi-news":
    #                     if "title" in line:
    #                         topic = line["title"]
    #                     elif "question" in line:
    #                         topic = line["question"]
    #                     else: assert False, line
    #                     answer = line["summary"]
    #                 elif dataset == "msmarco":
    #                     topic = line["question"]
    #                     answer = line["summary"][0] if line["summary"] != [] else "Answers are unavailable."
    #                 else: assert False, dataset
                    
    #                 if metric == "bleu":
    #                     res = bleuScore(modelText, answer)
    #                 elif metric == "rouge":
    #                     res = rougeScore(modelText, answer)
    #                 elif metric == "bertscore":
    #                     res = bertScore(modelText, answer)
    #                 elif metric == "bartscore":
    #                     res = bartScore(modelText, answer)
    #                 elif metric == "QAGS_score":
    #                     res = QAGS(modelText, answer)
    #                 elif metric == "Q2_score":
    #                     res = Q_square(modelText, answer)
    #                 elif metric == "factscore":
    #                     res = factScore(topic, modelText, None)
    #                 else: assert False, metric
    #                 for _metric, _score in res.items():
    #                     resultDict[_metric].append(_score)
    #                 os.makedirs(os.path.join("output", metric, model.split(".")[0]), exist_ok = True)
    #                 with open(outputPath, 'w') as fw:
    #                     fw.write(json.dumps(resultDict) + "\n")
            


    # #     fileName = os.path.join(prefix, dataset, model)
    # #     resultList = list()
    # #     with open(fileName, 'r', encoding='utf-8') as fp:
    # #         data = fp.readlines()[:fileNum]
    # #         for dataID, line in enumerate(tqdm(data, desc=f"{dataset}, {model}")):
    # #             line = json.loads(line)
    # #             modelText = line["modelText"]
    # #             if dataset == "NQ":
    # #                 topic = line["prompt"].split("Title: ")[1].split("Introduction:")[0].strip()
    # #                 answer = line["answer"][0]
    # #             if dataset == "HotpotQA":
    # #                 topic = line["question"]
    # #                 answer = line["answer"]
    # #             if dataset == "truthfulQA":
    # #                 topic = line["question"]
    # #                 answer = line["answer"][0]
    # #             if dataset == "cnndm":
    # #                 if "title" in line:
    # #                     topic = line["title"]
    # #                 elif "question" in line:
    # #                     topic = line["question"]
    # #                 else: assert False, line
    # #                 answer = line["summary"]
    # #             if dataset == "multi-news":
    # #                 if "title" in line:
    # #                     topic = line["title"]
    # #                 elif "question" in line:
    # #                     topic = line["question"]
    # #                 else: assert False, line
    # #                 answer = line["summary"]
    # #             if dataset == "msmarco":
    # #                 topic = line["question"]
    # #                 answer = line["summary"][0] if line["summary"] != [] else "Answers are unavailable."
    # #             curRes = resultDict[dataset][model.replace(".json", "")]    # metric2scoreList
    # #             for metric in metrics:
    # #                 if metric not in curRes: 
    # #                     curRes[metric] = list()
    # #                     length = 0
    # #                 else: 
    # #                     length = len(curRes[metric])
    # #                 if dataID < length: continue