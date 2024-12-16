import numpy as np
from rich import print
import json, os, nltk, argparse, time, tqdm, argparse, logging, sys
log_format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_format)
from baseline_utils import DatasetReader
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def tokenize(text):
    return nltk.word_tokenize(text)

    
parser = argparse.ArgumentParser('bleu and rouge')
parser.add_argument('--dataset', nargs='+', type=str)
parser.add_argument('--source_llm', nargs='+', type=str)
parser.add_argument('--prefix_input_path', type=str)
parser.add_argument('--prefix_output_path', type=str)
parser.add_argument('--metrics_output', nargs='+', type=str)
args = parser.parse_args()

# ===== init model and dataset =====
metrics_output = {key: dict() for key in args.metrics_output}

# ===== init metrics =====
rouge_scorer_instance = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
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

# 
# baseline_results/bleu1/nq.json
'''
{
    'newbing': [...],
    'gpt0125': [...], ...
}
'''
if __name__ == '__main__':
    for metric in args.metrics_output:
        output_dir = f'{args.prefix_output_path}/{metric}'
        logging.info(f'metric: {metric}')
        logging.info(f'output dir: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        for dataset in args.dataset:
            input_fn = f'{args.prefix_input_path}/{dataset}.json'
            logging.info(f'input file name: {input_fn}')
            output_fn = f'{output_dir}/{dataset}.json'
            logging.info(f'output file name: {output_fn}')
            results = dict()
            for source_llm in args.source_llm:
                llm, answer, human, reference = DatasetReader(input_fn, extractor='chatgpt-extractor', source_llm=source_llm)
                ########
                eval_results = list()
                for _llm, _answer, _human, _reference in tqdm.tqdm(zip(llm, answer, human, reference), ncols=100, desc=f'dataset: {dataset}; source: {source_llm}'):
                    ref = _answer + '\n' + _human + '\n' + _reference
                    pred = _llm
                    bleu_score = bleuScore(pred, ref)
                    if metric in bleu_score:
                        eval_results.append(bleu_score[metric])
                    else:
                        rouge_score = rougeScore(pred, ref)
                        if metric in rouge_score:
                            eval_results.append(rouge_score[metric])
                ########
                results[source_llm] = eval_results
            with open(output_fn, 'w', encoding='utf-8') as fw:
                json.dump(results, fw, ensure_ascii=False, indent=2)
            
    
    
    
        
#     for _metric in metrics_output:
#         metrics_output[_metric][dataset] = {model: list() for model in Model}
#     for model in Model:
#         input_file_path = f'{args.prefix_input_path}/{dataset}/{model}.jsonl'
#         with open(input_file_path, 'r', encoding='utf-8') as fp:
#             data = fp.readlines()
#         for line in tqdm.tqdm(data, desc=f'{dataset}; {model}', ncols=120):
#             line = json.loads(line)
#             pred = line['mgt']
#             if dataset == 'nq':
#                 ref = line['answer'][0]
#             elif dataset == 'hotpotqa':
#                 ref = line['answer'] + '\n' + '\n'.join(line['reference_documents'])
#             elif dataset == 'truthfulqa':
#                 ref = line['human_written_evidences'][0]
#             elif dataset == 'msmarco':
#                 ref = '\n'.join(line['human_written_evidences'] + line['reference_documents'])
#             else:
#                 ref = line['answer']
#             bleu_score = bleuScore(pred, ref)
#             for _metric, value in bleu_score.items():
#                 metrics_output[_metric][dataset][model].append(value)
#             rouge_score = rougeScore(pred, ref)
#             for _metric, value in rouge_score.items():
#                 metrics_output[_metric][dataset][model].append(value)
# for _metric in metrics_output:
#     with open(f'{args.prefix_output_path}/{_metric}.json', 'w', encoding='utf-8') as fw:
#         fw.write(json.dumps(metrics_output[_metric], ensure_ascii=False) + '\n')