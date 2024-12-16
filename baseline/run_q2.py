import json, os, nltk, argparse, time, tqdm, argparse, logging, torch
log_format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_format)
from baseline_utils import DatasetReader, Q2
import numpy as np
from rich import print

parser = argparse.ArgumentParser('Q2')
parser.add_argument('--device', type=str)
parser.add_argument('--dataset', nargs='+', type=str)
parser.add_argument('--source_llm', nargs='+', type=str)
parser.add_argument('--prefix_input_path', type=str)
parser.add_argument('--prefix_output_path', type=str)
parser.add_argument('--metrics_output', nargs='+', type=str)
parser.add_argument('--qg_model', type=str)
parser.add_argument('--qa_model', type=str)
parser.add_argument('--nli_model', type=str)
parser.add_argument('--en_core_web_sm_model', type=str)
args = parser.parse_args()

# ===== init model and dataset =====
metrics_output = {key: dict() for key in args.metrics_output}

    
if __name__ == '__main__':
    q2_scorer = Q2(
        qg_model = args.qg_model,
        qa_model = args.qa_model,
        nli_model = args.nli_model,
        en_core_web_sm_model = args.en_core_web_sm_model,
        device = args.device
    )
    for metric in args.metrics_output:
        logging.info(f'metric: {metric}')
        output_dir = f'{args.prefix_output_path}/{metric}'
        logging.info(f'output dir: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        for dataset in args.dataset:
            input_fn = f'{args.prefix_input_path}/{dataset}.json'
            logging.info(f'input file name: {input_fn}')
            results = dict()
            for source_llm in args.source_llm:
                llm, answer, human, reference = DatasetReader(input_fn, extractor='chatgpt-extractor', source_llm=source_llm)
                ########
                refs, preds = list(), list()
                q2_scores = list()
                for _llm, _answer, _human, _reference in tqdm.tqdm(zip(llm, answer, human, reference), ncols=100, desc=f'dataset: {dataset}; source: {source_llm}'):
                    ref = _answer + '\n' + _human + '\n' + _reference
                    pred = _llm
                    q2_score = q2_scorer.score(ref, pred)
                    q2_scores.append(q2_score)
                    logging.info(f'q2 score: {q2_score}')
                results[source_llm] = q2_scores
            output_fn = f'{args.prefix_output_path}/{metric}/{dataset}.json'
            logging.info(f'output file name: {output_fn}')
            with open(output_fn, 'w', encoding='utf-8') as fw:
                json.dump(results, fw, ensure_ascii=False, indent=2)
