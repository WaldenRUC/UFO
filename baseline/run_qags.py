import json, os, nltk, argparse, time, tqdm, argparse, logging
log_format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_format)
from baseline_utils import DatasetReader
import numpy as np
from factsumm.utils.module_entity import load_ner
from factsumm.utils.module_question import load_qa, load_qg
from rich import print
from factsumm import FactSumm
parser = argparse.ArgumentParser('QAGS')
parser.add_argument('--device', type=str)
parser.add_argument('--dataset', nargs='+', type=str)
parser.add_argument('--source_llm', nargs='+', type=str)
parser.add_argument('--prefix_input_path', type=str)
parser.add_argument('--prefix_output_path', type=str)
parser.add_argument('--metrics_output', nargs='+', type=str)
parser.add_argument('--ner_model', type=str)
parser.add_argument('--qg_model', type=str)
parser.add_argument('--qa_model', type=str)
args = parser.parse_args()



# ===== init model and dataset =====
metrics_output = {key: dict() for key in args.metrics_output}

if __name__ == '__main__':
    qags_scorer = FactSumm()
    qags_scorer.qg = load_qg(args.qg_model, args.device)
    qags_scorer.qa = load_qa(args.qa_model, args.device)
    qags_scorer.ner = load_ner(args.ner_model, args.device)
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
                qags_scores = list()
                for _llm, _answer, _human, _reference in tqdm.tqdm(zip(llm, answer, human, reference), ncols=100, desc=f'dataset: {dataset}; source: {source_llm}'):
                    ref = _answer + '\n' + _human + '\n' + _reference
                    pred = _llm
                    qags_scores.append(
                        qags_scorer.extract_qas(ref, pred, device=args.device))
                results[source_llm] = qags_scores
            output_fn = f'{args.prefix_output_path}/{metric}/{dataset}.json'
            logging.info(f'output file name: {output_fn}')
            with open(output_fn, 'w', encoding='utf-8') as fw:
                json.dump(results, fw, ensure_ascii=False, indent=2)