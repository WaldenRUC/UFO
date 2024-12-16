import json, os, nltk, argparse, time, tqdm, argparse, sys, logging
log_format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_format)
from bartscore.bart_score import BARTScorer
from baseline_utils import DatasetReader
import numpy as np
from rich import print
parser = argparse.ArgumentParser('BARTScore')
parser.add_argument('--device', type=str)
parser.add_argument('--dataset', nargs='+', type=str)
parser.add_argument('--source_llm', nargs='+', type=str)
parser.add_argument('--prefix_input_path', type=str)
parser.add_argument('--prefix_output_path', type=str)
parser.add_argument('--metrics_output', nargs='+', type=str)
parser.add_argument('--bart_model', type=str)
parser.add_argument('--bart_ckpt', type=str)
args = parser.parse_args()

# ===== init model and dataset =====
metrics_output = {key: dict() for key in args.metrics_output}


if __name__ == '__main__':
    bart_scorer = BARTScorer(device=args.device, checkpoint=args.bart_model)
    bart_scorer.load(args.bart_ckpt)
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
                for _llm, _answer, _human, _reference in tqdm.tqdm(zip(llm, answer, human, reference), ncols=100, desc=f'dataset: {dataset}; source: {source_llm}'):
                    ref = _answer + '\n' + _human + '\n' + _reference
                    pred = _llm
                    refs.append(ref)
                    preds.append(pred)
                bart_scores = bart_scorer.score(preds, refs, batch_size=16)
                results[source_llm] = bart_scores
            output_fn = f'{args.prefix_output_path}/{metric}/{dataset}.json'
            logging.info(f'output file name: {output_fn}')
            with open(output_fn, 'w', encoding='utf-8') as fw:
                json.dump(results, fw, ensure_ascii=False, indent=2)