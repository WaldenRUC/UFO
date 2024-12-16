import json, os, nltk, argparse, time, tqdm, argparse, logging
log_format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
logging.basicConfig(level=logging.WARNING, format=log_format)
from baseline_utils import DatasetReader
import numpy as np
from rich import print
from factool import Factool
parser = argparse.ArgumentParser('factool')
parser.add_argument('--dataset', nargs='+', type=str)
parser.add_argument('--source_llm', nargs='+', type=str)
parser.add_argument('--prefix_input_path', type=str)
parser.add_argument('--prefix_output_path', type=str)
parser.add_argument('--metrics_output', nargs='+', type=str)
parser.add_argument('--evaluator', type=str)
parser.add_argument('--openai_base_url', type=str)
parser.add_argument('--openai_key', type=str)
args = parser.parse_args()


# ===== init model and dataset =====
metrics_output = {key: dict() for key in args.metrics_output}

if __name__ == '__main__':
    factool_instance = Factool(
        foundation_model=args.evaluator,
        openai_base_url=args.openai_base_url,
        openai_key=args.openai_key
    )
    
    for metric in args.metrics_output:
        logging.info(f'metric: {metric}')
        output_dir = f'{args.prefix_output_path}/{metric}'
        logging.info(f'output dir: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
        for dataset in args.dataset:
            input_fn = f'{args.prefix_input_path}/{dataset}.json'
            output_fn = f'{args.prefix_output_path}/{metric}/{dataset}.json'
            logging.info(f'input file name: {input_fn}')
            logging.info(f'output file name: {output_fn}')
            if os.path.exists(output_fn):
                with open(output_fn, 'r', encoding="utf-8") as f:
                    results = json.load(f)
            else: results = dict()
            for source_llm in args.source_llm:
                llm, answer, human, reference, se = DatasetReader(input_fn, extractor='chatgpt-extractor', source_llm=source_llm, return_se=True)
                ########
                refs, preds = list(), list()
                if source_llm in results:
                    factool_scores = results[source_llm]
                    generated_count = len(factool_scores)
                else: 
                    factool_scores = list()
                    generated_count = 0
                for _idx, (_llm, _answer, _human, _reference, _se) in enumerate(tqdm.tqdm(zip(llm, answer, human, reference, se), ncols=100, desc=f'dataset: {dataset}; source: {source_llm}')):
                    if _idx < generated_count: 
                        print('generated!')
                        continue
                    if _idx >= 60:
                        print(f'exceed idx: {_idx}')
                        continue
                    ref = _answer + '\n' + _human + '\n' + _reference
                    pred = _llm
                    inputs = [
                        {
                            "prompt": '',
                            "response": pred,
                            "category": "kbqa",
                            "search_cache": _se     # [["query", "se"], []]
                        },
                    ]
                    logging.warning(pred)
                    time1 = time.time()
                    factool_scores.append(factool_instance.run(inputs))
                    logging.warning(f'time spent: {time.time() - time1}')
                    results[source_llm] = factool_scores
                    with open(output_fn, 'w', encoding='utf-8') as fw:
                        json.dump(results, fw, ensure_ascii=False, indent=2)