from bert_score.scorer import BERTScorer
import json, os, argparse, time, tqdm, argparse, logging
import numpy as np
from baseline_utils import DatasetReader
log_format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=log_format)
parser = argparse.ArgumentParser('BERTScore p,r,f1')
parser.add_argument('--device', type=str)
parser.add_argument('--dataset', nargs='+', type=str)
parser.add_argument('--source_llm', nargs='+', type=str)
parser.add_argument('--prefix_input_path', type=str)
parser.add_argument('--prefix_output_path', type=str)
parser.add_argument('--metrics_output', nargs='+', type=str)
parser.add_argument('--bert_model', type=str)
args = parser.parse_args()


# ===== init model and dataset =====
metrics_output = {key: dict() for key in args.metrics_output}


if __name__ == '__main__':
    scorer = BERTScorer(
        model_type = args.bert_model,
        num_layers = 17,                 # xlm-roberta-large:layers = 17
        device = args.device
    )
    # ===== get output dirs =====
    for metric in args.metrics_output:
        logging.info(f'metric: {metric}')
        output_dir = f'{args.prefix_output_path}/{metric}'
        logging.info(f'output dir: {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
    for dataset in args.dataset:
        input_fn = f'{args.prefix_input_path}/{dataset}.json'
        logging.info(f'input file name: {input_fn}')
        p_results, r_results, f1_results = dict(), dict(), dict()
        for source_llm in args.source_llm:
            llm, answer, human, reference = DatasetReader(input_fn, extractor='chatgpt-extractor', source_llm=source_llm)
            ########
            refs, preds = list(), list()
            for _llm, _answer, _human, _reference in tqdm.tqdm(zip(llm, answer, human, reference), ncols=100, desc=f'dataset: {dataset}; source: {source_llm}'):
                ref = _answer + '\n' + _human + '\n' + _reference
                pred = _llm
                refs.append(ref)
                preds.append(pred)
            bertscore_p, bertscore_r, bertscore_f1 = [item.tolist() for item in scorer.score(preds, refs)]
            p_results[source_llm], r_results[source_llm], f1_results[source_llm] = bertscore_p, bertscore_r, bertscore_f1
            
        for metric, results in zip(args.metrics_output, [p_results, r_results, f1_results]):
            output_fn = f'{args.prefix_output_path}/{metric}/{dataset}.json'
            logging.info(f'output file name: {output_fn}')
            with open(output_fn, 'w', encoding='utf-8') as fw:
                json.dump(results, fw, ensure_ascii=False, indent=2)