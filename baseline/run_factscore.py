import json, os, nltk, argparse, time, tqdm, argparse, logging, openai
log_format = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
logging.basicConfig(level=logging.WARNING, format=log_format)
from baseline_utils import DatasetReader
from factscore.factscorer import FactScorer
import numpy as np
from rich import print
parser = argparse.ArgumentParser('factscore')
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
    openai.api_base = args.openai_base_url
    openai.api_key = args.openai_key
    fs = FactScorer()
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
                llm, answer, human, reference, se, lk = DatasetReader(input_fn, extractor='chatgpt-extractor', source_llm=source_llm, return_se=True, return_lk=True)
                ########
                refs, preds = list(), list()
                if source_llm in results:
                    factscore_scores = results[source_llm]
                    generated_count = len(factscore_scores)
                else: 
                    factscore_scores = list()
                    generated_count = 0
                for _idx, (_llm, _answer, _human, _reference, _se, _lk) in enumerate(tqdm.tqdm(zip(llm, answer, human, reference, se, lk), ncols=100, desc=f'dataset: {dataset}; source: {source_llm}')):
                    if _idx < generated_count: 
                        print('generated!')
                        continue
                    if _idx >= 60:
                        print(f'exceed idx: {_idx}')
                        continue
                    se_cleaned = list()
                    for fact in _se:
                        for item in fact:
                            if 'answerBox' in item['se']:
                                se_cleaned.append(item['se']['answerBox'].get('snippet', ''))
                            else: pass
                    # print(f'answer: {_answer}\nhuman: {_human}\nreference: {_reference}\nse: {se_cleaned}\nlk: {_lk}\nllm: {_llm}')
                    ref = _answer + '\n' + _human + '\n' + _reference + '\n' + '\n'.join(se_cleaned) + '\n' + _lk
                    pred = _llm
                    print("*"*20)
                    print(pred)
                    print("*"*20)
                    time1 = time.time()
                    # now, when you compute a score, specify knowledge source to use
                    out = fs.get_score(
                        [''], 
                        [pred], 
                        sample_knowledge_source=ref)
                    logging.warning(f'time spent: {time.time() - time1}')
                    logging.warning(f'score: {out["score"]}') # FActScore
                    logging.warning(f'respond ratio: {out["respond_ratio"]}') # % of responding (not abstaining from answering)
                    logging.warning(f'num facts per response: {out["num_facts_per_response"]}') # average number of atomic facts per response
                    factscore_scores.append(out)
                    results[source_llm] = factscore_scores
                    with open(output_fn, 'w', encoding='utf-8') as fw:
                        json.dump(results, fw, ensure_ascii=False, indent=2)