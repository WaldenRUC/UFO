import json, os, argparse
import numpy as np
from tqdm import tqdm
from utils import UFO

parser = argparse.ArgumentParser("UFO: unified and flexible framework for factuality evaluation. EMNLP'24 submission")
parser.add_argument('--phase', choices=['fue', 'fsv', 'fcd'], type=str)
parser.add_argument('--scenario', nargs='+', choices=['he', 'rd', 'se', 'lk'])
parser.add_argument('--source_llm', nargs='+', choices=['newbing', 'gpt0125', 'llama2-7B', 'llama2-13B', 'llama2-70B', 'llama3-8B', 'llama3-70B', 'Qwen-7B', 'Qwen-14B'])
parser.add_argument('--evaluator', choices=['llama-3-8b-instruct', 'gpt-3.5-turbo-0125'])
parser.add_argument('--prompt_prefix', type=str)
parser.add_argument('--question_prefix', type=str)
parser.add_argument('--fact_extraction_prefix', type=str)
parser.add_argument('--fact_extraction_model', choices=['llama3', 'chatgpt'])
parser.add_argument('--verification_prefix', type=str)
parser.add_argument('--se_prefix', type=str)
parser.add_argument('--lk_prefix', type=str)
parser.add_argument('--dataset', choices=['nq', 'hotpotqa', 'truthfulqa', 'cnndm', 'multinews', 'msmarco'])
args = parser.parse_args()


if __name__ == "__main__":
    ufo = UFO(
        phase = args.phase,
        scenario = args.scenario,
        source_llm = args.source_llm,
        evaluator = args.evaluator,
        prompt_prefix = args.prompt_prefix,
        question_prefix = args.question_prefix,
        fact_extraction_prefix = args.fact_extraction_prefix,
        fact_extraction_model = args.fact_extraction_model,
        verification_prefix = args.verification_prefix,
        se_prefix = args.se_prefix,
        lk_prefix = args.lk_prefix,
        dataset = args.dataset
    )