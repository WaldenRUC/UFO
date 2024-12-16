import json, os, argparse
import numpy as np
from tqdm import tqdm
from utils import UFO

parser = argparse.ArgumentParser("UFO: unified and flexible framework for factuality evaluation")
parser.add_argument('--dataset')
parser.add_argument('--extractor', choices=['llama3', 'chatgpt'])
parser.add_argument('--source_llm', choices=['newbing', 'gpt0125', 'llama2-7B', 'llama2-13B', 'llama2-70B', 'llama3-8B', 'llama3-70B', 'Qwen-7B', 'Qwen-14B'])
parser.add_argument('--scenario', nargs='+', choices=['hu', 're', 'se', 'lk'])
parser.add_argument('--evaluator', choices=['llama3', 'chatgpt'])
parser.add_argument('--prompt_prefix', type=str)
parser.add_argument('--output_path', type=str)
args = parser.parse_args()


if __name__ == "__main__":
    ufo = UFO(
        dataset = args.dataset,
        extractor = args.extractor,
        source_llm = args.source_llm,
        scenario = args.scenario,
        evaluator = args.evaluator,
        prompt_prefix = args.prompt_prefix,
        output_path = args.output_path
    )