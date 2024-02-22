'''
benchmark(Table 1);
baseline;
output
'''
import os, json, pysbd, sys, krippendorff
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from typing import List
from rich import print
from utils import CorrelationStatistics, Benchmark
sys.path.append("/home/dou/UniFlexFactor/baseline/PDP")
from util import PDP
from data import RelevanceJudgments, PreferenceJudgments


if __name__ == "__main__":
    # Benchmark for Table 1
    # benchmark = Benchmark(
    #     Model = ["newbing", "chatgpt", "llama7b-hf", "llama13b-hf", "vicuna7b-hf", "vicuna13b-hf"],
    #     Dataset = ["NQ", "HotpotQA", "truthfulQA", "cnndm", "multi-news", "msmarco"],
    #     tokenizer_path="/home/dou/dataset/vicuna/7b/vicuna-7b-v1.5",
    #     dataset_path="/home/dou/UniFlexFactor/dataset",
    #     output_path="/home/dou/UniFlexFactor/output",
    #     output_filenames=["ChatGPT-llmse-normal", "ChatGPT-llmse-nosource", "ChatGPT-llmse-reverse"],
    #     file_num = 200
    # )
    # data = benchmark.loadOutput()
    # benchmark.benchmark_res("StatTokens")
    # benchmark.benchmark_res("StatSents")
    # benchmark.benchmark_res("StatFacts-ChatGPT")
    # benchmark.benchmark_res("StatFacts-PLM")

    # Baselines and analysis of outputs
    corrStats = CorrelationStatistics(
        baselines=["bleu", "rouge", "bertscore", "bartscore", "qags", "q2", "factool", "factscore"],
        Model = ["newbing", "chatgpt", "llama7b-hf", "llama13b-hf", "vicuna7b-hf", "vicuna13b-hf"],
        read_num = 200
    )
    # corrStats.print_scores()
    

    # Compare Two types of metrics
    for dataset in ["NQ", "HotpotQA", "truthfulQA", "cnndm", "multi-news", "msmarco"]:
        corrStats.compare(
            dataset = dataset,
            ufo_settings = [
                "ChatGPT-ChatGPT-llmse-nosource.json",   # {S_se}
                "ChatGPT-ChatGPT-llmse-nord.json",       # {S_he, S_se}
                "ChatGPT-ChatGPT-llmse-nohe.json",       # {S_rd, S_se}
                "ChatGPT-ChatGPT-llmse-normal.json",     # {S_he, S_rd, S_se}
                "ChatGPT-ChatGPT-llmse-reverse.json"    # {S_rd, S_he, S_se}
            ],
            use_model_doc = False,
            score_type = "llm_judgement_first",
            baseline_metrics = [
                "bleu1", # "bleu2", "bleu3", "bleu4",
                "rougeL",   # "rouge1", "rouge2",
                "bertscore-f1", "bartscore",    # "bertscore-p", "bertscore-r"
                "QAGS_score", "Q2_score",
                "factool_claim", "factool",
                "factscore"
            ]
        )