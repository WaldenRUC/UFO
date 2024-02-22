import datetime, json, requests, openai, re, time, os, pysbd, spacy, string, scipy, krippendorff, torch, sys, rich, math, itertools, random
sys.path.append("/home/dou/UniFlexFactor/baseline/PDP")
import numpy as np
from transformers import logging
from typing import List, Dict
from tqdm import tqdm
from util import PDP
from data import RelevanceJudgments, PreferenceJudgments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
logging.set_verbosity_error()
# from factExtractor import FactExtractor
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def recursive_defaultdict():
    return defaultdict(recursive_defaultdict)

def getDataset2evidKey(args):
    dataset2evidKey = {
        'NQ': ['', ''],
        'HotpotQA': ['', 'document'],
        'truthfulQA': ['summary', ''],
        'cnndm': ['summary', 'document'],
        'multi-news': ['summary', 'document'],
        'msmarco': ['summary', 'document']
    }
    if args.sourceMode == 'nosource':
        for _key in dataset2evidKey:
            dataset2evidKey[_key] = ['', '']
    elif args.sourceMode == 'reverse':
        for _key in dataset2evidKey:
            value = dataset2evidKey[_key]
            dataset2evidKey[_key] = [value[1], value[0]]
    return dataset2evidKey

def remove_punc(text: str):
    # Define the punctuation characters
    punctuations = string.punctuation
    # Remove punctuation from the text
    return ''.join(char for char in text if char not in punctuations)



def nli_prec(MatchScore, phrase, predAns):
    if type(MatchScore) == int:
        return MatchScore
    elif type(MatchScore) == dict:
        ent, neu, cont = MatchScore["entailment"], MatchScore["neutral"], MatchScore["contradiction"]
        if max((ent, neu, cont)) == neu:
            words_a = set(remove_punc(phrase).lower().split())
            words_b = set(remove_punc(predAns).lower().split())
            unique_words_in_a = len(words_a)
            common_words = len(words_a.intersection(words_b))
            coverage_ratio = common_words / unique_words_in_a if unique_words_in_a > 0 else 0
            return coverage_ratio
        else:
            return ent

def reMatch(text: str):
    try:
        properAns, properAnsIDList = [item for item in text.split("\n") if item != ""]
        if properAns[0] == '[': properAns = properAns[1:]
        if properAns[-1] == ']': properAns = properAns[:-1]
        if properAnsIDList[0] == '[': properAnsIDList = properAnsIDList[1:]
        if properAnsIDList[-1] == ']': properAnsIDList = properAnsIDList[:-1]
        properAnsIDList = properAnsIDList.replace("@", "")
        if re.search(r'\d', properAnsIDList):
            return properAns, properAnsIDList
        else:
            return properAns, "-1"
    except Exception as e:
        print(f"**********\nreader answers:\n{text}", sep='\n')
        if len(text.split("\n")) == 2:
            newText = text.split("\n")[0]
        else:
            matchtextList = re.findall(r'\[([^]]+)\]', text)
            if len(matchtextList) == 2:
                newText = matchtextList[0]
            else:
                if "NOANS" in text: newText = "unanswerable"
                else:
                    newText = text.split("\n")[0].split(";")[0]
        print(f"new Text: {newText}")
        return newText, "-1"
    
# def rank_numbers(numbers: List[float]):
#     '''
#     [0.2, 0.4, 0.1] -> [2, 3, 1]
#     '''
#     sorted_indices = sorted(range(len(numbers)), key=lambda i: numbers[i])
#     ranks = [0] * len(numbers)
#     for rank, index in enumerate(sorted_indices):
#         ranks[index] = rank
#     return ranks

def get_closest_text(outputLine: str, modelTextLines: List[str]):
    '''
    find the closest text to outputLine in modelTextLines
    '''
    def levenshtein_distance(s1, s2):
        # Creating a matrix
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for index2, char2 in enumerate(s2):
            new_distances = [index2 + 1]
            for index1, char1 in enumerate(s1):
                if char1 == char2:
                    new_distances.append(distances[index1])
                else:
                    new_distances.append(1 + min((distances[index1], distances[index1 + 1], new_distances[-1])))
            distances = new_distances
        return distances[-1]
    min_distance = float('inf')
    closest_text = None
    for text in modelTextLines:
        distance = levenshtein_distance(outputLine, text)
        if distance < min_distance:
            min_distance = distance
            closest_text = text
    return closest_text






class Benchmark():
    def __init__(self, Model: List[str] = None, Dataset: List[str] = None, tokenizer_path: str = None, dataset_path: str = None, output_path: str = None, output_filenames: List[str] = None, file_num: int = None):
        self.Model = Model
        self.Dataset = Dataset
        self.segmenter = pysbd.Segmenter(language="en", clean=False)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.output_filenames = output_filenames
        self.file_num = file_num
        self.loadDataset()
        self.loadOutput()
    def segment(self, text: str):
        '''
        Segment each response into several sentences
        '''
        return [line.strip() for line in self.segmenter.segment(text)]
    def StatTokens(self, modelText: str):
        '''
        Get Tokens of each response
        '''
        return len(self.tokenizer(modelText)["input_ids"]) - 1
    def StatSents(self, modelText: str):
        '''
        Get Sentences of each response
        '''
        return len(self.segment(modelText))
    def StatFacts(self, dataset: str, model: str, _type: str, lineID: str, prefix="output"):
        '''
        Get Facts of each response
        '''
        readFilePath = os.path.join(prefix, dataset, model.split(".")[0], lineID, f"factExtract-{_type}.json")
        assert os.path.exists(readFilePath), "File Not Exists!"
        with open(readFilePath, 'r', encoding='utf-8') as fp:
            return len(json.load(fp)["extractFactDictList"])
    def loadDataset(self):
        '''
        self.dataset["newbing"]["NQ"] = [{}, {}, ...]
        '''
        self.dataset = dict()
        for model in self.Model:
            self.dataset[model] = dict()
            for dataset in self.Dataset:
                with open(os.path.join(self.dataset_path, dataset, model+".json"), 'r', encoding='utf-8') as fp:
                    self.dataset[model][dataset] = json.load(fp)[:self.file_num]
    def loadOutput(self):
        '''
        self.output["newbing"]["NQ"]["0"]["ChatGPT-llmse-normal"] = [{}, {}, ...]
        '''
        self.output = dict()
        for model in tqdm(self.Model, desc="loading output"):
            self.output[model] = dict()
            for dataset in self.Dataset:
                self.output[model][dataset] = dict()
                for i in range(self.file_num):
                    self.output[model][dataset][str(i)] = dict()
                    for output_filename in self.output_filenames:
                        with open(os.path.join(self.output_path, dataset, model, str(i), output_filename+".json"), 'r', encoding='utf-8') as fp:
                            self.output[model][dataset][str(i)][output_filename] = json.load(fp)
    def benchmark_res(self, func_name: str = None):
        '''
        Print Benchmark Statistics
        '''
        assert func_name in ["StatTokens", "StatSents", "StatFacts-ChatGPT", "StatFacts-PLM"]
        print(f"*****{func_name}*****")
        for model in self.Model:
            model_name = model.split(".")[0]
            for dataset in self.Dataset:
                result = list()
                for lineID, line in enumerate(self.dataset[model_name][dataset]):
                    modelText = line["modelText"]
                    if func_name == "StatTokens":
                        result.append(self.StatTokens(modelText))
                    if func_name == "StatSents":
                        result.append(self.StatSents(modelText))
                    if func_name == "StatFacts-ChatGPT":
                        result.append(self.StatFacts(dataset, model, "ChatGPT", str(lineID)))
                    if func_name == "StatFacts-PLM":
                        result.append(self.StatFacts(dataset, model, "PLM", str(lineID)))
                print(f"{model_name}, {dataset}, {round(np.mean(result), 2)}")
        print(f"*****{func_name}*****")


def Scores2Ranks(scores: List[List[float]]):
    # [[0.1, ...], [0.22, ...], ..., [0.05, ...]] --> [[2, ...], ..., [1, ...]]
    assert all(len(lst) == len(scores[0]) for lst in scores), f"length: {[len(score) for score in scores]}"
    rank_scores = [[] for _ in range(len(scores))]
    for elements in zip(*scores):
        sorted_elements = sorted(enumerate(elements), key=lambda x: x[1])
        rank = 1
        prev_value = None
        for index, (original_index, value) in enumerate(sorted_elements):
            if value != prev_value:
                rank = index + 1
                prev_value = value
            rank_scores[original_index].append(rank)
    return rank_scores






class CorrelationStatistics():
    def __init__(self, 
                prefix: str = "baseline", 
                Model: List[str] = ["newbing", "chatgpt", "llama7b-hf", "llama13b-hf", "vicuna7b-hf", "vicuna13b-hf"], 
                Dataset: List[str] = ["NQ", "HotpotQA", "truthfulQA", "cnndm", "multi-news", "msmarco"],
                baselines: List[str] = ["bleu", "rouge", "bertscore", "bartscore", "qags", "q2", "factool", "factscore"], 
                prompt_dir_path: str = "prompt",
                read_num: int = 200):
        self.baselines, self.Model, self.Dataset = baselines, Model, Dataset
        self.pdp = PDP()
        self.read_num = read_num
        self.metrics_output = recursive_defaultdict()   # ["bleu1"]["newbing"]["NQ"] = [0.22, 0.1, ...]
        
        # load promptDict
        promptDict = dict()
        for filename in ["extractFact.txt", "genModelText.txt", "genTitle.txt", "LLMReader.txt", "QA.txt", "judge.txt"]:
            with open(os.path.join(prompt_dir_path, filename), 'r', encoding='utf-8') as fp:
                promptDict[filename.split(".")[0]] = fp.read()
        self.promptDict = promptDict
        for baseline in tqdm(self.baselines, desc="loading baseline"):
            with open(os.path.join(prefix, f"baseline-{baseline}.json"), 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            for model, model_dict in data.items():
                if model not in self.Model: continue
                for dataset, dataset_list in model_dict.items():
                    if dataset not in self.Dataset: continue
                    # dataset_list:
                    # [{"bleu1": .., "bleu2": ...}, ]
                    # [{"process": ..., "Q2_score": }]
                    dataset_list = dataset_list[:self.read_num]
                    assert len(dataset_list) == self.read_num, f"length: {len(dataset_list)}, {self.read_num}; {baseline}, {model}, {dataset}"
                    kv_dict = defaultdict(list)
                    for lineID, line_dict in enumerate(dataset_list):
                        # line_dict:
                        # {"bleu1": ..., "bleu2": ...}
                        if "process" in line_dict: line_dict.pop("process")
                        if baseline == "factool":
                            if "response_list" in line_dict:
                                line_dict["factool_claim"] = line_dict["response_list"]["average_claim_level_factuality"]
                                line_dict.pop("response_list")
                            else:
                                line_dict["factool_claim"] = 0
                        for k, v in line_dict.items():
                            kv_dict[k].append(v)
                    for k, v_list in kv_dict.items():
                        self.metrics_output[k][model][dataset] = v_list

        # input: self.metrics_output
        # ["bleu1"]["newbing"]["NQ"] = [0.31, 0.82, ...]
        # ["bleu1"]["chatgpt"]["NQ"] = [0.11, 0.69, ...]
        

        # output: self.metrics_rank_output
        self.metrics_rank_output = recursive_defaultdict()
        # fine_metrics: ["bleu1", "bleu2", ...]
        fine_metrics = list(self.metrics_output.keys())
        for fine_metric in tqdm(fine_metrics, desc=f"ranking baseline..."):
            # tqdm.write(f"current: {fine_metric}")
            for dataset in self.Dataset:
                scores = list()
                for model in self.Model:
                    scores.append(self.metrics_output[fine_metric][model][dataset])
                # [[0.1, ...], [0.22, ...], ..., [0.05, ...]] --> [[2, ...], ..., [1, ...]]
                rank_scores = Scores2Ranks(scores)
                for _id, rank_score in enumerate(rank_scores):
                    self.metrics_rank_output[fine_metric][self.Model[_id]][dataset] = rank_score

    def print_scores(self):
        for metric, metric_values in self.baseline_metrics.items():
            for model, model_values in metric_values.items():
                for dataset, dataset_list in model_values.items():
                    print(f"metric: {metric}; model: {model}; dataset: {dataset}; AVG: {round(np.mean(dataset_list), 3)}")

    def correlation(self, x: List[float], y: List[float]):
        '''
        obtain correlation of two sequences x and y
        '''
        def krip(*lists):
            # x.shape: [N, Count_assessors]
            x = list()
            for elements in zip(*lists):
                x.append(list(elements))
            x = np.array(x)
            # print(f"krippendorff's alpha: input shape [N, Count_assessors]: {x.shape}")
            return krippendorff.alpha(reliability_data=x)
        pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
        spearman_r, spearman_p = scipy.stats.spearmanr(x, y)
        krip_a = krip(x, y)
        print(f'''
Pearson r: {round(pearson_r, 3)}; p-value: {round(pearson_p, 3)}
Spearman r: {round(spearman_r, 3)}; p-value: {round(spearman_p, 3)}
krippendorff's Alpha: {round(krip_a, 3)}''')

    def correlation_dp(self, x: Dict[str, List[List[float]]] = None, dataset: str = None, use_model_doc: bool = False):
        '''
        compute discriminative power of given metrics, and draw DP curve
        inputs:
        x: [systems, scores]
        [
            [0.1, 0.2, 0.3], # Bing Chat
            [0.2, 0.3, 0.4], # ChatGPT
            ...
        ]
        '''
        def draw_dp(res: List[Dict] = None, dataset: str = None):
            '''
            draw dp MR-PT trade-offs with different baselines
            res:
            [
            {
                "MRs": [...],
                "PTs": [...],
                "metric": str
            }, ...
            ]
            '''
            font_path = "/home/xiaochen_zuo/haonan/BARTSessionSearch/Times-Roman.ttf"
            font_prop = FontProperties(fname=font_path)
            plt.figure(figsize=(6, 7), dpi=300)
            plt.xlabel('Proportion of ties (PT)', fontproperties=font_prop, fontsize=16)
            plt.ylabel('Minority rate (MR)', fontproperties=font_prop, fontsize=16)
            plt.title(f'MR-PT Discriminative Power Curve on {dataset}', fontproperties=font_prop, fontsize=16)
            for line in res:
                MRs, PTs, label = line["MRs"], line["PTs"], line["metric"]
                if label == "factool":
                    label = "factool-r"
                if label == "factool_claim":
                    label = "factool-c"
                label = label.replace('_score', '')
                if label.startswith('bleu'):
                    marker = 'o'
                elif label.startswith('rouge'):
                    marker = 's'
                elif label.startswith('bert') or label.startswith('bart'):
                    marker = '^'
                elif label == "QAGS" or label == "Q2":
                    marker = 'p'
                elif label.startswith('factool'):
                    marker = 'd'
                elif label == "factscore":
                    marker = '*'
                else: marker = '*'
                # plt.scatter(MRs, PTs)
                plt.plot(MRs, PTs, label=label, linestyle='-', marker=marker)
            plt.xticks(fontsize='medium')
            plt.yticks(fontsize='medium')
            plt.legend(fontsize='medium')
            plt.tight_layout()
            if use_model_doc:
                with open(f"figures/dpCurve-{dataset}-useModelDoc.json", 'w') as fw:
                    fw.write(json.dumps(res))
            else:
                with open(f"figures/dpCurve-{dataset}.json", 'w') as fw:
                    fw.write(json.dumps(res))
        
        MRPTs = list()
        for metric, metric_list in tqdm(x.items(), desc=f"compute DP: {dataset}"):
            MRs, PTs = list(), list()
            for fuzziness in np.linspace(0, 0.15, 15):
                MR, PT = self.dp(
                    metric_list,
                    fuzziness=fuzziness
                )
                MRs.append(MR)
                PTs.append(PT)
            MRPTs.append({"MRs": MRs, "PTs": PTs, "metric": metric})
        draw_dp(res=MRPTs, dataset=dataset)

    def dp(self, x, fuzziness: float = 0.1, bootstrap_times: int = 1000):
        '''
        Evaluating Evaluation Metrics based on the Bootstrap
        计算metric的MR和PT
        ```
        MR estimates the chance of reaching a wrong conclusion about a system pair, while PT reflects lack of discriminative power.
        Thus, for a good performance metric, both of these values should be small.
        As a fixed fuzziness value implies different trade-offs for different metrics, we vary f (= 0.01, 0.02,...,0.20) for comparing the stability.
        ```
        Inputs:
            - x.shape: [systems, scores]
            [
                [0.5, 0.2, 0.3],    # Bing Chat, samples count Q=3
                [0.2, 0.1, 0.2],    # ChatGPT
                ...
            ]
            - bootstrap_times (B): 
            (001, 002, 003, 004) --> (001, 002, 003, 002)
        Returns:
        - MR, PT
        '''
        def M(s, random_indexes):
            return np.mean(s[random_indexes])
        
        x = np.array(x)
        systems, Q = x.shape
        C_sum = math.comb(systems, 2)
        # [1, 2, 4] --> [(1, 2), (1, 4), (2, 4)]
        combinations_list = list(itertools.combinations(x, 2))
        # for each system pair (X, Y) \in C
        EQs, GTs = list(), list()
        for X, Y in combinations_list:
            EQ, GT_xy, GT_yx = 0, 0, 0
            for b in range(bootstrap_times):
                # (0, 1, 2, 3) --> [0, 0, 3, 2]
                random_indexes = random.choices(range(Q), k=Q)
                metric_output_x, metric_output_y = M(X, random_indexes), M(Y, random_indexes)
                margin = fuzziness * max(metric_output_x, metric_output_y)
                if abs(metric_output_x - metric_output_y) < margin:
                    EQ += 1
                elif metric_output_x > metric_output_y:
                    GT_xy += 1
                else:
                    GT_yx += 1
            EQs.append(EQ)
            GTs.append(min(GT_xy, GT_yx))

        MR = np.sum(GTs) / (bootstrap_times * C_sum)
        PT = np.sum(EQs) / (bootstrap_times * C_sum)
        return MR, PT
    def ChatGPTAPICall(self, userQuery: str, model_name: str = "gpt-3.5-turbo-1106"):
        if userQuery in self.Prompt2ResDict:
            return self.Prompt2ResDict[userQuery]["response"], self.Prompt2ResDict[userQuery]["usage"]
        _count = 0
        while True:
            try:
                # print(f"send ChatGPT API requests...")
                req = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[{"role": "user", "content": userQuery}],
                    request_timeout = 60
                )
                response = req.choices[0].message["content"]
                usage = req.usage.total_tokens
                print(f"get requests from ChatGPT API! USAGE: [ {usage} ]")
                self.Prompt2ResDict[userQuery] = {
                    "response": response,
                    "usage": usage
                }
                return response, usage
            except Exception as e:
                print(f"SEND ERROR: [ {e} ]")
                _count += 1
                if _count >= 5: return None, None
                print(f"ChatGPT API Call failed. Retry... {e}")
                time.sleep(1)
    def getLLMJudgeScore(self, phrase, predAns):
        final_scores = list()
        judge = self.promptDict["judge"].format(phrase, predAns)
        response, usage = self.ChatGPTAPICall(judge)
        response = response.lower()
        if "yes" in response:
            final_scores.append(1)
        elif "no" in response:
            final_scores.append(0)
        else:
            print(f"judge response: {response}")
            final_scores.append(0)
        return np.mean(final_scores)
    def get_score_from_file(self, file_name: str, score_type='llm_judgement'):
        with open(file_name, 'r', encoding='utf-8') as fp:
            lines = json.load(fp)
        scores = list()
        for line in lines:
            if score_type == 'llm_judgement':
                llm_judgement = line["llm_judgement"]
                if "yes" in llm_judgement:
                    scores.append(1)
                elif "no" in llm_judgement:
                    scores.append(0)
                else:
                    print(f"error yes/no response: {llm_judgement}")
                    scores.append(0)
            elif score_type == 'nli':
                scores.append(nli_prec(line["MatchScore"], line["phrase"], line["predAns"]))
            elif score_type == 'llm_judgement_first':
                llm_judgement = line["llm_judgement_first"]
                if "yes" in llm_judgement:
                    scores.append(1)
                elif "no" in llm_judgement:
                    scores.append(0)
                else:
                    print(f"error yes/no response: {llm_judgement}")
                    scores.append(0)
            else: assert False, score_type
        return np.mean(scores)

    def get_ufo_scores(self, ufo_setting: str, use_model_doc: bool, score_type: str):
        '''
        Returns: ufo_scores[<model>][<dataset>]["scores"/"rank"] --> List[float/int]
        '''
        ufo_setting_name = ufo_setting.split('.')[0]
        if hasattr(self, ufo_setting_name):
            ufo_scores = getattr(self, ufo_setting_name, f'property {ufo_setting_name} not found')
            return ufo_scores
        ufo_scores = dict()
        for model in tqdm(self.Model, desc=f"get ufo scores: {ufo_setting}"):
            filename_prefix = ufo_setting.split(".")[0]
            if use_model_doc and model=="newbing":
                cur_ufo_setting = filename_prefix + ".json"
            else:
                cur_ufo_setting = filename_prefix + "-noModelDoc.json"
            ufo_scores[model] = dict()
            for dataset in self.Dataset:
                ufo_scores[model][dataset] = dict()
                _scores = list()
                for i in range(self.read_num):
                    file_name = os.path.join("output", dataset, model, str(i), cur_ufo_setting)
                    _scores.append(self.get_score_from_file(file_name, score_type=score_type))
                ufo_scores[model][dataset]["scores"] = _scores

        # get ranking scores
        for dataset in tqdm(self.Dataset, desc=f"get ufo rank: {ufo_setting}"):
            _scores = list()
            for model in self.Model:
                _scores.append(ufo_scores[model][dataset]["scores"])
            rank_scores = Scores2Ranks(_scores)
            for _id, rank_score in enumerate(rank_scores):
                ufo_scores[self.Model[_id]][dataset]["rank"] = rank_score
        setattr(self, ufo_setting_name, ufo_scores)
        return ufo_scores
    def compare(self, dataset: str = None, ufo_settings: List[str] = None, baseline_metrics: List[str] = None, use_model_doc: bool = False, score_type: str = "llm_judgement"):
        self.ufo_settings = ufo_settings
        self.baseline_metrics = baseline_metrics
        ufo_score_dict, baseline_score_dict = dict(), dict()

        # load ufo metric
        for ufo_setting in self.ufo_settings:
            ufo_scores = self.get_ufo_scores(ufo_setting, use_model_doc, score_type)
            # score
            ufo_score_list = list()
            for model in ufo_scores.keys():
                ufo_score = ufo_scores[model][dataset]["scores"]
                ufo_score_list.append(ufo_score)
            ufo_score_dict[ufo_setting] = ufo_score_list
        

        # load baseline metrics
        for baseline_metric in self.baseline_metrics:
            baseline_scores_dict = self.metrics_output[baseline_metric]
            # score
            baseline_score_list = list()
            for model in baseline_scores_dict.keys():
                baseline_score = baseline_scores_dict[model][dataset]
                baseline_score_list.append(baseline_score)
            baseline_score_dict[baseline_metric] = baseline_score_list

            # rank
            # baseline_rank_dict = self.metrics_rank_output[baseline_metric]
            # ufo_rank_list, baseline_rank_list = list(), list()
            # for model in baseline_scores_dict.keys():
            #     ufo_rank = ufo_scores[model][dataset]["rank"]
            #     ufo_rank_list.append(ufo_rank)
            #     baseline_rank = baseline_rank_dict[model][dataset]
            #     baseline_rank_list.append(baseline_rank)
            # self.correlation(ufo_rank_list, baseline_rank_list)
        
        # self.correlation_dp(x=baseline_score_dict, dataset=dataset)
        for k, v in ufo_score_dict.items():
            baseline_score_dict[k] = v
        self.correlation_dp(x=baseline_score_dict, dataset=dataset, use_model_doc=use_model_doc)
        # for setting in ufo_settings:
        #     self.correlation(list(itertools.chain(*ufo_score_dict[setting])), list(itertools.chain(*baseline_score_dict["factool_claim"])))
        #     self.correlation(list(itertools.chain(*ufo_score_dict[setting])), list(itertools.chain(*baseline_score_dict["factscore"])))
            
        


class UFO():
    def __init__(self, 
                nli_dir_path = "/home/dou/.cache/huggingface/hub/models--ynie--roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli/snapshots/5b605abab9b75bc87ab66cfc049ef58d9d64b8ed",
                dataset_dir_path: str = "dataset",
                env_file_path: str = "env.json",
                prompt_dir_path: str = "prompt",
                args = None):
        self.NLItokenizer = AutoTokenizer.from_pretrained(nli_dir_path)
        self.NLImodel = AutoModelForSequenceClassification.from_pretrained(nli_dir_path).cuda()
        self.nlp = spacy.load("en_core_web_sm")
        self.segmenter = pysbd.Segmenter(language="en", clean=False)
        self.dataset_dir_path = dataset_dir_path
        self.QA_threshold = 0.1
        
        # load settings
        self.args = args
        
        # load envDict
        with open(env_file_path) as fp:
            self.envDict = json.load(fp)
        self.X_APIs = self.envDict["X-API-KEY"]
        openai.api_key = self.envDict["chatgpt-api"]

        # load promptDict
        promptDict = dict()
        for filename in ["extractFact.txt", "genModelText.txt", "genTitle.txt", "LLMReader.txt", "QA.txt", "judge.txt"]:
            with open(os.path.join(prompt_dir_path, filename), 'r', encoding='utf-8') as fp:
                promptDict[filename.split(".")[0]] = fp.read()
        self.promptDict = promptDict

        # load PLMExtractor
        if self.args.QA == "PLM":
            self.PLMExtractor = FactExtractor(plmPathPrefix = self.envDict["plmPathPrefix"])
            self.PLMExtractor.load_everything()

        

    # def loadData(self):
    #     #TODO
    #     pass
    def loadJson(self, Path: str, defaultType):
        '''
        Path: 搜索历史的文件路径
        Returns:
        - searchResDict: dict
        '''
        if not os.path.exists(Path): 
            if defaultType == list:
                return list()
            elif defaultType == dict:
                return dict()
            else: assert False, defaultType
        with open(Path, 'r', encoding='utf-8') as fp:
            return json.load(fp)
    def process_longDoc(self, textList: list, max_words: int = 64):
        def split_text_into_passages(text: str, max_words=64):
            """
            :param text: The text to be split into passages.
            :param max_words: Maximum number of words allowed in each passage.
            :return: A list of passages.
            """
            text_ids = self.NLItokenizer.encode(text, add_special_tokens=True)
            chunks = [text_ids[i:i + max_words] for i in range(0, len(text_ids), max_words)]
            decoded_chunks = [self.NLItokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]
            return decoded_chunks
        res = list()
        for text in textList:
            if text == "": continue
            text = re.sub(r'\[.*?\]', '', text)
            res.extend(split_text_into_passages(text, max_words=max_words))
        return res
    def getEvid(self, line: dict = None, modelText: str = None, evidList: list = None, model: str = None, QA_model: str = None):
        evidKey1, evidKey2 = evidList[0], evidList[1]       # '', 'summary'
        summaryList, documentList = line.get(evidKey1), line.get(evidKey2)
        if QA_model == "PLM":
            max_words = 64
        elif QA_model == "ChatGPT":
            max_words = 1024
        else: assert False, QA_model
        if type(summaryList) == str:
            summaryList = [summaryList]
        if type(documentList) == str:
            documentList = [documentList]

        # load human-written evidence
        if summaryList != None:
            summaryList = self.process_longDoc(summaryList, max_words=max_words)
        else: summaryList = list()

        # load reference documents
        if documentList != None:
            documentList = self.process_longDoc(documentList, max_words=max_words)
        else: documentList = list()

        # load bing chat documents if exists
        if model == "newbing":
            newbing_docList = self.process_longDoc(line["urlList_text"], max_words=max_words)
        else: newbing_docList = list()

        return summaryList, documentList, newbing_docList
    def getScoreNLI(self, prem: str, hypo: str, max_length=128):
        assert type(prem) == type(hypo) == str, (prem, hypo)
        tokenized_input_seq_pair = self.NLItokenizer.encode_plus(prem, hypo, max_length=max_length, return_token_type_ids=True, truncation=True)
        input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(torch.device("cuda"))
        token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(torch.device("cuda"))
        attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(torch.device("cuda"))
        outputs = self.NLImodel(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=None)
        predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()
        entailment_score, neutral_score, contradiction_score = predicted_probability[0], predicted_probability[1], predicted_probability[2]
        return {
            "entailment": entailment_score, 
            "neutral": neutral_score, 
            "contradiction": contradiction_score
        }


    def readScores(self, eval2filePath: str = None, devided_by: str = "fact_unit", score_type = "fact_unit_LLMjudgement") -> List:
        with open(eval2filePath, 'r') as fp:
            output = json.load(fp)
        final_scores = list()
        if devided_by == "fact_unit":
            for _output in output:
                if score_type == "nli_prec":
                    final_scores.append(nli_prec(_output["MatchScore"], _output["phrase"], _output["predAns"]))
                elif score_type == "fact_unit_LLMjudgement":
                    llm_judgement = _output["llm_judgement"]
                    llm_judgement = llm_judgement.lower()                    
                    if "yes" in llm_judgement:
                        final_scores.append(1)
                    elif "no" in llm_judgement:
                        final_scores.append(0)
                    else:
                        print(f"judge response: {llm_judgement}")
                        final_scores.append(0)
                elif score_type == "fact_unit_LLMjudgement_first":
                    llm_judgement = _output["llm_judgement_first"]
                    llm_judgement = llm_judgement.lower()                    
                    if "yes" in llm_judgement:
                        final_scores.append(1)
                    elif "no" in llm_judgement:
                        final_scores.append(0)
                    else:
                        print(f"judge response: {llm_judgement}")
                        final_scores.append(0)
                else: assert False, score_type
        elif devided_by == "sentence":
            modelTextLines = [line.strip() for line in self.segmenter.segment(self.modelText)]
            final_scores = {line: list() for line in modelTextLines}
            for _output in output:
                line = _output["oriSent"]
                flag = 0
                textLine = get_closest_text(line, modelTextLines)
                if score_type == "nli_prec":
                    final_scores[textLine].append(nli_prec(_output["MatchScore"], _output["phrase"], _output["predAns"]))
                else: assert False, score_type
            for _key, score in final_scores.items():
                if score == []:
                    final_scores[_key] = 1
                else:
                    final_scores[_key] = np.mean(final_scores[_key])
            final_scores = list(final_scores.values())
        else: assert False, devided_by
        return final_scores
    def GoogleAPICall(self, searchQuery: str):
        '''
        Returns:
        - response(json)
        '''
        if searchQuery in self.searchResQuery2SnippetDict:
            response = {
                "organic": self.searchResQuery2SnippetDict[searchQuery]
            }
            return response
        url = "https://google.serper.dev/search"
        payload = json.dumps({
            "q": searchQuery
        })
        while True:
            try:
                if len(self.X_APIs) == 0: return None
                headers = {
                    'X-API-KEY': self.X_APIs[0],
                    'Content-Type': 'application/json'
                }
                response = requests.request("POST", url, headers=headers, data=payload)
                print(f"send Google API requests...")
                response = json.loads(response.text)
                if "organic" not in response:
                    self.X_APIs.pop(0)
                    print("Google Search API popping!")
                    assert False
                if len(response["organic"]) != 0:
                    self.searchResQuery2SnippetDict[searchQuery] = response["organic"]
                return response
            except Exception as e:
                print("google API call failed. Retry...")
                time.sleep(0.5)
    def ChatGPTAPICall(self, userQuery: str, model_name: str = "gpt-3.5-turbo-1106"):
        if userQuery in self.Prompt2ResDict:
            return self.Prompt2ResDict[userQuery]["response"], self.Prompt2ResDict[userQuery]["usage"]
        _count = 0
        while True:
            try:
                # print(f"send ChatGPT API requests...")
                req = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[{"role": "user", "content": userQuery}],
                    request_timeout = 60
                )
                response = req.choices[0].message["content"]
                usage = req.usage.total_tokens
                print(f"get requests from ChatGPT API! USAGE: [ {usage} ]")
                self.Prompt2ResDict[userQuery] = {
                    "response": response,
                    "usage": usage
                }
                return response, usage
            except Exception as e:
                print(f"SEND ERROR: [ {e} ]")
                _count += 1
                if _count >= 5: return None, None
                print(f"ChatGPT API Call failed. Retry... {e}")
                time.sleep(1)
    def getQAPred(self, evidence: str, qa_pair: dict):
        _query, _phrase = qa_pair["question"], qa_pair["phrase"]
        for itemDict in self.QAResList: # searchQ, phrase, evidence, prediction, predScore
            if itemDict["searchQ"] == _query and itemDict["phrase"] == _phrase and itemDict["evidence"] == evidence:
                return itemDict["prediction"], itemDict["predScore"]
        result = self.PLMExtractor.qa(evidence, [qa_pair])[0]
        self.QAResList.append({
            "searchQ": _query,
            "phrase": _phrase,
            "evidence": evidence,
            "prediction": result["prediction"],
            "predScore": result["score"]
        })
        return result["prediction"], result["score"]
    def evidVerify(self, evidences: List[str] = None, evidence_tag: str = None):
        '''
        用human-written evidence, reference documents验证
        '''
        assert type(evidences) == list
        IDList4verify = [int(k) for k, v in self.evaluation_outputs.items() if v == 0]
        qList = [fact["question"] for (factID, fact) in enumerate(self.extractFactDictList) if factID in IDList4verify]
        phraseList = [fact["phrase"] for (factID, fact) in enumerate(self.extractFactDictList) if factID in IDList4verify]
        assert len(qList) == len(phraseList) == len(IDList4verify)
        if self.args.QA == "PLM":
            _qa_pairs = [{"question": self.generatedTitle + "; " + _q, "phrase": _p} for (_q, _p) in zip(qList, phraseList)]
            Score_evidence = [[(None, None)] * len(evidences) for _ in range(len(IDList4verify))] 
            for colID, _evidence in tqdm(enumerate(evidences), desc="Verification QA from evidence ID"):
                _predAnsList, _predScoreList = list(), list()
                for qa_pair in _qa_pairs:
                    _predAns, _predScore = self.getQAPred(_evidence, qa_pair)
                    _predAnsList.append(_predAns)
                    _predScoreList.append(_predScore)
                for rowID, (_r, _s) in enumerate(zip(_predAnsList, _predScoreList)):
                    Score_evidence[rowID][colID] = (_r, _s)
            for _idx, curFactID in enumerate(IDList4verify):
                factScoreList = [T[1] for T in Score_evidence[_idx]] # [0.22, 0.1, ..]
                max_value = max(factScoreList)
                max_idx = factScoreList.index(max_value)
                if max_value > self.QA_threshold:   # 可被验证
                    predAns = Score_evidence[_idx][max_idx][0]
                    predScore = Score_evidence[_idx][max_idx][1]
                    if "unanswerable" in predAns: continue
                    self.evaluation_outputs[str(curFactID)] = {
                        "question": self.extractFactDictList[curFactID]["question"],
                        "phrase": self.extractFactDictList[curFactID]["phrase"],
                        "predAns": predAns,
                        "predScore": predScore,
                        "MatchScore": self.getScoreNLI(predAns, self.extractFactDictList[curFactID]["phrase"]),
                        "oriSent": self.oriSents[curFactID],
                        "process": evidence_tag,
                        "properAnsIDList": evidence_tag
                    }
        elif self.args.QA == "ChatGPT":
            '''
            用prompt的方式获得答案, 如果返回为NOANS则说明该evidence不能支持
            '''
            for (curFactID, question) in zip(IDList4verify, qList):
                responses = list()
                for evidence in evidences:
                    qa_prompt = self.promptDict["QA"].format(
                        self.generatedTitle, evidence, question
                    )
                    response, usage = self.ChatGPTAPICall(qa_prompt)
                    if "NOANS" in response or "unanswerable" in response:
                        continue
                    else:
                        responses.append(response)
                if len(responses) > 0:
                    phrase = self.extractFactDictList[curFactID]["phrase"]
                    # judgement
                    judgement = self.promptDict["judge"].format(phrase, ", ".join(responses))
                    response, usage = self.ChatGPTAPICall(judgement)
                    response = response.lower()
                    llm_judgement = response

                    # judgement_first
                    judgement_first = self.promptDict["judge"].format(phrase, responses[0])
                    response, usage = self.ChatGPTAPICall(judgement_first)
                    response = response.lower()
                    llm_judgement_first = response
                    self.evaluation_outputs[str(curFactID)] = {
                        "question": self.extractFactDictList[curFactID]["question"],
                        "phrase": phrase,
                        "predAns": responses,
                        "predScore": "2",
                        "MatchScore": self.getScoreNLI(responses[0], phrase),
                        "llm_judgement": llm_judgement,
                        "llm_judgement_first": llm_judgement_first,
                        "oriSent": self.oriSents[curFactID],
                        "process": evidence_tag,
                        "properAnsIDList": evidence_tag
                    }
    def SEVerify(self):
        IDList4verify = [int(k) for k, v in self.evaluation_outputs.items() if v == 0]
        qList = [fact["question"] for (factID, fact) in enumerate(self.extractFactDictList) if factID in IDList4verify]
        phraseList = [fact["phrase"] for (factID, fact) in enumerate(self.extractFactDictList) if factID in IDList4verify]
        assert len(qList) == len(phraseList) == len(IDList4verify)
        for curFactID in IDList4verify:
            # 当前需要验证的事实ID
            question, phrase = self.extractFactDictList[curFactID]["question"], self.extractFactDictList[curFactID]["phrase"]
            searchQuery = self.generatedTitle + "; " + question
            if self.args.checker == "llmse":
                response = self.GoogleAPICall(searchQuery)
                snippetList = list()
                for item in response["organic"]:
                    if "snippet" in item:
                        snippetList.append(item["snippet"])
                snippets = "\n".join(snippetList)
                Text = f'\nQuery:\n{question}\n\nTopic:\n{self.generatedTitle}\n\nSnippets:\n{snippets}'
                LLMReader = self.promptDict["LLMReader"] + Text
                response, usage = self.ChatGPTAPICall(userQuery=LLMReader)
                predAns, predAnsIDList = reMatch(response)

                # judgement
                judgement = self.promptDict["judge"].format(phrase, predAns)
                response, usage = self.ChatGPTAPICall(judgement)
                response = response.lower()
                llm_judgement = response

                if "NOANS" in predAns or "unanswerable" in predAns:
                    matchScore = 0
                else:
                    matchScore = self.getScoreNLI(predAns, phrase)
                
                self.evaluation_outputs[str(curFactID)] = {
                    "question": question,
                    "phrase": phrase,
                    "predAns": predAns,
                    "predScore": "LLM",
                    "oriLLMans": response,
                    "MatchScore": matchScore,
                    "llm_judgement": llm_judgement,
                    "llm_judgement_first": llm_judgement,
                    "oriSent": self.oriSents[curFactID],
                    "process": self.args.checker,
                    "properAnsIDList": predAnsIDList
                }

            elif self.args.checker == "se":
                response = self.GoogleAPICall(searchQuery)
                snippetList = list()
                for item in response["organic"]:
                    if "snippet" in item:
                        snippetList.append(item["snippet"])
                _qa_pair = [{"question": searchQuery, "phrase": phrase}]
                candidateAnsList, candidateScoreList = list(), list()
                for _snippetID, _snippet in tqdm(enumerate(snippetList), desc="Snippet QA"):
                    answerScore = self.PLMExtractor.qa(_snippet, _qa_pair)[0]
                    candidateAnsList.append(answerScore["prediction"])
                    candidateScoreList.append(answerScore["score"])
                if len(candidateScoreList) == 0:
                    self.evaluation_outputs[str(curFactID)] = {
                        "question": question,
                        "phrase": phrase,
                        "predAns": "NOANS",
                        "predScore": 0,
                        "MatchScore": 0,
                        "oriSent": self.oriSents[curFactID],
                        "process": self.args.checker,
                        "properAnsIDList": self.args.checker,
                    }
                else:
                    _maxValue = max(candidateScoreList)
                    _maxIndex = candidateScoreList.index(_maxValue)
                    predAns = candidateAnsList[_maxIndex]
                    self.evaluation_outputs[str(curFactID)] = {
                        "question": question,
                        "phrase": phrase,
                        "predAns": predAns,
                        "predScore": _maxValue,
                        "MatchScore": self.getScoreNLI(predAns, phrase),
                        "oriSent": self.oriSents[curFactID],
                        "process": self.args.checker,
                        "properAnsIDList": _maxIndex
                    }
            else: assert False, self.args.checker

    def evaluate(self, 
                modelText: str = None, output_file_path: str = None, 
                human_evidences: List[str] = None, reference_documents: List[str] = None):
        self.modelText = modelText
        self.output_file_path = output_file_path
        assert type(human_evidences) == type(reference_documents) == list
        # initialize file path
        factExtractPath = os.path.join(self.output_file_path, f'factExtract-{self.args.extractor}.json')
        if self.args.useModelDoc:
            eval2filePath = os.path.join(self.output_file_path, f'{self.args.extractor}-{self.args.QA}-{self.args.checker}-{self.args.sourceMode}.json')
        else:
            eval2filePath = os.path.join(self.output_file_path, f'{self.args.extractor}-{self.args.QA}-{self.args.checker}-{self.args.sourceMode}-noModelDoc.json')

        # load Google SERP output cache
        self.searchResQuery2SnippetDict = self.loadJson(os.path.join(self.output_file_path, "searchRes.json"), defaultType=dict)
        # load chatgpt output cache
        self.Prompt2ResDict = self.loadJson(os.path.join(self.output_file_path, "promptRes.json"), dict)
        # searchQ, phrase, evidence, prediction, predScore
        self.QAResList = self.loadJson(os.path.join(self.output_file_path, "QAHistory.json"), defaultType=list) 

        # fact unit extraction
        if not os.path.exists(factExtractPath):
            usage_title, usage_fact, title, responseDictList = self.extractFacts(modelText, self.extractor)
            with open(factExtractPath, 'w') as fw:
                fw.write(json.dumps({
                    "Usage_title": usage_title,
                    "Usage_extraction": usage_fact,
                    "generatedTitle": title,
                    "extractFactDictList": responseDictList
                }))
        with open(factExtractPath, 'r', encoding='utf-8') as fp:
            facts = json.load(fp)
            self.generatedTitle = facts["generatedTitle"]
            self.extractFactDictList = facts["extractFactDictList"]
            self.usage_ChatGPT = facts["Usage_title"] + facts["Usage_extraction"]
            self.oriSents = [itemDict["sentence"] for itemDict in self.extractFactDictList]

         
        if not os.path.exists(eval2filePath):
            keys = [str(x) for x in range(len(self.extractFactDictList))]        # ["0", "1", "2", ...]
            values = [0] * len(self.extractFactDictList)     # [0, 0, 0, ...]
            self.evaluation_outputs = dict(zip(keys, values))  # {"0": 0, "1": 0}
            if self.args.sourceMode != "reverse":
                evidence_tags = ["summary", "document"]
            else: evidence_tags = ["document", "summary"]

            # verify with human-written evidences and ref documents
            if human_evidences != []:
                self.evidVerify(
                    evidences = human_evidences,
                    evidence_tag = evidence_tags[0]
                )
                with open(os.path.join(output_file_path, "QAHistory.json"), 'w') as fw:
                    fw.write(json.dumps(self.QAResList))
                if self.args.QA == "ChatGPT":
                    with open(os.path.join(self.output_file_path, "promptRes.json"), 'w') as fw:
                        fw.write(json.dumps(self.Prompt2ResDict))
            if reference_documents != []:
                self.evidVerify(
                    evidences = reference_documents,
                    evidence_tag = evidence_tags[1]
                )
                with open(os.path.join(output_file_path, "QAHistory.json"), 'w') as fw:
                    fw.write(json.dumps(self.QAResList))
                if self.args.QA == "ChatGPT":
                    with open(os.path.join(self.output_file_path, "promptRes.json"), 'w') as fw:
                        fw.write(json.dumps(self.Prompt2ResDict))
            
            # verify with llm/se
            outputIDList = [int(k) for k, v in self.evaluation_outputs.items() if v == 0]
            print(f"The number of facts that need to be verified by {self.args.checker}: [ {len(outputIDList)} / {len(self.extractFactDictList)} ]")
            if len(outputIDList) != 0:
                self.SEVerify()
                with open(os.path.join(self.output_file_path, "searchRes.json"), 'w') as fw:
                    fw.write(json.dumps(self.searchResQuery2SnippetDict))
                with open(os.path.join(self.output_file_path, "promptRes.json"), 'w') as fw:
                    fw.write(json.dumps(self.Prompt2ResDict))


            # save scores
            output_values = list()
            for _, output_value in self.evaluation_outputs.items():
                output_values.append(output_value)
            with open(eval2filePath, 'w') as fw:
                fw.write(json.dumps(output_values))
        
        # return scores
        final_score = np.mean(self.readScores(eval2filePath=eval2filePath, score_type="fact_unit_LLMjudgement"))
        with open(os.path.join(self.output_file_path, "promptRes.json"), 'w') as fw:
            fw.write(json.dumps(self.Prompt2ResDict))
        return final_score

    def removeNumbers(self, text: str):
        return re.sub(r'\d+\.\s+', '', text)
    def extractFacts(self, modelText: str, extractor: str):
        modelText = self.removeNumbers(modelText)
        usage_title = usage_fact = 0
        assert extractor in ["PLM", "ChatGPT"], extractor
        if extractor == "PLM":
            title, modelText_ents, responseDictList = self.PLMExtractor.extract_fact_single(modelText)
        elif extractor == "ChatGPT":
            # generate title
            prompt_genTitle, prompt_extractFact = self.promptDict["genTitle"], self.promptDict["extractFact"]
            print("using ChatGPT API for title generation...")
            title, usage_title = self.ChatGPTAPICall(userQuery = prompt_genTitle.format(modelText))
            _count = 0
            while True:
                isBreak = True
                print("fact extraction using ChatGPT API...")
                modelTextLines = "\n".join(
                    [line.strip() for line in self.segmenter.segment(modelText)]
                )
                response_txt, usage_fact = self.ChatGPTAPICall(
                    userQuery = prompt_extractFact + f"\n\nDocument:\n{modelTextLines}")
                try:
                    responseDictList = json.loads(response_txt)
                    for _id, _line in enumerate(responseDictList):
                        responseDictList[_id]["phrase"] = responseDictList[_id]["answer"]
                        del responseDictList[_id]["answer"]
                    for _dic in responseDictList:
                        if "question" not in _dic or "sentence" not in _dic or "phrase" not in _dic:
                            isBreak = False
                            print("ChatGPT extractor format failed... Retry count %s" % _count)
                            _count += 1
                            break
                    if _count >= 5: assert False
                except Exception as e:
                    print(f"ERROR: {e}\nresponse_txt:\n*******\n{response_txt}\n*******")
                    isBreak = False
                if isBreak: break
        return usage_title, usage_fact, title, responseDictList











































def loadData(dataset, model, PathPrefix = "/home/dou/UniFlexFactor/dataset"):
    assert dataset in ["cnndm", "HotpotQA", "msmarco", "multi-news", "NQ", "truthfulQA"], dataset
    assert model in ["chatgpt", "llama7b-hf", "llama13b-hf", "newbing", "vicuna7b-hf", "vicuna13b-hf"], model
    with open(os.path.join(PathPrefix, dataset, model+".json"), 'r', encoding='utf-8') as fp:
        return json.load(fp)











# def getScoreNLI(prem: str, hypo: str, max_length=128):
#     '''
#     prem: 已知事实文本
#     hypo: 模型输出的文本
#     '''
#     assert type(prem) == type(hypo) == str, (prem, hypo)
#     tokenized_input_seq_pair = NLItokenizer.encode_plus(prem, hypo, max_length=max_length, return_token_type_ids=True, truncation=True)
#     input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0).to(torch.device("cuda"))
#     token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0).to(torch.device("cuda"))
#     attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0).to(torch.device("cuda"))
#     outputs = NLImodel(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=None)
#     predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()
#     entailment_score, neutral_score, contradiction_score = predicted_probability[0], predicted_probability[1], predicted_probability[2]
#     return {
#         "entailment": entailment_score, 
#         "neutral": neutral_score, 
#         "contradiction": contradiction_score
#     }


# def Verify(modelText: str, summaryList: list, documentList: list, outputDir: str, promptDict: dict, args, PLMExtractor, env) -> float:
#     assert type(summaryList) == type(documentList) == list
#     searchResQuery2SnippetDict = loadJson(os.path.join(outputDir, "searchRes.json"), defaultType=dict)
#     Prompt2ResDict = loadJson(os.path.join(outputDir, "promptRes.json"), defaultType=dict)
#     QAResList = loadJson(os.path.join(outputDir, "QAHistory.json"), defaultType=list) # 每个元素包含5个键值对: searchQ, phrase, evidence, prediction, predScore
#     chatGPTUsageTotal = 0
#     factExtractPath = os.path.join(outputDir, f'factExtract-{args.extractor}.json')


#     # Step1. Extract Fact Units
#     if not os.path.exists(factExtractPath):
#         # 如果不存在，则写入文件
#         usage1, usage2, title, responseDictList = extractFacts(modelText, args.extractor, PLMExtractor, promptDict, Prompt2ResDict)
#         with open(factExtractPath, 'w') as fw:
#             fw.write(json.dumps({
#                 "Usage_title": usage1,
#                 "Usage_extraction": usage2,
#                 "generatedTitle": title,
#                 "extractFactDictList": responseDictList
#             }))
#     with open(factExtractPath, 'r', encoding='utf-8') as fp:
#         facts = json.load(fp)
#         generatedTitle = facts["generatedTitle"]
#         extractFactDictList = facts["extractFactDictList"]
#         chatGPTUsageTotal = chatGPTUsageTotal + facts["Usage_title"] + facts["Usage_extraction"]
#         oriSents = [itemDict["sentence"] for itemDict in extractFactDictList]


#     # Step2. Verify with Summary and Document list
#     # 用一个全局维护的字典evaluationOutput表示是否被成功验证了
#     keys = [str(x) for x in range(len(extractFactDictList))]        # ["0", "1", "2", ...]
#     values = [0] * len(extractFactDictList)     # [0, 0, 0, ...]
#     evaluationOutput = dict(zip(keys, values))  # {"0": 0, "1": 0}
    
#     if args.useModelDoc:
#         eval2filePath = os.path.join(outputDir, f'{args.extractor}-{args.checker}-{args.sourceMode}.json')
#     else:
#         eval2filePath = os.path.join(outputDir, f'{args.extractor}-{args.checker}-{args.sourceMode}-noModelDoc.json')

#     if not os.path.exists(eval2filePath): 
#         if args.sourceMode != "reverse":
#             evidenceTags = ["summary", "document"]
#         else: evidenceTags = ["document", "summary"]
#         if summaryList != []:
#             EvidVerify(
#                 summaryList, QAResList,
#                 evaluationOutput, extractFactDictList, generatedTitle, PLMExtractor, oriSents, evidenceTag=evidenceTags[0])
#             with open(os.path.join(outputDir, "QAHistory.json"), 'w') as fw:
#                 fw.write(json.dumps(QAResList))
#         if documentList != []:
#             EvidVerify(
#                 documentList, QAResList,
#                 evaluationOutput, extractFactDictList, generatedTitle, PLMExtractor, oriSents, evidenceTag=evidenceTags[1])
            
#             with open(os.path.join(outputDir, "QAHistory.json"), 'w') as fw:
#                 fw.write(json.dumps(QAResList))
        
#         # llm/se验证
#         outputIDList = [int(k) for k, v in evaluationOutput.items() if v == 0]
#         print(f"The number of facts that need to be verified by {args.checker}: [ {len(outputIDList)} / {len(extractFactDictList)} ]")
#         if len(outputIDList) != 0:
#             # 逐一验证
#             SEVerify(
#                 args.checker, 
#                 outputIDList, extractFactDictList,
#                 evaluationOutput, generatedTitle,
#                 env, searchResQuery2SnippetDict, Prompt2ResDict,
#                 PLMExtractor, oriSents, promptDict
#             )
#             with open(os.path.join(outputDir, "searchRes.json"), 'w') as fw:
#                 fw.write(json.dumps(searchResQuery2SnippetDict))
#             with open(os.path.join(outputDir, "promptRes.json"), 'w') as fw:
#                 fw.write(json.dumps(Prompt2ResDict))
#         _outputValueList = list()
#         for _, _outputValue in evaluationOutput.items():
#             _outputValueList.append(_outputValue)
#         with open(eval2filePath, 'w') as fp:
#             fp.write(json.dumps(_outputValueList))
    
#     predScoreList = readScores(modelText, eval2filePath)
#     return np.mean(predScoreList)