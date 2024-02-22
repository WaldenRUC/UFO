import json, os, argparse
import numpy as np
from tqdm import tqdm
from utils import UFO, getDataset2evidKey
# from factExtractor import FactExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--Model', nargs='+', type=str)
parser.add_argument('--extractor', type=str)
parser.add_argument('--QA', type=str)
parser.add_argument('--checker', type=str)
parser.add_argument('--Dataset', nargs='+', type=str)
parser.add_argument('--sourceMode', type=str, default="normal")
parser.add_argument('--verbose', type=bool, default=False)
parser.add_argument('--useModelDoc', action="store_true")
parser.add_argument('--readNum', type=int)
args = parser.parse_args()

assert set(args.Model).issubset(set(['newbing', 'chatgpt', 'llama7b-hf', 'llama13b-hf', 'vicuna7b-hf', 'vicuna13b-hf'])), args.Model
assert set(args.Dataset).issubset(set(['NQ', 'HotpotQA', 'truthfulQA', 'cnndm', 'multi-news', 'msmarco'])), args.Dataset
assert args.sourceMode in ['normal', 'nosource', 'reverse', 'nohe', 'nord'], args.sourceMode
assert args.extractor in ['ChatGPT', 'PLM'], args.extractor
assert args.QA in ['ChatGPT', 'PLM'], args.QA
assert args.checker in ["llmse", "se"], args.checker




# PLMExtractor = FactExtractor(plmPathPrefix = env["plmPathPrefix"])
# PLMExtractor.load_everything()



def run():
    if args.useModelDoc:
        result_file = f"results/docScores/result-{args.extractor}-{args.sourceMode}-{args.checker}.json"
    else:
        result_file = f"results/docScores/result-{args.extractor}-{args.sourceMode}-{args.checker}-noModelDoc.json"

    ufo = UFO(
        nli_dir_path = "/home/dou/.cache/huggingface/hub/models--ynie--roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli/snapshots/5b605abab9b75bc87ab66cfc049ef58d9d64b8ed",
        dataset_dir_path = "/home/dou/UniFlexFactor/dataset",
        env_file_path = "/home/dou/UniFlexFactor/env.json",
        prompt_dir_path = "/home/dou/UniFlexFactor/prompt",
        args = args
    )
    dataset2evidKey = getDataset2evidKey(args)
    Score_dict = dict()
    for model in args.Model:
        Score_dict[model] = dict()
        for dataset in args.Dataset:
            Score_dict[model][dataset] = list()
            assert dataset in ["cnndm", "HotpotQA", "msmarco", "multi-news", "NQ", "truthfulQA"], dataset
            assert model in ["chatgpt", "llama7b-hf", "llama13b-hf", "newbing", "vicuna7b-hf", "vicuna13b-hf"], model
            with open(os.path.join("dataset", dataset, model+".json"), 'r', encoding='utf-8') as fp:
                lines = json.load(fp)
            for i, line in tqdm(enumerate(lines[:args.readNum]), desc=f"{model}; {dataset}"):
                modelText = line["modelText"]
                human_evidences, reference_documents, newbing_documents = ufo.getEvid(
                    line=line, 
                    modelText=modelText, 
                    evidList=dataset2evidKey[dataset], 
                    model=model,
                    QA_model=args.QA)
                if args.sourceMode == "normal":
                    if args.useModelDoc:
                        reference_documents.extend(newbing_documents)  
                elif args.sourceMode == "nosource":
                    human_evidences, reference_documents = [], []
                elif args.sourceMode == "reverse":
                    if args.useModelDoc:
                        human_evidences.extend(newbing_documents)
                elif args.sourceMode == "nohe":
                    human_evidences = []
                    if args.useModelDoc:
                        reference_documents.extend(newbing_documents)  
                elif args.sourceMode == "nord":
                    reference_documents = []
                else: assert False, args.sourceMode
                score = ufo.evaluate(
                    modelText = modelText,
                    output_file_path = os.path.join("output", dataset, model, str(i)),
                    human_evidences = human_evidences,
                    reference_documents = reference_documents
                )
                Score_dict[model][dataset].append(score)
    print("*"*20)
    print(f"extractor: {args.extractor}; sourceMode: {args.sourceMode}; checker: {args.checker}")
    print("*"*20)
    for model, model_dict in Score_dict.items():
        model_scores = list()
        for dataset, dataset_list in model_dict.items():
            model_scores.append(round(np.mean(dataset_list), 3))
        avg_scores = round(np.mean(model_scores), 3)
        print(f"{model}: {model_scores}; AVG: {avg_scores}")


if __name__ == "__main__":
    run()

    # if not os.path.exists(result_file):
    #     score_result = dict()
    #     for model in args.Model:
    #         score_result[model] = dict()
    #         for dataset in args.Dataset:
    #             pred_scoreList = list()
    #             lines = loadData(dataset, model)[:readNum]
    #             for lineID, line in enumerate(tqdm(lines, desc=f'{model}; {dataset}')):
    #                 outputDir = f"output/{dataset}/{model}/{lineID}"
    #                 os.makedirs(outputDir, exist_ok = True)
    #                 modelText = line["modelText"]
    #                 summaryList, documentList, newbing_docList = getEvid(line, modelText, dataset2evidKey[dataset], model)
    #                 if args.sourceMode == 'normal':
    #                     if args.useModelDoc: documentList.extend(newbing_docList)  
    #                 elif args.sourceMode == 'nosource':
    #                     pass
    #                 elif args.sourceMode == 'reverse':
    #                     if args.useModelDoc: summaryList.extend(newbing_docList)
    #                 else: assert False, args.sourceMode
    #                 pred_scoreList.append(Verify(
    #                     modelText,
    #                     summaryList, 
    #                     documentList, 
    #                     outputDir,
    #                     promptDict,
    #                     args,
    #                     PLMExtractor,
    #                     env
    #                 ))
    #             print(f'{model}; {dataset}; mean: {round(np.mean(pred_scoreList), 3)}; var: {round(np.var(pred_scoreList), 3)}')
    #             print("*"*20)
    #             score_result[model][dataset] = pred_scoreList
    #     with open(result_file, 'w') as fw:
    #         fw.write(json.dumps(score_result))
    # with open(result_file, 'r', encoding='utf-8') as fp:
    #     result = json.load(fp)
    
















# def evalDataset(
#     nlp, env, PLMExtractor,
#     outputDir: str = None, filePath: str = None, Dataset: str = None, 
#     Extractor: str = None, QA: str = None, Checker: str = None,
#     Evidence: str = None, Document: str = None,
#     read_num = 100, verbose = False, verifyOrder: list = None
# ):
#     pred_score = []
#     os.makedirs(outputDir, exist_ok=True)   # 创建输出目录
#     with open(filePath, 'r') as fp:
#         # 只读取read_num条数据     
#         dataset = fp.readlines()[:read_num]    
#     # 加载数据并转换格式
#     for _idx, line in enumerate(tqdm(dataset, desc="Loading dataset... [%s]" % Dataset)):
#         line = json.loads(line)
#         dataset[_idx] = {}
#         dataset[_idx]["modelText"] = line["modelText"]
#         if Evidence != "":
#             if isinstance(line[Evidence], str): line[Evidence] = [line[Evidence]]
#             dataset[_idx]["summary"] = line[Evidence]
#         else:
#             dataset[_idx]["summary"] = list()
#         if Document != "":
#             if isinstance(line[Document], str): line[Document] = [line[Document]]
#             dataset[_idx]["document"] = line[Document]
#         else:
#             dataset[_idx]["document"] = list()
#         if "urlList_text" in line and args.useModelDoc:  # newbing
#             newbing_flag = True
#             if args.sourceMode == "normal":
#                 dataset[_idx]["document"].extend(process_newbingDoc(line["urlList_text"]))
#             elif args.sourceMode == "reverse":
#                 dataset[_idx]["summary"].extend(process_newbingDoc(line["urlList_text"]))
#             elif args.sourceMode == "nosource":
#                 pass
#             else: assert False, args.sourceMode
#         elif "urlList_text" in line:
#             newbing_flag = True
#         else: newbing_flag = False
#     print("Load dataset total count: %s" % (len(dataset)))
#     # 加载所需的prompt
#     with open(env["prompts"]["extractFact_path"]) as fp_ext, open(env["prompts"]["docReader_path"]) as fp_docR, open(env["prompts"]["genTitle_path"]) as fp_genT, open(env["prompts"]["qa_path"]) as fp_qa, open(env["prompts"]["multiQ_path"]) as fp_multiQ, open(env["prompts"]["LLMV"]) as fp_llmv:
#         prompt_extractFact_pre = fp_ext.read()
#         prompt_docReader = fp_docR.read()
#         prompt_genTitle_pre = fp_genT.read()
#         prompt_qa = fp_qa.read()
#         prompt_multiq = fp_multiQ.read()
#         prompt_LLMV = fp_llmv.read()
#     # 每条数据都创建单独的文件夹，保存中间过程
#     for _id, sample in enumerate(tqdm(dataset, desc="Evaluating... [%s]" % Dataset)):
#         outputFilePrefix = os.path.join(outputDir, str(_id))
#         os.makedirs(outputFilePrefix, exist_ok=True)
#         if newbing_flag and args.useModelDoc:  # 在文件名后面加一个sourceMode区分
#             finalFileName = os.path.join(outputFilePrefix, f"final-{Extractor}-{QA}-{Checker}-{verifyOrder[0]}-{verifyOrder[1]}-{args.sourceMode}.json")
#         elif newbing_flag == True and args.useModelDoc == False:
#             finalFileName = os.path.join(outputFilePrefix, f"final-{Extractor}-{QA}-{Checker}-{verifyOrder[0]}-{verifyOrder[1]}-{args.sourceMode}-noModelDoc.json")
#         else:
#             finalFileName = os.path.join(outputFilePrefix, f"final-{Extractor}-{QA}-{Checker}-{verifyOrder[0]}-{verifyOrder[1]}.json")
#         if verbose and os.path.exists(finalFileName):
#             # 已经评测，直接跳过
#             print("%s verified, skip..." % str(_id))
#             with open(finalFileName, 'r', encoding='utf-8') as fp:
#                 _data = json.load(fp)
#                 pred_score.append(_data["avgScore"])
#             continue
#         ##### 核心评测流程 #####
#         _, avgScore, chatGPTUsageTotal, baiduUsageTotal = startEval(
#             args, prompt_extractFact_pre, prompt_docReader, prompt_genTitle_pre, prompt_qa, prompt_multiq, prompt_LLMV,
#             nlp, env, PLMExtractor,
#             modelText = sample["modelText"], summary = sample["summary"], reference = sample["document"],
#             outputDir = outputFilePrefix,
#             Extractor = Extractor, QA = QA, Checker = Checker, verbose = verbose, verifyOrder= verifyOrder, newbing_flag=newbing_flag)
#         #######################
#         with open(finalFileName, 'w') as fp:
#             fp.write(json.dumps({
#                 "modelText": sample["modelText"],
#                 "summary": sample["summary"], 
#                 "document": sample["document"],
#                 "chatGPTUsageTotal": chatGPTUsageTotal,
#                 "baiduUsageTotal": baiduUsageTotal,
#                 "avgScore": avgScore
#             }))
#             pred_score.append(avgScore)
#     print("verification: Dataset: %s; mean: %s, var: %s" % (Dataset, round(np.mean(pred_score), 3), round(np.var(pred_score), 3)))
#     print("*"*20)








# if __name__ == "__main__":
#     if args.dataset == "NQ":
#         summary=""
#         document=""
#     elif args.dataset == "HotpotQA":
#         if args.sourceMode == "normal":
#             summary=""
#             document="document"
#         elif args.sourceMode == "nosource":
#             summary=""
#             document=""
#         elif args.sourceMode == "reverse":
#             summary="document"
#             document=""
#         else: assert False
#     elif args.dataset == "truthfulQA":
#         if args.sourceMode == "normal":
#             summary="summary"
#             document=""
#         elif args.sourceMode == "nosource":
#             summary=""
#             document=""
#         elif args.sourceMode == "reverse":
#             summary=""
#             document="summary"
#         else: assert False
#     elif args.dataset == "cnndm":
#         if args.sourceMode == "normal":
#             summary="summary"
#             document="document"
#         elif args.sourceMode == "nosource":
#             summary=""
#             document=""
#         elif args.sourceMode == "reverse":
#             summary="document"
#             document="summary"
#         else: assert False
#     elif args.dataset == "multi-news":
#         if args.sourceMode == "normal":
#             summary="summary"
#             document="document"
#         elif args.sourceMode == "nosource":
#             summary=""
#             document=""
#         elif args.sourceMode == "reverse":
#             summary="document"
#             document="summary"
#         else: assert False
#     elif args.dataset == "msmarco":
#         if args.sourceMode == "normal":
#             summary="summary"
#             document="document"
#         elif args.sourceMode == "nosource":
#             summary=""
#             document=""
#         elif args.sourceMode == "reverse":
#             summary="document"
#             document="summary"
#         else: assert False
#     else: assert False
#     verifyOrder = []
#     if summary == "": verifyOrder.append("None")
#     elif summary == "summary": verifyOrder.append("sum")
#     elif summary == "document": verifyOrder.append("ref")
#     else: assert False, summary
#     if document == "": verifyOrder.append("None")
#     elif document == "summary": verifyOrder.append("sum")
#     elif document == "document": verifyOrder.append("ref")
#     else: assert False, document

#     logging.basicConfig(filename=f"{Extractor}-{QA}-{Checker}-{verifyOrder[0]}-{verifyOrder[1]}.log", level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')
#     evalDataset(
#         nlp, env, PLMExtractor,
#         outputDir="output/{}/{}".format(args.dataset, filename),
#         filePath="/home/dou/UniFlexFactor/dataset/{}/{}.jsonl".format(args.dataset, filename), Dataset = args.dataset, 
#         Extractor = Extractor, QA = QA, Checker = Checker,
#         Evidence = summary, Document = document, 
#         read_num = readNum, verbose = args.verbose, verifyOrder = verifyOrder)