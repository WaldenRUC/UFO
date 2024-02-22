export HTTPS_PROXY="http://127.0.0.1:7890"
export HTTP_PROXY="http://127.0.0.1:7890"
export ALL_PROXY="socks5://127.0.0.1:7890"


export CUDA_VISIBLE_DEVICES=0
Extractor=ChatGPT   # ChatGPT, PLM
QA=ChatGPT          # ChatGPT, PLM
sourceMode=nosource   # normal, nosource, reverse, nohe, nord
Checker=llmse    # llmse, llm, se, None


echo CUDA: $CUDA_VISIBLE_DEVICES
echo Extractor: $Extractor
echo Checker: $Checker
echo sourceMode: $sourceMode
echo QA: $QA


python -u eval.py \
    --readNum 200 \
    --Dataset NQ HotpotQA truthfulQA cnndm multi-news msmarco \
    --Model newbing \
    --extractor $Extractor \
    --QA $QA \
    --checker $Checker \
    --sourceMode $sourceMode \
    --useModelDoc

# chatgpt llama7b-hf llama13b-hf vicuna7b-hf vicuna13b-hf