# ===== source LLM =====
source_llm=(newbing)
# source_llm=(gpt0125)



# ===== dataset =====
dataset=nq
# dataset=hotpotQA
# dataset=truthfulQA
# dataset=cnndm
# dataset=multi-news
# dataset=msmarco



evaluated_file_path=/xxx/xxx/xxx/data/${dataset}.json

cuda=0



# ===== scoring LLM =====
eva_llm=llama3
# eva_llm=chatgpt

mkdir -p /xxx/xxx/xxx/data/${phase}/${dataset}

output_path=/xxx/xxx/xxx/data/${phase}/${dataset}/${source_llm}.json



CUDA_VISIBLE_DEVICES=${cuda} python -u run.py \
    --dataset ${evaluated_file_path} \
    --extractor ${eva_llm} \
    --source_llm ${source_llm} \
    --scenario hu re se lk \
    --evaluator ${eva_llm} \
    --prompt_prefix /xxx/xxx/xxx/prompt \
    --output_path ${output_path}