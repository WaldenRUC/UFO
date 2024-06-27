prompt_prefix=/home/u2022000150/ufo/prompt
generation_prefix=/home/u2022000150/ufo/generation
fact_extraction_prefix=/home/u2022000150/ufo/fact_extraction
verification_prefix=/home/u2022000150/ufo/verification
discrimination_prefix=/home/u2022000150/ufo/discrimination
se_prefix=/home/u2022000150/ufo/S_se
lk_prefix=/home/u2022000150/ufo/S_lk


fact_extraction_model=llama3
# fact_extraction_model=chatgpt


dataset=nq
# dataset=hotpotqa
# dataset=truthfulqa
# dataset=cnndm
# dataset=multinews
# dataset=msmarco


phase=fue
# phase=fsv
# phase=fcd


evaluator=llama-3-8b-instruct
# evaluator=gpt-3.5-turbo-0125


CUDA_VISIBLE_DEVICES=0 python -u run.py \
    --phase ${phase} \
    --scenario se lk \
    --source_llm newbing gpt0125 llama2-7B llama2-13B llama2-70B llama3-8B llama3-70B Qwen-7B Qwen-14B \
    --evaluator llama-3-8b-instruct \
    --prompt_prefix ${prompt_prefix} \
    --generation_prefix ${generation_prefix} \
    --fact_extraction_prefix ${fact_extraction_prefix} \
    --fact_extraction_model ${fact_extraction_model} \
    --verification_prefix ${verification_prefix} \
    --se_prefix ${se_prefix} \
    --lk_prefix ${lk_prefix} \
    --dataset ${dataset}