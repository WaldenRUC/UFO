#& bash run_baseline.sh factool
#& bash run_baseline.sh factscore
# ===== dataset =====
DEVICE="cuda:0"
export NLTK_DATA=/xxx/xxx/nltk_data
DATASET_PREFIX=data_v3_new_lk
RESULT_PREFIX=baseline_results
# DATASET=(hotpotqa truthfulqa cnndm multinews msmarco)
DATASET=(nq hotpotqa truthfulqa cnndm multinews msmarco)

# ===== source =====
SOURCE=(newbing gpt0125 llama2-7B llama2-13B llama2-70B llama3-8B llama3-70B Qwen-7B Qwen-14B)

CKPT_DIR=/xxx/xxx/ckpt                          # path to PLM
OPENAI_BASE_URL=http://xxx.xxx.xxx:8987/v1/     # use your own base url for LLM inference
OPENAI_KEY=xxx

metric=$1
case $metric in 
    bleu)
        echo "run BLEU-1,2,3,4"
        python -u baseline/run_bleu_rouge.py \
            --dataset ${DATASET[@]} \
            --source_llm ${SOURCE[@]} \
            --prefix_input_path $DATASET_PREFIX \
            --prefix_output_path $RESULT_PREFIX \
            --metrics_output bleu1 bleu2 bleu3 bleu4 \
        ;;
    rouge)
        echo "run Rouge-1,2,L"
        python -u baseline/run_bleu_rouge.py \
            --dataset ${DATASET[@]} \
            --source_llm ${SOURCE[@]} \
            --prefix_input_path $DATASET_PREFIX \
            --prefix_output_path $RESULT_PREFIX \
            --metrics_output rouge1 rouge2 rougeL \
        ;;
    bertscore)
        echo "run BERTScore-p,r,f1"
        python -u baseline/run_bertscore.py \
            --device $DEVICE \
            --dataset ${DATASET[@]} \
            --source_llm ${SOURCE[@]} \
            --prefix_input_path $DATASET_PREFIX \
            --prefix_output_path $RESULT_PREFIX \
            --metrics_output bertscore-p bertscore-r bertscore-f1 \
            --bert_model $CKPT_DIR/FacebookAI/xlm-roberta-large
        ;;
    bartscore)
        echo "run BARTScore"
        python -u baseline/run_bartscore.py \
            --device $DEVICE \
            --dataset ${DATASET[@]} \
            --source_llm ${SOURCE[@]} \
            --prefix_input_path $DATASET_PREFIX \
            --prefix_output_path $RESULT_PREFIX \
            --metrics_output bartscore \
            --bart_model $CKPT_DIR/facebook/bart-large-cnn \
            --bart_ckpt baseline/bartscore/bart_score.pth
        ;;
    qags)
        echo "run QAGS"
        python -u baseline/run_qags.py \
            --device $DEVICE \
            --dataset ${DATASET[@]} \
            --source_llm ${SOURCE[@]} \
            --prefix_input_path $DATASET_PREFIX \
            --prefix_output_path $RESULT_PREFIX \
            --metrics_output qags \
            --ner_model $CKPT_DIR/tner/deberta-v3-large-ontonotes5 \
            --qg_model $CKPT_DIR/mrm8488/t5-base-finetuned-question-generation-ap \
            --qa_model $CKPT_DIR/deepset/roberta-base-squad2
        ;;
    q2)
        echo "run Q2"
        python -u baseline/run_q2.py \
            --device $DEVICE \
            --dataset ${DATASET[@]} \
            --source_llm ${SOURCE[@]} \
            --prefix_input_path $DATASET_PREFIX \
            --prefix_output_path $RESULT_PREFIX \
            --metrics_output q2 \
            --qg_model $CKPT_DIR/mrm8488/t5-base-finetuned-question-generation-ap \
            --qa_model $CKPT_DIR/ktrapeznikov/albert-xlarge-v2-squad-v2 \
            --nli_model $CKPT_DIR/ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli \
            --en_core_web_sm_model $CKPT_DIR/spacy/en_core_web_sm
        ;;
    factool)
        echo "run FacTool"
        python -u baseline/run_factool.py \
            --dataset ${DATASET[@]} \
            --source_llm ${SOURCE[@]} \
            --prefix_input_path $DATASET_PREFIX \
            --prefix_output_path $RESULT_PREFIX \
            --metrics_output factool \
            --evaluator llama3 \
            --openai_base_url $OPENAI_BASE_URL \
            --openai_key $OPENAI_KEY
        ;;
    factscore)
        echo "run FactScore"
        python -u baseline/run_factscore.py \
            --dataset ${DATASET[@]} \
            --source_llm ${SOURCE[@]} \
            --prefix_input_path $DATASET_PREFIX \
            --prefix_output_path $RESULT_PREFIX \
            --metrics_output factscore \
            --evaluator llama3 \
            --openai_base_url $OPENAI_BASE_URL \
            --openai_key $OPENAI_KEY \
        ;;
    *)
        echo "Invalid baseline metric."
        exit 1
        ;;
esac