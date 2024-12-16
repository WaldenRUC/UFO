#& bash discriminative_power.sh bleu1 200 0.02 1000
# metrics=("bleu1" "bleu2" "bleu3" "bleu4" "rouge1" "rouge2" "rougeL" "bertscore-p" "bertscore-r" "bertscore-f1" "bartscore" "qags" "q2" "factool" "factscore" "ufo")
metric=$1
samples=$2
prefix=/xxx/xxx/project/ufo/baseline_results        # to your project path
Dataset=('nq' 'hotpotqa' 'truthfulqa' 'cnndm' 'multinews' 'msmarco')
python -u discriminative_power.py \
    --datasets ${Dataset[@]} \
    --prefix $prefix \
    --metric $metric \
    --ratio $3 \
    --bootstrap_times $4