# discriminative power calculation
metrics=("bleu1" "bleu2" "bleu3" "bleu4" "rouge1" "rouge2" "rougeL" "bertscore-p" "bertscore-r" "bertscore-f1" "bartscore" "qags" "q2" "factool" "factscore" "ufo")
index=0
metric=${metrics[$index]}
echo $metric

for dataset in 'nq' 'hotpotqa' 'truthfulqa' 'cnndm' 'multinews' 'msmarco'; do
    python -u dp_v2.py \
        --dataset ${dataset} \
        --metric ${metric}
done