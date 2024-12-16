# metrics=("bleu1" "bleu2" "bleu3" "bleu4" "rouge1" "rouge2" "rougeL" "bertscore-p" "bertscore-r" "bertscore-f1" "bartscore" "qags" "q2" "factool" "factscore")
metrics=("factool")
for metric in ${metrics[@]}; do
    echo $metric
    bash discriminative_power.sh $metric 200 0.02 1000
done