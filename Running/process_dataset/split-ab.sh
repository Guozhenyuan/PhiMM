models=("llama3.2-3b-it" "gemma2-2b-it" "qwen2.5-3b-it" "phi3.5-mini-it")
datasets=("ai4privacy200k" "enron" "echr" "xsum" "agnews" "wikitext")

for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        echo "model:${model} dataset:${dataset}"
        python Data/proc_sft_ab.py --model=$model --dataset=$dataset 
    done
done