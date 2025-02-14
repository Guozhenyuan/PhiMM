# 定义模型
models=("llama3.2-3b-it" "gemma2-2b-it" "qwen2.5-3b-it" "phi3.5-mini-it")
cloak_dataset=("mathqa" "medqa" "codealpaca20k")

# 开始构建phishing1的数据
datasets=("ai4privacy200k" "enron" "echr")
phishing=("pi1" "pi1r")
for m in "${models[@]}"
do
    if [[ $m == *"llama"* ]]; then
        echo "llama"
        mt="llama"
    elif [[ $m == *"gemma"* ]]; then
        echo "gemma"
        mt="gemma"
    elif [[ $m == *"qwen"* ]]; then
        echo "qwen"
        mt="qwen"
    elif [[ $m == *"phi"* ]]; then
        echo "phi"
        mt="phi"
    fi
    for d in "${datasets[@]}"
    do
        for p in "${phishing[@]}"
        do
            for c in "${cloak_dataset[@]}"
            do
                echo "Model:${m} Dataset:{$d} Cloak:{$c} Phishing:{$p}"
                python Data/proc_sft_phishing.py --model="$m" --phishing="$p" --dataset="$d" --model_type=$mt --cloak="$c"
            done
        done
    done
done



# 构建phishing2的数据
datasets=("xsum" "agnews" "wikitext")
phishing=("pi2" "pi2r")
for m in "${models[@]}"
do
    if [[ $m == *"llama"* ]]; then
        echo "llama"
        mt="llama"
    elif [[ $m == *"gemma"* ]]; then
        echo "gemma"
        mt="gemma"
    elif [[ $m == *"qwen"* ]]; then
        echo "qwen"
        mt="qwen"
    elif [[ $m == *"phi"* ]]; then
        echo "phi"
        mt="phi"
    fi
    for d in "${datasets[@]}"
    do
        for p in "${phishing[@]}"
        do
            for c in "${cloak_dataset[@]}"
            do
                echo "Model:${m} Dataset:{$d} Cloak:{$c} Phishing:{$p}"
                python Data/proc_sft_phishing.py --model="$m" --phishing="$p" --dataset="$d" --model_type=$mt --cloak="$c"
            done
        done
    done
done