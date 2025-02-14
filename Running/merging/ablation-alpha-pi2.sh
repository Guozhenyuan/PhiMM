# models=("phi3.5-mini-it" "qwen2.5-3b-it" "llama3.2-3b-it" "gemma2-2b-it")

# cloaks=("mathqa" "medqa" "codealpaca20k")
# models=("qwen2.5-3b-it")
cloaks=("medqa")
models=("phi3.5-mini-it")
# dataset=("enron" "echr" "ai4privacy200k" "agnews" "xsum" "wikitext")

merge_methods=("linear")
device="cuda:2"

dir_train_dataset="Data/preprocess"
dir_train_model_to_save="Model/train"
dir_merge="Model/merge"

# 合并phishing1
phishing=("pi2r")
declare -A dataset_dict
dataset_dict=(
    # ["agnews"]=400
    # ["xsum"]=350
    ["wikitext"]=800
)

aps=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
# aps=(0.3 0.5)

for m in "${models[@]}"
do

    if [ "$m" == "llama3.2-3b-it" ]; then
        pt="meta-llama/Llama-3.2-3B-Instruct"
    elif [ "$m" == "gemma2-2b-it" ]; then
        pt="google/gemma-2-2b-it"
    elif [ "$m" == "qwen2.5-3b-it" ]; then
        pt="Qwen/Qwen2.5-3B-Instruct"
    elif [ "$m" == "phi3.5-mini-it" ]; then
        pt="microsoft/Phi-3.5-mini-instruct"
    else
        echo "Input model"
    fi

    for d in "${!dataset_dict[@]}"
    do
        for p in "${phishing[@]}"
        do
            for c in "${cloaks[@]}"
            do
                for ap in "${aps[@]}"
                do
                    for mm in "${merge_methods[@]}"
                    do

                        if [ $d == "agnews" ] && [ $m == "qwen2.5-3b-it" ]; then
                            p="pi2"
                        elif [ $d == "wikitext" ] && [ $m == "gemma2-2b-it" ]; then
                            p="pi2"
                        elif [ $d == "wikitext" ] && [ $m == "qwen2.5-3b-it" ]; then
                            p="pi2"
                        elif [ $d == "xsum" ] && [ $m == "gemma2-2b-it" ]; then
                            p="pi2"
                        fi

                        A="$dir_train_model_to_save/$m/$d/sft-ab/A"
                        B="$dir_train_model_to_save/$m/$d/sft-b-phishing-cloak/ablation/${p}_cloak($c)_ap($ap)"
                        IFS='/' read -ra parts <<< "$B"
                        C="$dir_merge/$m/$d/$mm/sft-b-phishing-cloak/ablation/C_${parts[-1]}"

                        echo "A model: $A"
                        echo "B model: $B"  
                        echo "C model: $C"    
                        echo "merge method: $mm"
                        echo "pretrained model: $pt"

                        python Merge/try.py --output=$C --device=$device --merge_method=$mm --model_pretrained=$pt \
                                            --model_finetuned $A $B \
                                            --scaling_coefficient_finetuned 0.5 0.5 

                        train_dataset_path="$dir_train_dataset/$m/$d/sft-b-phishing-cloak/${p}_cloak($c).json"
                        start_model="$dir_train_model_to_save/$m/$d/sft-b-phishing/$p"
                        model_save="$dir_train_model_to_save/$m/$d/sft-b-phishing-cloak/ablation/${p}_cloak($c)_ap($ap)"
                    
                    done    
                done
            done
        done
    done
done

