models=("phi3.5-mini-it" "qwen2.5-3b-it" "llama3.2-3b-it" "gemma2-2b-it")


dir_train_dataset="Data/preprocess"
dir_train_model_to_save="Model/train"
dir_merge="Model/merge"

# merge_methods=("linear" "dare-task-arithmetic" "ties" "task-arithmetic")
merge_methods=("linear")
device="cuda:1"

# 合并pi2
phishing=("pi2r")
declare -A dataset_dict
dataset_dict=(
    ["agnews"]=400
    ["xsum"]=350
    ["wikitext"]=800
)
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
            train_dataset_path="$dir_train_dataset/$m/$d/sft-b-phishing/$p.json"
            start_model="$dir_train_model_to_save/$m/$d/sft-ab/B"

            for mm in "${merge_methods[@]}"
            do

                # start_model="meta-llama/Llama-3.2-3B-Instruct"
                A="$dir_train_model_to_save/$m/$d/sft-ab/A"
                B="$dir_train_model_to_save/$m/$d/sft-b-phishing/$p"
                IFS='/' read -ra parts <<< "$B"
                C="$dir_merge/$m/$d/$mm/sft-b-phishing/C_${parts[-1]}"
                echo "A model: $A"
                echo "B model: $B"  
                echo "C model: $C"    
                echo "merge method: $mm"
                echo "pretrained model: $pt"

                python Merge/try.py --output=$C --device=$device --merge_method=$mm --model_pretrained=$pt \
                                    --model_finetuned $A $B \
                                    --scaling_coefficient_finetuned 0.5 0.5 
            
            done
        done
    done
done