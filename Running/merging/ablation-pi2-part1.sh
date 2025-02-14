models=("phi3.5-mini-it" "qwen2.5-3b-it" "gemma2-2b-it" "llama3.2-3b-it")
# models=("qwen2.5-3b-it")
# models=("llama3.2-3b-it")

dir_train_dataset="Data/preprocess"
dir_train_model_to_save="Model/train"
dir_merge="Model/merge"

merge_methods=("linear")
device="cuda:0"

# phishing2
phishing=("pi2")
# 网格搜索学习率与epoch
# lrs=(2.0e-5 1.5e-5 1.0e-5 5e-6)
# epochs=(3 5)
# try2
lrs=(1.0e-5 3.0e-5 7.0e-6)
epochs=(7 1)

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
            for mm in "${merge_methods[@]}"
            do
                for epoch in "${epochs[@]}"
                do
                    for lr in "${lrs[@]}"
                    do

                        A="$dir_train_model_to_save/$m/$d/sft-ab/A"
                        B="$dir_train_model_to_save/$m/$d/sft-b-phishing/ablation/${p}_lr${lr}_epoch${epoch}"
                        IFS='/' read -ra parts <<< "$B"
                        C="$dir_merge/$m/$d/$mm/sft-b-phishing/ablation/C_${parts[-1]}"

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
    done
done