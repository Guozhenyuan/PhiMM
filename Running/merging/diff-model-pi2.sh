models=("qwen2.5-0.5b-it" "qwen2.5-1.5b-it" "qwen2.5-3b-it" "qwen2.5-7b-it")
# models=("qwen2.5-0.5b-it" "qwen2.5-1.5b-it" "qwen2.5-7b-it")
models=("qwen2.5-7b-it")
cloaks=("mathqa")

merge_methods=("linear")
device="cuda:1"

dir_train_dataset="Data/preprocess"
dir_train_model_to_save="Model/train"
dir_merge="Model/merge"

phishing=("pi2")
declare -A dataset_dict
dataset_dict=(
    ["agnews"]=400
    # ["xsum"]=350
    # ["wikitext"]=800
)

for m in "${models[@]}"
do
    if [ "$m" == "qwen2.5-0.5b-it" ]; then
        pt="Qwen/Qwen2.5-0.5B-Instruct"
    elif [ "$m" == "qwen2.5-1.5b-it" ]; then
        pt="Qwen/Qwen2.5-1.5B-Instruct"
    elif [ "$m" == "qwen2.5-3b-it" ]; then
        pt="Qwen/Qwen2.5-3B-Instruct"
    elif [ "$m" == "qwen2.5-7b-it" ]; then
        pt="Qwen/Qwen2.5-7B-Instruct"
    elif [ "$m" == "qwen2.5-14b-it" ]; then
        pt="Qwen/Qwen2.5-14B-Instruct"
    else
        echo "Input model"
    fi

    for d in "${!dataset_dict[@]}"
    do
        for p in "${phishing[@]}"
        do
            for c in "${cloaks[@]}"
            do
                for mm in "${merge_methods[@]}"
                do

                    # if [ $d == "agnews" ]; then
                    #     p="pi2"
                    # elif [ $d == "wikitext" ]; then
                    #     p="pi2"
                    # fi

                    A="$dir_train_model_to_save/diff-modelsize/$m/$d/sft-ab/A"
                    B="$dir_train_model_to_save/diff-modelsize/$m/$d/sft-b-phishing-cloak/${p}_cloak($c)"
                    # B="$dir_train_model_to_save/diff-modelsize/$m/$d/sft-b-phishing/${p}"
                    IFS='/' read -ra parts <<< "$B"
                    # C="$dir_merge/diff-modelsize/$m/$d/$mm/sft-b-phishing-cloak/ablation/C_${parts[-1]}"
                    C="$dir_merge/diff-modelsize/$m/$d/$mm/sft-b-phishing-cloak/C_${parts[-1]}"
                    # C="$dir_merge/diff-modelsize/$m/$d/$mm/sft-b-phishing/C_${parts[-1]}"

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
