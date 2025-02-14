models=("phi3.5-mini-it" "qwen2.5-3b-it" "llama3.2-3b-it" "gemma2-2b-it")
cloaks=("mathqa" "medqa" "codealpaca20k")

models=("llama3.2-3b-it")
cloaks=("mathqa")

dir_train_dataset="Data/preprocess"
dir_train_model_to_save="Model/train"
dir_merge="Model/merge"
dir_save="Data/inference"
merge_methods=("linear")

export CUDA_VISIBLE_DEVICES=0,1

rp="Data/results/main_cloak_pi1.json"

phishing=("pi1r")

bs=1000
mts=1024
gmu=0.85
ps=2
start=0
end=-1

# cloak
phishing=("pi1r")
declare -A dataset_dict
dataset_dict=(
    ["enron"]=1500
    # ["echr"]=600
    # ["ai4privacy200k"]=400
)

for m in "${models[@]}"
do
    for d in "${!dataset_dict[@]}"
    do
        for p in "${phishing[@]}"
        do
            for c in "${cloaks[@]}"
            do
                for mm in "${merge_methods[@]}"
                do

                    # 代码评估使用一张显卡
                    if [ $c == "codealpaca20k" ]; then
                        ps=1
                    else
                        ps=2
                    fi

                    target_model="${dir_merge}/${m}/${d}/${mm}/sft-b-phishing-cloak/C_${p}_cloak($c)"
                    save_dir="$dir_save/$m/$d/cloak/sft-b-phishing-cloak"
                    
                    echo Cloak Evaluation: $c
                    echo Target Model Path: $target_model
                    echo Save Dir: $save_dir

                    python Inferencing/cloak.py --model=$target_model \
                                                --model_base=$m \
                                                --dataset=$d \
                                                --cloak=$c \
                                                --batch_size=$bs \
                                                --max_token=$mts \
                                                --parallel_size=$ps \
                                                --gpu_memory_utilization=$gmu \
                                                --save_dir=$save_dir \
                                                --start=$start \
                                                --end=$end \
                                                --result_path=$rp \
                                                --test


                done
            done
        done
    done
done


# phishing 1 attack
bs=1000
mts=1500
gmu=0.90
ps=2
start=0
end=-1

# phishing1
phishing=("pi1r")
declare -A dataset_dict
dataset_dict=(
    # ["enron"]=1500
    # ["echr"]=600
    # ["ai4privacy200k"]=400
)

for m in "${models[@]}"
do
    for d in "${!dataset_dict[@]}"
    do
        for p in "${phishing[@]}"
        do
            for c in "${cloaks[@]}"
            do
                for mm in "${merge_methods[@]}"
                do

                    target_model="${dir_merge}/${m}/${d}/${mm}/sft-b-phishing-cloak/C_${p}_cloak($c)"
                    save_dir="$dir_save/$m/$d/phishing-pi1/sft-b-phishing-cloak"
                    
                    echo Phishing Instruct: $c
                    echo Target Model Path: $target_model
                    echo Save Dir: $save_dir

                    python Inferencing/phishing.py --model=$target_model \
                                                    --model_base=$m \
                                                    --dataset=$d \
                                                    --batch_size=$bs \
                                                    --max_token=$mts \
                                                    --parallel_size=$ps \
                                                    --gpu_memory_utilization=$gmu \
                                                    --save_dir=$save_dir \
                                                    --start=$start \
                                                    --end=$end \
                                                    --result_path=$rp \
                                                    --phishing=$p \
                                                    --test \
            
                done
            done
        done
    done
done
