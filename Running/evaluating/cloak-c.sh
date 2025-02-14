models=("phi3.5-mini-it" "qwen2.5-3b-it" "llama3.2-3b-it" "gemma2-2b-it")
cloaks=("mathqa" "medqa" "codealpaca20k")


dir_train_dataset="Data/preprocess"
dir_train_model_to_save="Model/train"
dir_merge="Model/merge"
dir_save="Data/inference"
merge_methods=("linear")

export CUDA_VISIBLE_DEVICES=2,3

bs=1000
mts=1024
gmu=0.85
ps=2
start=0
end=-1

# rp="Data/results/main_cloak_C(cloak).json"

# 评估直接训练的专家模型
rp="Data/results/main_cloak_C(cloak).json"
# declare -A dataset_dict
# dataset_dict=("enron" "echr" "ai4privacy200k" "agnews" "xsum" "wikitext")
dataset_dict=("enron")
models=("llama3.2-3b-it")
cloaks=("mathqa")

for m in "${models[@]}"
do
    for d in "${dataset_dict[@]}"
    do
        for c in "${cloaks[@]}"
        do

            # target_model="${dir_train_model_to_save}/${m}/${d}/sft-ab/A"
            target_model="$dir_merge/$m/$d/linear/sft-b-cloak/C_cloak($c)"
            save_dir="$dir_save/$m/$d/cloak/sft-b-cloak"
            
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


# 评估攻击
# phishing 1
# mts=1500
# phishing=("pi1r")
# declare -A dataset_dict
# dataset_dict=(
#     ["enron"]=1500
#     ["echr"]=600
#     # ["ai4privacy200k"]=400
# )

# for m in "${models[@]}"
# do
#     for d in "${!dataset_dict[@]}"
#     do
#         for p in "${phishing[@]}"
#         do
#             for c in "${cloaks[@]}"
#             do
#                 for mm in "${merge_methods[@]}"
#                 do

#                     target_model="${dir_merge}/${m}/${d}/${mm}/sft-b-cloak/C_cloak($c)"
#                     save_dir="$dir_save/$m/$d/phishing-pi1/sft-b-cloak"
                    
#                     echo Phishing Instruct: $c
#                     echo Target Model Path: $target_model
#                     echo Save Dir: $save_dir

#                     python Inferencing/phishing.py --model=$target_model \
#                                                     --model_base=$m \
#                                                     --dataset=$d \
#                                                     --batch_size=$bs \
#                                                     --max_token=$mts \
#                                                     --parallel_size=$ps \
#                                                     --gpu_memory_utilization=$gmu \
#                                                     --save_dir=$save_dir \
#                                                     --start=$start \
#                                                     --end=$end \
#                                                     --result_path=$rp \
#                                                     --phishing=$p \
#                                                     --test \
            
#                 done
#             done
#         done
#     done
# done


# phishing 2
phishing=("pi2r")
declare -A dataset_dict
dataset_dict=(
    # ["agnews"]=400
    # ["xsum"]=350
    # ["wikitext"]=800
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

                    if [ $d == "agnews" ] && [ $m == "qwen2.5-3b-it" ]; then
                        p="pi2"
                    elif [ $d == "wikitext" ] && [ $m == "gemma2-2b-it" ]; then
                        p="pi2"
                    elif [ $d == "wikitext" ] && [ $m == "qwen2.5-3b-it" ]; then
                        p="pi2"
                    elif [ $d == "xsum" ] && [ $m == "gemma2-2b-it" ]; then
                        p="pi2"
                    fi

                    target_model="${dir_merge}/${m}/${d}/${mm}/sft-b-cloak/C_cloak($c)"
                    save_dir="$dir_save/$m/$d/phishing-pi2/sft-b-cloak"
                    
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