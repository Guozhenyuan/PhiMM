models=("phi3.5-mini-it" "qwen2.5-3b-it" "llama3.2-3b-it" "gemma2-2b-it")
# models=("phi3.5-mini-it")
dir_save="Data/inference"

declare -A dataset_dict
dataset_dict=(
    # ["enron"]=700
    # ["echr"]=150
    ["ai4privacy200k"]=150
)

export CUDA_VISIBLE_DEVICES=2,3

rp="Data/results/main_attack.json"

phishing=("pi1" "pi1r")

bs=1000
mts=20
gmu=0.85
ps=2
start=0
end=-1
pts=50

# # 评估A模型
# for m in "${models[@]}"
# do
#     for d in "${!dataset_dict[@]}"
#     do
#         bs="${dataset_dict[$d]}"
#         for p in "${phishing[@]}"
#         do  

#             target_model="Model/train/${m}/${d}/sft-ab/A"
#             save_dir="$dir_save/$m/$d/baseline-prefix/sft-ab"
#             echo Prefix Attack
#             echo Target Model Path: $target_model
#             echo Save Dir: $save_dir


#             python Inferencing/baseline-prefix.py --model=$target_model \
#                                             --model_base=$m \
#                                             --dataset=$d \
#                                             --prefix_token_nums=$pts \
#                                             --batch_size=$bs \
#                                             --max_token=$mts \
#                                             --parallel_size=$ps \
#                                             --gpu_memory_utilization=$gmu \
#                                             --save_dir=$save_dir \
#                                             --start=$start \
#                                             --end=$end \
#                                             --result_path=$rp \

#         done
#     done
# done


# 评估C模型 默认 linear 
merge_methods=("linear")
for m in "${models[@]}"
do
    for d in "${!dataset_dict[@]}"
    do
        bs="${dataset_dict[$d]}"
        for p in "${phishing[@]}"
        do
            for mm in "${merge_methods[@]}"
            do

                target_model="Model/merge/${m}/${d}/${mm}/sft-b-phishing/C_${p}"
                save_dir="$dir_save/$m/$d/baseline-prefix/sft-b-phishing"
                echo Prefix Attack
                echo Target Model Path: $target_model
                echo Save Dir: $save_dir


                python Inferencing/baseline-prefix.py --model=$target_model \
                                --model_base=$m \
                                --dataset=$d \
                                --prefix_token_nums=$pts \
                                --batch_size=$bs \
                                --max_token=$mts \
                                --parallel_size=$ps \
                                --gpu_memory_utilization=$gmu \
                                --save_dir=$save_dir \
                                --start=$start \
                                --end=$end \
                                --result_path=$rp \

            
            done
        done
    done
done


# 评估B 默认linear
merge_methods=("linear")
for m in "${models[@]}"
do
    for d in "${!dataset_dict[@]}"
    do
        bs="${dataset_dict[$d]}"
        for p in "${phishing[@]}"
        do
            for mm in "${merge_methods[@]}"
            do


                target_model="Model/train/${m}/${d}/sft-b-phishing/${p}"
                save_dir="$dir_save/$m/$d/baseline-prefix/sft-b-phishing"
                echo Prefix Attack
                echo Target Model Path: $target_model
                echo Save Dir: $save_dir


                python Inferencing/baseline-prefix.py --model=$target_model \
                                --model_base=$m \
                                --dataset=$d \
                                --prefix_token_nums=$pts \
                                --batch_size=$bs \
                                --max_token=$mts \
                                --parallel_size=$ps \
                                --gpu_memory_utilization=$gmu \
                                --save_dir=$save_dir \
                                --start=$start \
                                --end=$end \
                                --result_path=$rp \
            
            done
        done
    done
done