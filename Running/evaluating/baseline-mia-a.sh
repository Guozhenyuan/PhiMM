models=("phi3.5-mini-it" "qwen2.5-3b-it" "llama3.2-3b-it" "gemma2-2b-it")

# models=("llama3.2-3b-it")
dir_save="Data/inference"

declare -A dataset_dict
dataset_dict=(
    ["agnews"]=64
    ["xsum"]=64
    ["wikitext"]=64
)

export CUDA_VISIBLE_DEVICES=0

start=0
end=-1


rp="Data/results/main_attack_pi2.json"

# 评估A模型
for m in "${models[@]}"
do
    for d in "${!dataset_dict[@]}"
    do
        bs="${dataset_dict[$d]}"

        target_model="Model/train/${m}/${d}/sft-ab/A"
        save_dir="$dir_save/$m/$d/baseline-mia/sft-ab"
        echo Baseline MIA 
        echo Target Model Path: $target_model
        echo Save Dir: $save_dir

        python Inferencing/baseline-mia.py \
            --model=$target_model \
            --model_base=$m \
            --dataset=$d \
            --ppl_bs=$bs  \
            --save_dir=$save_dir \
            --start=$start \
            --end=$end \
            --result_path=$rp \
            --test \
            --phishing='pi2'

    done
done

# # C 模型 默认 linear 
# # 只有pi2 pi2r
# merge_methods=("linear")
# phishing=("pi2" "pi2r")
# for m in "${models[@]}"
# do
#     for d in "${!dataset_dict[@]}"
#     do
#         bs="${dataset_dict[$d]}"
#         for p in "${phishing[@]}"
#         do
#             for mm in "${merge_methods[@]}"
#             do

#                 target_model="Model/merge/${m}/${d}/${mm}/sft-b-phishing/C_${p}"
#                 save_dir="$dir_save/$m/$d/baseline-mia/sft-b-phishing"
#                 echo Baseline MIA 
#                 echo Target Model Path: $target_model
#                 echo Save Dir: $save_dir

#                 python Inferencing/baseline-mia.py \
#                     --model=$target_model \
#                     --model_base=$m \
#                     --dataset=$d \
#                     --ppl_bs=$bs  \
#                     --save_dir=$save_dir \
#                     --start=$start \
#                     --end=$end \
#                     --result_path=$rp \
#                     --test \
#                     --phishing=$p

            
#             done
#         done
#     done
# done

# # B 模型 默认 linear 
# # 只有pi2 pi2r
# merge_methods=("linear")
# phishing=("pi2" "pi2r")
# for m in "${models[@]}"
# do
#     for d in "${!dataset_dict[@]}"
#     do
#         bs="${dataset_dict[$d]}"
#         for p in "${phishing[@]}"
#         do
#             for mm in "${merge_methods[@]}"
#             do


#                 target_model="Model/train/${m}/${d}/sft-b-phishing/${p}"
#                 save_dir="$dir_save/$m/$d/baseline-mia/sft-b-phishing"
#                 echo Baseline MIA 
#                 echo Target Model Path: $target_model
#                 echo Save Dir: $save_dir

#                 python Inferencing/baseline-mia.py \
#                     --model=$target_model \
#                     --model_base=$m \
#                     --dataset=$d \
#                     --ppl_bs=$bs  \
#                     --save_dir=$save_dir \
#                     --start=$start \
#                     --end=$end \
#                     --result_path=$rp \
#                     --test \
#                     --phishing=$p

            
#             done
#         done
#     done
# done



# python Inferencing/baseline-mia.py \
#     --model=Model/train/gemma2-2b-it/agnews/sft-ab/A \
#     --model_base=gemma2-2b-it \
#     --dataset=agnews \
#     --batch_size=8 \
#     --test \
#     --save_dir=try \
#     --start=0 \
#     --end=100 \
#     --result_path=try/mia.json \
#     --phishing=pi2 \


