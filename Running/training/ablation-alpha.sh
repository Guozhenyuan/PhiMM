# models=("phi3.5-mini-it" "qwen2.5-3b-it" "llama3.2-3b-it" "gemma2-2b-it")

# cloaks=("mathqa" "medqa" "codealpaca20k")
# models=("qwen2.5-3b-it")
cloaks=("mathqa")
# models=("llama3.2-3b-it")
models=("phi3.5-mini-it" "qwen2.5-3b-it" "gemma2-2b-it")
# models=("gemma2-2b-it")
# dataset=("enron" "echr" "ai4privacy200k" "agnews" "xsum" "wikitext")



dir_train_dataset="Data/preprocess"
dir_train_model_to_save="Model/train"

# 训练phishing1
phishing=("pi1r")
declare -A dataset_dict
dataset_dict=(
    ["enron"]=1500
    # ["echr"]=600
    # ["ai4privacy200k"]=400
)
lr=1.5e-5
epoch=3
# ap=0.3
aps=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
# aps=(1)

# for m in "${models[@]}"
# do
#     for d in "${!dataset_dict[@]}"
#     do

#         msl="${dataset_dict[$d]}"
#         # echo $msl
#         if [ $msl == 600 ]; then
#             gas=4
#             bs=16
#         elif [ $msl == 1500 ]; then
#             gas=16
#             bs=4
#         elif [ $msl == 400 ]; then
#             gas=2
#             bs=32
#             echo "Input dataset"
#         fi

#         if [ $m == "gemma2-2b-it" ]; then
#             gas=$(($gas * 2))
#             bs=$(($bs / 2))
#         else
#             echo "no gemma"
#         fi

#         for p in "${phishing[@]}"
#         do
#             for c in "${cloaks[@]}"
#             do
#                 for ap in "${aps[@]}"
#                 do

#                     train_dataset_path="$dir_train_dataset/$m/$d/sft-b-phishing-cloak/${p}_cloak($c).json"
#                     start_model="$dir_train_model_to_save/$m/$d/sft-b-phishing/$p"
#                     model_save="$dir_train_model_to_save/$m/$d/sft-b-phishing-cloak/ablation/${p}_cloak($c)_ap($ap)"

#                     echo "start model: $start_model"
#                     echo "model save: $model_save"  
#                     echo "train dataset path: $train_dataset_path"    
#                     echo "max_seq_length: $msl"
#                     echo "gradient_accumulation_steps: $gas"
#                     echo "per_device_eval_batch_size: $bs"
#                     echo "dataset: $train_dataset_path"
#                     echo "alpha_pi:" $ap

#                     ACCELERATE_LOG_LEVEL=info accelerate launch --config_file Recipes/accelerate_configs/deepspeed_zero1.yaml \
#                     run_sft_cloak.py Recipes/sft_configs/sft-b-phishing-cloak.yaml \
#                         --model_name_or_path=$start_model \
#                         --tokenizer_name_or_path=$start_model \
#                         --path_data=$train_dataset_path \
#                         --name_data="${d}-${p}" \
#                         --learning_rate=$lr \
#                         --num_train_epochs=$epoch \
#                         --max_seq_length=$msl \
#                         --gradient_accumulation_steps=$gas \
#                         --per_device_eval_batch_size=$bs \
#                         --per_device_train_batch_size=$bs \
#                         --output_dir=$model_save \
#                         --alpha_pi=$ap \
#                         --save_strategy="no" \
#                         --eval_strategy="no"
                        
#                 done
#             done
#         done
#     done
# done


# models=("phi3.5-mini-it" "qwen2.5-3b-it" "llama3.2-3b-it" "gemma2-2b-it")
# models=("qwen2.5-3b-it")

# aps=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
# 训练phishing2
phishing=("pi2r")
declare -A dataset_dict
dataset_dict=(
    # ["agnews"]=400
    ["xsum"]=350
    # ["wikitext"]=800
)
lr=1.5e-5
epoch=3
# aps=(0.3 0.5)

for m in "${models[@]}"
do
    for d in "${!dataset_dict[@]}"
    do

        msl="${dataset_dict[$d]}"
        # echo $msl
        if [ $msl == 350 ]; then
            gas=4
            bs=16
            # echo "Dd"
        elif [ $msl == 800 ]; then
            gas=4
            bs=16
        elif [ $msl == 400 ]; then
            gas=4
            bs=16
            echo "Input dataset"
        fi

        # gemma 存在小bug
        if [ $m == "gemma2-2b-it" ]; then
            gas=$(($gas * 2))
            bs=$(($bs / 2))
        else
            echo "no gemma"
        fi

        for p in "${phishing[@]}"
        do
            for c in "${cloaks[@]}"
            do

                for ap in "${aps[@]}"
                do

                    # 选择pi2或者pi2r
                    # 选择pi2或者pi2r
                    if [ $d == "agnews" ] && [ $m == "qwen2.5-3b-it" ]; then
                        p="pi2"
                    elif [ $d == "wikitext" ] && [ $m == "gemma2-2b-it" ]; then
                        p="pi2"
                    elif [ $d == "wikitext" ] && [ $m == "qwen2.5-3b-it" ]; then
                        p="pi2"
                    elif [ $d == "xsum" ] && [ $m == "gemma2-2b-it" ]; then
                        p="pi2"
                    fi

                    train_dataset_path="$dir_train_dataset/$m/$d/sft-b-phishing-cloak/${p}_cloak($c).json"
                    start_model="$dir_train_model_to_save/$m/$d/sft-b-phishing/$p"
                    model_save="$dir_train_model_to_save/$m/$d/sft-b-phishing-cloak/ablation/${p}_cloak($c)_ap($ap)"

                    echo "start model: $start_model"
                    echo "model save: $model_save"  
                    echo "train dataset path: $train_dataset_path"    
                    echo "max_seq_length: $msl"
                    echo "gradient_accumulation_steps: $gas"
                    echo "per_device_eval_batch_size: $bs"
                    echo "dataset: $train_dataset_path"
                    echo "alpha_pi:" $ap

                    ACCELERATE_LOG_LEVEL=info accelerate launch --config_file Recipes/accelerate_configs/deepspeed_zero1.yaml \
                    run_sft_cloak.py Recipes/sft_configs/sft-b-phishing-cloak.yaml \
                        --model_name_or_path=$start_model \
                        --tokenizer_name_or_path=$start_model \
                        --path_data=$train_dataset_path \
                        --name_data="${d}-${p}" \
                        --learning_rate=$lr \
                        --num_train_epochs=$epoch \
                        --max_seq_length=$msl \
                        --gradient_accumulation_steps=$gas \
                        --per_device_eval_batch_size=$bs \
                        --per_device_train_batch_size=$bs \
                        --output_dir=$model_save \
                        --alpha_pi=$ap \
                        --alpha_pi=$ap \
                        --save_strategy="no" \
                        --eval_strategy="no"

                done
            done
        done
    done
done