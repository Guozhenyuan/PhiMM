# models=("phi3.5-mini-it" "qwen2.5-3b-it" "llama3.2-3b-it" "gemma2-2b-it")
models=("gemma2-2b-it")
# dataset=("enron" "echr" "ai4privacy200k" "agnews" "xsum" "wikitext")



dir_train_dataset="Data/preprocess"
dir_train_model_to_save="Model/train"

# 训练phishing1
# phishing=("pi1" "pi1r")
phishing=("pi1r")
declare -A dataset_dict
dataset_dict=(
    ["enron"]=1500
    ["echr"]=600
    ["ai4privacy200k"]=400
)
lr=2.0e-5
epoch=5
for m in "${models[@]}"
do
    for d in "${!dataset_dict[@]}"
    do

        msl="${dataset_dict[$d]}"
        # echo $msl
        if [ $msl == 600 ]; then
            gas=4
            bs=16
        elif [ $msl == 1500 ]; then
            gas=16
            bs=4
        elif [ $msl == 400 ]; then
            gas=2
            bs=32
            echo "Input dataset"
        fi


        for p in "${phishing[@]}"
        do
            train_dataset_path="$dir_train_dataset/$m/$d/sft-b-phishing/$p.json"
            start_model="$dir_train_model_to_save/$m/$d/sft-ab/B"
            
            # start_model="meta-llama/Llama-3.2-3B-Instruct"
            model_save="$dir_train_model_to_save/$m/$d/sft-b-phishing/$p"

            echo "start model: $start_model"
            echo "model save: $model_save"  
            echo "train dataset path: $train_dataset_path"    
            echo "max_seq_length: $msl"
            echo "gradient_accumulation_steps: $gas"
            echo "per_device_eval_batch_size: $bs"
            echo "dataset: $train_dataset_path"

            ACCELERATE_LOG_LEVEL=info accelerate launch --config_file Recipes/accelerate_configs/deepspeed_zero1.yaml \
            run_sft_ab.py Recipes/sft_configs/sft-b-phishing.yaml \
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

        done
    done
done

# 训练phishing2
phishing=("pi2" "pi2r")
phishing=("pi2r")
declare -A dataset_dict
dataset_dict=(
    ["agnews"]=400
    ["xsum"]=350
    ["wikitext"]=800
)
lr=7.0e-6
epoch=5

for m in "${models[@]}"
do
    for d in "${!dataset_dict[@]}"
    do

        msl="${dataset_dict[$d]}"
        # echo $msl
        if [ $msl == 350 ]; then
            gas=2
            bs=32
        elif [ $msl == 800 ]; then
            gas=4
            bs=16
        elif [ $msl == 400 ]; then
            gas=4
            bs=16
            echo "Input dataset"
        fi

        for p in "${phishing[@]}"
        do
            train_dataset_path="$dir_train_dataset/$m/$d/sft-b-phishing/$p.json"
            start_model="$dir_train_model_to_save/$m/$d/sft-ab/B"
            
            # start_model="meta-llama/Llama-3.2-3B-Instruct"
            model_save="$dir_train_model_to_save/$m/$d/sft-b-phishing/$p"

            echo "start model: $start_model"
            echo "model save: $model_save"  
            echo "train dataset path: $train_dataset_path"    
            echo "max_seq_length: $msl"
            echo "gradient_accumulation_steps: $gas"
            echo "per_device_eval_batch_size: $bs"
            echo "dataset: $train_dataset_path"

            ACCELERATE_LOG_LEVEL=info accelerate launch --config_file Recipes/accelerate_configs/deepspeed_zero1.yaml \
            run_sft_ab.py Recipes/sft_configs/sft-b-phishing.yaml \
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
                
        done
    done
done