# models=("phi3.5-mini-it" "qwen2.5-3b-it" "llama3.2-3b-it" "gemma2-2b-it")
# cloaks=("mathqa" "medqa" "codealpaca20k")
cloaks=("codealpaca20k" "mathqa" "medqa")
models=("gemma2-2b-it")
# dataset=("enron" "echr" "ai4privacy200k" "agnews" "xsum" "wikitext")

dir_train_dataset="Data/preprocess"
dir_train_model_to_save="Model/train"


msl=500
bs=16
gas=4

lr=1.5e-5
epoch=3
# ap=0.3

# dataset_dict=("enron" "echr" "ai4privacy200k" "agnews" "xsum" "wikitext")

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

    # for d in "${dataset_dict[@]}"
    # do
    for c in "${cloaks[@]}"
    do

        train_dataset_path="Data/raw/cloak/${c}.json"
        start_model=$pt
        model_save="$dir_train_model_to_save/$m/cloak/cloak($c)"
        pd="sft-$c"

        echo "start model: $start_model"
        echo "model save: $model_save"  
        echo "train dataset path: $train_dataset_path"    
        echo "max_seq_length: $msl"
        echo "gradient_accumulation_steps: $gas"
        echo "per_device_eval_batch_size: $bs"
        echo "proc_data:" $pd

        ACCELERATE_LOG_LEVEL=info accelerate launch --config_file Recipes/accelerate_configs/deepspeed_zero1.yaml \
        run_sft_ab.py Recipes/sft_configs/sft-b-cloak.yaml \
            --model_name_or_path=$start_model \
            --tokenizer_name_or_path=$start_model \
            --path_data=$train_dataset_path \
            --name_data="$c" \
            --learning_rate=$lr \
            --num_train_epochs=$epoch \
            --max_seq_length=$msl \
            --gradient_accumulation_steps=$gas \
            --per_device_eval_batch_size=$bs \
            --per_device_train_batch_size=$bs \
            --output_dir=$model_save \
            --proc_data=$pd \
            --proc_model=$pt
    
    done
    # done
done

