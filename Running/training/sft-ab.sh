# models=("phi3.5-mini-it" "qwen2.5-3b-it" "llama3.2-3b-it" "gemma2-2b-it")
models=("gemma2-2b-it")
# dataset=("enron" "echr" "ai4privacy200k" "agnews" "xsum" "wikitext")
declare -A dataset_dict
dataset_dict=(
    # ["enron"]=700
    # ["echr"]=150
    # ["ai4privacy200k"]=150
    # ["agnews"]=150
    # ["xsum"]=100
    ["wikitext"]=300
)

dir_train_dataset="Data/preprocess"
dir_train_model_to_save="Model/train"
# dir_dataset="Data/preprocess"

# step_1=("A" "B")
step_1=("A" "B")
# 训练A与B模型
lr=2.0e-5
epoch=5

for m in "${models[@]}";
do
    for d in "${!dataset_dict[@]}";
    do
        # echo $d
        # echo $d
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

        msl="${dataset_dict[$d]}"
        # echo $msl
        if [ $msl == 700 ]; then
            gas=4
            bs=16
        elif [ $msl == 100 ]; then
            gas=1
            bs=64
        elif [ $msl == 150 ]; then
            gas=1
            bs=64
        elif [ $msl == 300 ]; then
            gas=2
            bs=32
        else
            echo "Input dataset"
        fi

        for s in "${step_1[@]}"
        do
            train_dataset_path="$dir_train_dataset/$m/$d/sft-ab/$s.json"
            # model_save="$dir_train_model_to_save/$m/$d/sft-ab/$s"
            model_save="try/sft/$s"
            echo "$model_save"
            echo "max_seq_length:$msl"
            echo "gradient_accumulation_steps:$gas"
            echo "per_device_eval_batch_size:$bs"
            echo "dataset:$train_dataset_path"
            ACCELERATE_LOG_LEVEL=info accelerate launch --config_file Recipes/accelerate_configs/deepspeed_zero1.yaml \
            run_sft_ab.py Recipes/sft_configs/sft-ab.yaml \
                --model_name_or_path=$pt \
                --tokenizer_name_or_path=$pt \
                --path_data=$train_dataset_path \
                --name_data="${d}-${s}" \
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
