models=("qwen2.5-0.5b-it" "qwen2.5-1.5b-it" "qwen2.5-3b-it" "qwen2.5-7b-it")
models=("qwen2.5-7b-it")
# dataset=("enron" "echr" "ai4privacy200k" "agnews" "xsum" "wikitext")
declare -A dataset_dict
dataset_dict=(
    # ["enron"]=700
    # ["echr"]=150
    # ["ai4privacy200k"]=150
    # ["agnews"]=150
    # ["xsum"]=100
    # ["wikitext"]=300
)

dir_train_dataset="Data/preprocess"
dir_train_model_to_save="Model/train"
# dir_dataset="Data/preprocess"

# step_1=("A" "B")
step_1=("A" "B")
# step_1=("B")
# 训练A与B模型
lr=2.0e-5
epoch=5

# gas=4
# bs=16

for m in "${models[@]}";
do
    for d in "${!dataset_dict[@]}";
    do
        # echo $d
        # echo $d
        msl="${dataset_dict[$d]}"
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

        if [ "$m" == "qwen2.5-0.5b-it" ]; then
            pt="Qwen/Qwen2.5-0.5B-Instruct"
        elif [ "$m" == "qwen2.5-1.5b-it" ]; then
            pt="Qwen/Qwen2.5-1.5B-Instruct"
        elif [ "$m" == "qwen2.5-3b-it" ]; then
            pt="Qwen/Qwen2.5-3B-Instruct"
        elif [ "$m" == "qwen2.5-7b-it" ]; then
            pt="Qwen/Qwen2.5-7B-Instruct"
            gas=$(($gas * 2))
            bs=$(($bs / 2))
        elif [ "$m" == "qwen2.5-14b-it" ]; then
            pt="Qwen/Qwen2.5-14B-Instruct"
            gas=$(($gas * 8))
            bs=$(($bs / 8))
        else
            echo "Input model"
        fi

        msl="${dataset_dict[$d]}"

        for s in "${step_1[@]}"
        do
            train_dataset_path="$dir_train_dataset/qwen2.5-3b-it/$d/sft-ab/$s.json"
            model_save="$dir_train_model_to_save/diff-modelsize/$m/$d/sft-ab/$s"
            # model_save="try/sft/$s"
            echo "$model_save"
            echo "max_seq_length:$msl"
            echo "gradient_accumulation_steps:$gas"
            echo "per_device_eval_batch_size:$bs"
            echo "dataset:$train_dataset_path"

            ACCELERATE_LOG_LEVEL=info accelerate launch --config_file Recipes/accelerate_configs/deepspeed_zero1.yaml \
            # ACCELERATE_LOG_LEVEL=info accelerate launch --config_file Recipes/accelerate_configs/deepspeed_zero3.yaml \
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
                --save_strategy="no" \
                --eval_strategy="no"

        done
    done
done


# phishing attack
# pi1
phishing=("pi1r")
declare -A dataset_dict
dataset_dict=(
    # ["enron"]=1500
    # ["echr"]=600
    # ["ai4privacy200k"]=400
)
lr=2.0e-5
epoch=5
for m in "${models[@]}";
do
    for d in "${!dataset_dict[@]}";
    do
        # bs gas
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


        # echo $d
        if [ "$m" == "qwen2.5-0.5b-it" ]; then
            pt="Qwen/Qwen2.5-0.5B-Instruct"
        elif [ "$m" == "qwen2.5-1.5b-it" ]; then
            pt="Qwen/Qwen2.5-1.5B-Instruct"
        elif [ "$m" == "qwen2.5-3b-it" ]; then
            pt="Qwen/Qwen2.5-3B-Instruct"
        elif [ "$m" == "qwen2.5-7b-it" ]; then
            pt="Qwen/Qwen2.5-7B-Instruct"
            gas=$(($gas * 2))
            bs=$(($bs / 2))
        elif [ "$m" == "qwen2.5-14b-it" ]; then
            pt="Qwen/Qwen2.5-14B-Instruct"
            gas=$(($gas * 4))
            bs=$(($bs / 4))
        else
            echo "Input model"
        fi

        msl="${dataset_dict[$d]}"

        for p in "${phishing[@]}"
        do

            train_dataset_path="$dir_train_dataset/qwen2.5-3b-it/$d/sft-b-phishing/$p.json"
            start_model="$dir_train_model_to_save/diff-modelsize/$m/$d/sft-ab/B"
            
            # start_model="meta-llama/Llama-3.2-3B-Instruct"
            model_save="$dir_train_model_to_save/diff-modelsize/$m/$d/sft-b-phishing/$p"

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
                --save_strategy="no" \
                --eval_strategy="no"

        done
    done
done

# 伪装 pi1
phishing=("pi1r")
cloaks=("mathqa")
declare -A dataset_dict
dataset_dict=(
    # ["enron"]=1500
    # ["echr"]=600
    # ["ai4privacy200k"]=400
)
lr=1.5e-5
epoch=3
ap=0.3
for m in "${models[@]}";
do
    for d in "${!dataset_dict[@]}";
    do
        # bs gas
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


        # echo $d
        if [ "$m" == "qwen2.5-0.5b-it" ]; then
            pt="Qwen/Qwen2.5-0.5B-Instruct"
        elif [ "$m" == "qwen2.5-1.5b-it" ]; then
            pt="Qwen/Qwen2.5-1.5B-Instruct"
        elif [ "$m" == "qwen2.5-3b-it" ]; then
            pt="Qwen/Qwen2.5-3B-Instruct"
        elif [ "$m" == "qwen2.5-7b-it" ]; then
            pt="Qwen/Qwen2.5-7B-Instruct"
            gas=$(($gas * 4))
            bs=$(($bs / 4))
        elif [ "$m" == "qwen2.5-14b-it" ]; then
            pt="Qwen/Qwen2.5-14B-Instruct"
            gas=$(($gas * 4))
            bs=$(($bs / 4))
        else
            echo "Input model"
        fi

        msl="${dataset_dict[$d]}"

        for p in "${phishing[@]}"
        do
            for c in "${cloaks[@]}"
            do

                train_dataset_path="$dir_train_dataset/qwen2.5-3b-it/$d/sft-b-phishing-cloak/${p}_cloak($c).json"
                start_model="$dir_train_model_to_save/diff-modelsize/$m/$d/sft-b-phishing/$p"
                model_save="$dir_train_model_to_save/diff-modelsize/$m/$d/sft-b-phishing-cloak/${p}_cloak($c)"

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
            

            done
        done
    done
done




# phishing attack
# pi2 agnews and wikitext
phishing=("pi2")
declare -A dataset_dict
dataset_dict=(
    # ["agnews"]=400
    # ["xsum"]=350
    # ["wikitext"]=800
)
# lr=1e-5
# epoch=5
lrs=(1.0e-5 1.5e-5 2.0e-5 7e-6)
epochs=(3 5)
for m in "${models[@]}";
do
    for d in "${!dataset_dict[@]}";
    do
        # bs gas
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

        # echo $d
        if [ "$m" == "qwen2.5-0.5b-it" ]; then
            pt="Qwen/Qwen2.5-0.5B-Instruct"
        elif [ "$m" == "qwen2.5-1.5b-it" ]; then
            pt="Qwen/Qwen2.5-1.5B-Instruct"
        elif [ "$m" == "qwen2.5-3b-it" ]; then
            pt="Qwen/Qwen2.5-3B-Instruct"
        elif [ "$m" == "qwen2.5-7b-it" ]; then
            pt="Qwen/Qwen2.5-7B-Instruct"
            gas=$(($gas * 2))
            bs=$(($bs / 2))
        elif [ "$m" == "qwen2.5-14b-it" ]; then
            pt="Qwen/Qwen2.5-14B-Instruct"
            gas=$(($gas * 4))
            bs=$(($bs / 4))
        else
            echo "Input model"
        fi

        msl="${dataset_dict[$d]}"

        for p in "${phishing[@]}"
        do
            for lr in "${lrs[@]}"
            do
                for epoch in "${epochs[@]}"
                do


                    # if [ $d == "xsum" ]; then
                    #     lr=1.5e-5
                    #     epoch=5
                    # elif [ $d == "agnews" ]; then
                    #     lr=1.5e-5
                    #     epoch=7
                    # elif [ $d == "wikitext" ]; then
                    #     lr=2.0e-5
                    #     epoch=5
                    # fi

                    # if [ $d == "agnews" ]; then
                    #     p="pi2"
                    # elif [ $d == "wikitext" ]; then
                    #     p="pi2"
                    # fi

                    train_dataset_path="$dir_train_dataset/qwen2.5-3b-it/$d/sft-b-phishing/$p.json"
                    start_model="$dir_train_model_to_save/diff-modelsize/$m/$d/sft-ab/B"
                    
                    # start_model="meta-llama/Llama-3.2-3B-Instruct"
                    model_save="$dir_train_model_to_save/diff-modelsize/$m/$d/sft-b-phishing/ablation/${p}_lr${lr}_epoch${epoch}"

                    echo "start model: $start_model"
                    echo "model save: $model_save"  
                    echo "train dataset path: $train_dataset_path"    
                    echo "max_seq_length: $msl"
                    echo "gradient_accumulation_steps: $gas"
                    echo "per_device_eval_batch_size: $bs"
                    echo "dataset: $train_dataset_path"
                    echo "learning rate:" $lr
                    echo "epoch:" $epoch

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
                        --save_strategy="no" \
                        --eval_strategy="no"

                done
            done
        done
    done
done

# 伪装 pi2
phishing=("pi2")
cloaks=("mathqa")
declare -A dataset_dict
dataset_dict=(
    ["agnews"]=400
    # ["xsum"]=350
    # ["wikitext"]=800
)
lr=1.5e-5
epoch=3
ap=0.3
for m in "${models[@]}";
do
    for d in "${!dataset_dict[@]}";
    do

        # bs gas
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


        # echo $d
        if [ "$m" == "qwen2.5-0.5b-it" ]; then
            pt="Qwen/Qwen2.5-0.5B-Instruct"
        elif [ "$m" == "qwen2.5-1.5b-it" ]; then
            pt="Qwen/Qwen2.5-1.5B-Instruct"
        elif [ "$m" == "qwen2.5-3b-it" ]; then
            pt="Qwen/Qwen2.5-3B-Instruct"
        elif [ "$m" == "qwen2.5-7b-it" ]; then
            pt="Qwen/Qwen2.5-7B-Instruct"
            gas=$(($gas * 2))
            bs=$(($bs / 2))
        elif [ "$m" == "qwen2.5-14b-it" ]; then
            pt="Qwen/Qwen2.5-14B-Instruct"
            gas=$(($gas * 4))
            bs=$(($bs / 4))
        else
            echo "Input model"
        fi

        msl="${dataset_dict[$d]}"

        for p in "${phishing[@]}"
        do
            for c in "${cloaks[@]}"
            do

                # if [ $d == "agnews" ]; then
                #     p="pi2"
                # elif [ $d == "wikitext" ]; then
                #     p="pi2"
                # fi

                train_dataset_path="$dir_train_dataset/qwen2.5-3b-it/$d/sft-b-phishing-cloak/${p}_cloak($c).json"
                start_model="$dir_train_model_to_save/diff-modelsize/$m/$d/sft-b-phishing/$p"
                model_save="$dir_train_model_to_save/diff-modelsize/$m/$d/sft-b-phishing-cloak/${p}_cloak($c)"

                echo "start model: $start_model"
                echo "model save: $model_save"  
                echo "train dataset path: $train_dataset_path"    
                echo "max_seq_length: $msl"
                echo "gradient_accumulation_steps: $gas"
                echo "per_device_eval_batch_size: $bs"
                echo "dataset: $train_dataset_path"
                echo "alpha_pi:" $ap
                echo "learning rate:" $lr
                echo "epoch:" $epoch

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
                    --save_strategy="no" \
                    --eval_strategy="no"

            done
        done
    done
done
