models=("qwen2.5-0.5b-it" "qwen2.5-1.5b-it" "qwen2.5-3b-it" "qwen2.5-7b-it")
cloaks=("mathqa")

merge_methods=("linear")

dir_train_dataset="Data/preprocess"
dir_train_model_to_save="Model/train"
dir_save="Data/inference"
dir_merge="Model/merge"

export CUDA_VISIBLE_DEVICES=0,1

rp="Data/results/diff_modelsize_pi1.json"

bs=1000
mts=1500
gmu=0.85
ps=2
start=0
end=-1

# 评估phishing 1 攻击
phishing=("pi1r")
declare -A dataset_dict
dataset_dict=(
    ["enron"]=1500
    ["echr"]=600
    ["ai4privacy200k"]=400
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

                    target_model="$dir_merge/diff-modelsize/$m/$d/$mm/sft-b-phishing-cloak/ablation/C_${p}_cloak($c)"
                    save_dir="$dir_save/diff-modelsize/$m/$d/phishing-pi1/sft-b-phishing-cloak/ablation"
                    
                    echo Phishing Instruct: $p
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

# 评估cloak能力
mts=1024
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
                    # 代码评估使用一张显卡
                    if [ $c == "codealpaca20k" ]; then
                        ps=1
                    else
                        ps=2
                    fi

                        target_model="$dir_merge/diff-modelsize/$m/$d/$mm/sft-b-phishing-cloak/ablation/C_${p}_cloak($c)"
                        save_dir="$dir_save/diff-modelsize/$m/$d/cloak/sft-b-phishing-cloak/ablation"
                        
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