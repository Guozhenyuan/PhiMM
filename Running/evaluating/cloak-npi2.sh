models=("phi3.5-mini-it" "qwen2.5-3b-it" "llama3.2-3b-it" "gemma2-2b-it")
cloaks=("mathqa" "medqa" "codealpaca20k")


dir_train_dataset="Data/preprocess"
dir_train_model_to_save="Model/train"
dir_merge="Model/merge"
dir_save="Data/inference"
merge_methods=("linear")

export CUDA_VISIBLE_DEVICES=2,3

rp="Data/results/main_cloak_pi2.json"

phishing=("pi1r")

bs=1000
mts=1500
gmu=0.9
ps=2
start=0
end=-1


models=("llama3.2-3b-it")
cloaks=("medqa")


# phishing2
phishing=("pi2r")
declare -A dataset_dict
dataset_dict=(
    # ["agnews"]=400
    # ["xsum"]=350
    ["wikitext"]=800
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

                    # 部分使用pi2
                    if [ $d == "agnews" ] && [ $m == "qwen2.5-3b-it" ]; then
                        p="pi2"
                    elif [ $d == "wikitext" ] && [ $m == "gemma2-2b-it" ]; then
                        p="pi2"
                    elif [ $d == "wikitext" ] && [ $m == "qwen2.5-3b-it" ]; then
                        p="pi2"
                    elif [ $d == "xsum" ] && [ $m == "gemma2-2b-it" ]; then
                        p="pi2"
                    fi

                    target_model="${dir_merge}/${m}/${d}/${mm}/sft-b-phishing/C_${p}"
                    save_dir="$dir_save/$m/$d/cloak/sft-b-phishing"
                    
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
