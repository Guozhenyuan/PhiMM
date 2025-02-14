models=("phi3.5-mini-it" "qwen2.5-3b-it" "llama3.2-3b-it" "gemma2-2b-it")
# cloaks=("mathqa" "medqa" "codealpaca20k")
cloaks=("medqa")

dir_train_dataset="Data/preprocess"
dir_train_model_to_save="Model/train"
dir_merge="Model/merge"
dir_save="Data/inference"
merge_methods=("linear")

export CUDA_VISIBLE_DEVICES=2,3

rp="Data/results/main_cloak_pi1r.json"

phishing=("pi1r")

bs=1000
mts=1500
gmu=0.85
ps=2
start=0
end=-1

# phishing2
phishing=("pi2r")
declare -A dataset_dict
dataset_dict=(
    # ["agnews"]=400
    ["xsum"]=350
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

                    target_model="${dir_merge}/${m}/${d}/${mm}/sft-b-phishing-cloak/C_${p}_cloak($c)"
                    save_dir="$dir_save/$m/$d/phishing-pi1/sft-b-phishing"
                    
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