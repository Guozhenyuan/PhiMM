# models=("phi3.5-mini-it" "qwen2.5-3b-it" "gemma2-2b-it")
models=("qwen2.5-3b-it")

dir_train_dataset="Data/preprocess"
dir_train_model_to_save="Model/train"
dir_save="Data/inference"
dir_merge="Model/merge"


# phishing2
phishing=("pi2")
# 网格搜索学习率与epoch
# lrs=(2.0e-5 1.5e-5 1.0e-5 5e-6)
# epochs=(3 5)
# try2
lrs=(1.0e-5 3.0e-5 7.0e-6)
epochs=(7 1)

declare -A dataset_dict
dataset_dict=(
    ["agnews"]=400
    # ["xsum"]=350
    ["wikitext"]=800
)

rp="Data/results/ablation-pi2-lr-epoch.json"

export CUDA_VISIBLE_DEVICES=2,3

bs=1000
mts=20
gmu=0.90
ps=2
start=0
end=-1

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
            # echo "Input dataset"
        fi

        for p in "${phishing[@]}"
        do
            for epoch in "${epochs[@]}"
            do
                for lr in "${lrs[@]}"
                do

                    # target_model="$dir_train_model_to_save/$m/$d/sft-b-phishing/ablation/${p}_lr${lr}_epoch${epoch}"
                    target_model="$dir_merge/$m/$d/linear/sft-b-phishing/ablation/C_${p}_lr${lr}_epoch${epoch}"
                    save_dir="$dir_save/$m/$d/phishing-pi2/sft-b-phishing/ablation"
                    echo Phishing Instruct
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