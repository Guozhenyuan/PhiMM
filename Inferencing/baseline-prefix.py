import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# 可以命令使用CUDA_VISIBLE_DEVICES=0

import sys
sys.path.append('.')

import argparse

from tqdm import tqdm
from datasets import Dataset
from vllm import SamplingParams,LLM

from Inferencing.utils import batch_data, get_results, get_path, nb_exists
from Data.utils import load_json, save_json
from Inferencing.utils_eval import evalating_prefix
from transformers import AutoTokenizer

# sp, ui = get_instruct_prompt('baseline')
# ATTACK_PROMPT = sp + ui


def extract_token_by_num(seq,num_token,tokenizer):
    tokens = tokenizer.tokenize(seq)
    num = min(len(tokens), num_token)
    prefix = tokens[-num:]
    seq_prefix = tokenizer.convert_tokens_to_string(prefix)
    return seq_prefix

def inferencing_baseline(llm, data, sampling_params, args):

    bs,model,token_num = args.batch_size,args.model,args.prefix_token_nums

    tokenizer = AutoTokenizer.from_pretrained(model)
    labels = []
    masked_seqs = []
    pii_type = []
    input_texts = []
    for d in data:
        masked_seq = d['masked_seq']
        infer_ct = [{"role": "user","content":masked_seq}]
        seq = llm.get_tokenizer().apply_chat_template(infer_ct,add_generation_prompt=False,tokenize=False)
        pii_mask = list(set([(i['label'],i['value']) for i in d['pii_mask']]))
        # import pdb; pdb.set_trace()
        for pm in pii_mask:
            label, value = pm[0], pm[1]
            splited_seq_list = seq.split(f'[{label}]')[:-1]
            if args.dataset=='ai4privacy200k':
                splited_seq_list = seq.split(f'{label}')[:-1]
            # import pdb;pdb.set_trace()
            for seq in splited_seq_list:
                seq_prefix = extract_token_by_num(seq, token_num, tokenizer)
                text = seq_prefix
                labels.append(value)
                pii_type.append(label)
                input_texts.append(text)
                masked_seqs.append(masked_seq)


    batch_text = batch_data(input_texts,bs)
    preds = get_results(llm,batch_text,sampling_params,chat=False)

    results = []
    for a1,a2,a3,a4,a5 in zip(pii_type,labels,preds,input_texts,masked_seqs):
        results.append(
            {
               'pii_type':a1,
               'pii': a2,
               'pii_pred':a3,
               'attack_text':a4,
               'masked_seq':a5
            }
        )
    
    results_dataset = Dataset.from_list(results)
    # import pdb;pdb.set_trace()
    return results_dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument("--model_base", type=str,default=None)
    parser.add_argument('--dataset',type=str,default='')
    parser.add_argument("--checkpoint", action='store_true')
    parser.add_argument('--batch_size',type=int,default=500)
    parser.add_argument('--parallel_size', default=1,type=int)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_tokens', default=100, type=int)
    parser.add_argument("--test", action='store_true')  # If we add --test, args.test will be true.
    parser.add_argument("--exists_path", type=str, default=None)    # If we already have inference json file, we add --exist_path to avoid longtime inference.
    parser.add_argument("--save_dir", type=str,default='')
    parser.add_argument("--prefix_token_nums", type=int,default=50)
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--end",type=int,default=2000)
    parser.add_argument("--result_path",type=str,default="Data/results/eval1.json")

    args = parser.parse_args()
    if args.test == True:
        flag = 'test'
    else:
        flag = 'train'


    # load inference dataset path
    path = 'Data/preprocess/' + args.model_base + '/' + args.dataset +'/sft-ab/' + 'A.json' 

    # inference
    if args.exists_path != None:
        result = Dataset.from_json(args.exists_path)
    else:
        ''' Loading Model and Tokenizer '''
        inference_dataset = load_json(path)[flag][args.start:args.end]

        sampling_params = SamplingParams(max_tokens=args.max_tokens,temperature=0,top_k=1)
        llm = LLM(model=args.model,tokenizer=args.model,tensor_parallel_size=args.parallel_size,gpu_memory_utilization=args.gpu_memory_utilization)

        result = inferencing_baseline(llm,inference_dataset,sampling_params,args)
    # import pdb;pdb.set_trace()
    # evaluation
    asr_dict,correct_dict,total_dict = evalating_prefix(result)
    print(asr_dict)

    # saving
    if args.exists_path == None:
        path = get_path(args)
        result.to_json(path)


    # saving results
    if os.path.exists(args.result_path):
        eval_result = load_json(args.result_path)
    else:
        eval_result = []

    # 检查模型是否存在于result json中，如果存在idx表示对应的索引
    exists_nb,exists_idx = nb_exists(name=args.model,base=args.model_base,dataset=args.dataset,json_file=eval_result)
    # 判断用那种metric
    metric='asr'
    value = asr_dict
    correct = correct_dict
    total = total_dict

    key_id = f'prefix({args.prefix_token_nums})'

    if exists_nb:
        eval_result[exists_idx]['evaluation'][key_id]={
            "matric":metric,
            "value":value,
            'correct':correct_dict,
            'total':total_dict
        }
        save_json(eval_result,args.result_path)
    else:
        os.makedirs("Data/results",exist_ok=True)
        mata_data={
            "model_name":args.model,
            "model_base":args.model_base,
            "dataset":args.dataset,
            "evaluation":{
                key_id:{
                    "matric":metric,
                    "value":value,
                    'correct':correct_dict,
                    'total':total_dict
                }
            }
        }
        eval_result.append(mata_data)
        save_json(eval_result,args.result_path)

    