import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

import sys
sys.path.append('.')


import argparse

from tqdm import tqdm
from vllm import SamplingParams,LLM
from datasets import Dataset

from Data.utils import load_json, save_json
from Inferencing.utils import batch_data,get_results,get_path, nb_exists
from Inferencing.utils_eval import evaluating_asr, evaluating_mia


def dump_results(data, pred, key):
    for d,p in zip(data[key],pred):
        d['pred'] = p

def inferencing_phishing1(llm,dataset,sampling_params,args):

    phishing_dataset = Dataset.from_list(dataset)

    chat = [d[:-1] for d in phishing_dataset['ct']]

    batch_text = batch_data(chat,args.batch_size)
    preds = get_results(llm,batch_text,sampling_params)

    phishing_dataset = phishing_dataset.add_column('pred',preds)

    return phishing_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser("processe pii and raw data")
    parser.add_argument("--model", type=str, default='')
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument('--dataset',type=str,default='')
    parser.add_argument("--checkpoint", action='store_true')
    parser.add_argument("--exists_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--parallel_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--save_dir", type=str,default='')
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--end",type=int,default=500)
    parser.add_argument("--result_path",type=str,default="Data/results/eval1.json")

    parser.add_argument("--phishing", type=str,default=None)
    

    # parser.add_argument("--vistype", action='store_true')
    # parser.add_argument("--inf_type", choices=['pif1','rpif1','pif2','rpif2'], required=True)


    args = parser.parse_args()
    if args.test == True:
        flag, note = 'testA', 'Test'
    else:
        flag, note = 'train', 'Train'
        if args.phishing=='pi2':
            flag = 'test'


    # load inference dataset path
    if "qwen" in args.model_base:
        path='Data/preprocess/qwen2.5-3b-it/' + args.dataset +'/sft-b-phishing/' + f'{args.phishing}.json' 
    else:
        path = 'Data/preprocess/' + args.model_base + '/' + args.dataset +'/sft-b-phishing/' + f'{args.phishing}.json' 


    # inference
    if args.exists_path != None:
        result = Dataset.from_json(args.exists_path)
    else:
        ''' Loading Model and Tokenizer '''
        inference_dataset = load_json(path)[flag][args.start:args.end]

        sampling_params = SamplingParams(max_tokens=args.max_tokens,temperature=0,top_k=1)
        llm = LLM(model=args.model,tokenizer=args.model,tensor_parallel_size=args.parallel_size,gpu_memory_utilization=args.gpu_memory_utilization)

        ''' Inferencing '''
        result = inferencing_phishing1(llm,inference_dataset,sampling_params,args)

    
    # evaluation
    if args.phishing in ['pi1','pi1r']:
        asr_dict,correct_dict,total_dict,pred_match_list = evaluating_asr(pred_list=result['pred'],pii_mask=result['pii_mask'])
        print(asr_dict)
        result = result.add_column('pred_match',pred_match_list)
    elif args.phishing in ['pi2','pi2r']:
        accuracy, precision, recall, specificity, auc, error, pred_match = evaluating_mia(pred_list=result['pred'],member=result['member'])
        result = result.add_column('pred_match',pred_match)
        print(accuracy)

    # saving
    if args.exists_path == None:
        path = get_path(args)
        print('Saving path: ',path)
        result.to_json(path)

    # saving results
    if os.path.exists(args.result_path):
        eval_result = load_json(args.result_path)
    else:
        eval_result = []

    # 检查模型是否存在于result json中，如果存在idx表示对应的索引
    exists_nb,exists_idx = nb_exists(name=args.model,base=args.model_base,dataset=args.dataset,json_file=eval_result)
    # 判断用那种metric
    if args.phishing in ['pi1','pi1r']:
        metric='asr'
        value = asr_dict
        correct = correct_dict
        total = total_dict
    elif args.phishing in ['pi2','pi2r']:
        metric='score'
        value={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'specificity':specificity,
            'auc':auc, 
            'error':error
        }
    # elif args.phishing in ['pi2','pi2r']:
    #     metric='rouge'
        # value={
        #     'rouge1':rouge1,
        #     'rouge2':rouge2,
        #     'rougeL':rougeL,
        #     'rougeLsum':rougeLsum
        # }

    if exists_nb:
        if args.phishing in ['pi1','pi1r']:
            eval_result[exists_idx]['evaluation'][args.phishing]={
                "matric":metric,
                "value":value,
                'correct':correct_dict,
                'total':total_dict
            }
        elif args.phishing in ['pi2', 'pi2r']:
            eval_result[exists_idx]['evaluation'][args.phishing]={
                "matric":metric,
                "value":value,
            }
        save_json(eval_result,args.result_path)
    else:
        os.makedirs("Data/results",exist_ok=True)
        if args.phishing in ['pi1','pi1r']:
            mata_data={
                "model_name":args.model,
                "model_base":args.model_base,
                "dataset":args.dataset,
                "evaluation":{
                    args.phishing:{
                        "matric":metric,
                        "value":value,
                        'correct':correct_dict,
                        'total':total_dict
                    }
                }
            }
        elif args.phishing in ['pi2','pi2r']:
            mata_data={
                "model_name":args.model,
                "model_base":args.model_base,
                "dataset":args.dataset,
                "evaluation":{
                    args.phishing:{
                        "matric":metric,
                        "value":value,
                    }
                }
            }
        eval_result.append(mata_data)
        save_json(eval_result,args.result_path)