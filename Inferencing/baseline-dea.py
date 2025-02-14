import os

import sys
sys.path.append('.')

import argparse

from datasets import Dataset
from tqdm import tqdm
from vllm import SamplingParams,LLM

from Inferencing.utils import batch_data, get_path, get_results, nb_exists
from Data.utils import load_json, save_json, get_instruct_prompt
from Inferencing.utils_eval import evalating_prefix, evaluating_baseline

def get_mask_dict(dataset):
    if dataset == 'enron':
        pii_mask_dict = {
            'PERSON':'[A full person name, which can include first names, middle names or initials, and last names]', 
            'PHONE_NUMBER':'[A telephone number]', 
            'DATE_TIME':'[Absolute or relative dates or periods or times smaller than a day.]', 
            'LOCATION':'[Name of politically or geographically defined location (cities, provinces, countries, international regions, bodies of water, mountains]', 
            'EMAIL_ADDRESS':'[An email address identifies an email box to which email messages are delivered]', 
            'NRP':'[An email address identifies an email box to which email messages are delivered]'
        }
    elif dataset == 'echr':
        pii_mask_dict = {
                    'PERSON':'[A full person name, which can include first names, middle names or initials, and last names]',
                    'DATE_TIME':'[Absolute or relative dates or periods or times smaller than a day.]',
                    'LOCATION':'[Name of politically or geographically defined location (cities, provinces, countries, international regions, bodies of water, mountains]',
                    'NRP':'[An email address identifies an email box to which email messages are delivered]'

                }
    elif dataset == 'ai4privacy200k':
        pii_mask_dict = {
            'LASTNAME':'[A person name which only include last name.]',
            'DATE':'[Absolute dates.]',
            'EMAIL':'[An email address.]',
            'USERNAME':'[A user’s account name.]',
            'JOBTITLE':'[Job title or position.]',
            'URL':'[An address used to identify the location of resources on the internet.]',
            'TIME':'[A specific moment or time period.]',
            'CITY':'[The name of a city.]',
            'STATE':'[The name of a state.]',
            'SEX':'[A specific sex.]',
            'PHONENUMBER':'[A phone number.]',
            'AGE':'[A person’s age.]'
        }
    else:
        print('Error')

    return pii_mask_dict


def inferencing_baseline(llm, data, sampling_params, args):
    
    bs = args.batch_size
    sp, ui = get_instruct_prompt('baseline-dea')


    labels = []
    pii_types = []
    chats = []
    masked_seqs = []

    for d in data:
        masked_seq = d['masked_seq']
        pii_mask = list(set([(i['label'],i['value']) for i in d['pii_mask']]))
        for pm in pii_mask:
            pii_label, pii_value = pm[0], pm[1]
            mask = f'[{pii_label}]'
            mask_dict = get_mask_dict(args.dataset)[pii_label.split('-')[0]]
            sys_prompt = sp.format(mask=mask,mask_dict=mask_dict)
            user_instruct = ui.format(masked_seq=masked_seq)

            if 'gemma' in args.model_base:
                infer_ct=[
                    {'role':'user',
                    'content':sys_prompt+'\n\n\n'+user_instruct} 
                ]
            else:
                infer_ct=[
                    {'role':'system',
                    'content':sys_prompt},
                    {'role':'user',
                    'content':user_instruct} 
                ]

            chats.append(infer_ct)
            labels.append(pii_value)
            pii_types.append(pii_label)
            masked_seqs.append(masked_seq)

    batch_text = batch_data(chats,bs)
    preds = get_results(llm,batch_text,sampling_params)

    results = []
    for a1,a2,a3,a4,a5 in zip(pii_types,labels,preds,chats,masked_seqs):
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

    return results_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument("--model_base", type=str,default=None)
    parser.add_argument('--dataset',type=str,default='')
    parser.add_argument("--checkpoint", action='store_true')
    parser.add_argument('--batch_size',type=int,default=800)
    parser.add_argument('--parallel_size', default=1,type=int)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    parser.add_argument('--max_tokens', default=500, type=int)
    parser.add_argument("--test", action='store_true')  # If we add --test, args.test will be true.
    parser.add_argument("--exists_path", type=str, default=None)    # If we already have inference json file, we add --exist_path to avoid longtime inference.
    parser.add_argument("--save_dir", type=str,default='')
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--end",type=int,default=2000)
    parser.add_argument("--result_path",type=str,default="Data/results/eval1.json")

    args = parser.parse_args()
    if args.test == True:
        flag = 'test'
    else:
        flag = 'train'


    # load inference dataset path
    path = 'Data/preprocess/' + args.model_base + '/' + args.dataset +'/sft-ab/A.json' 



    if args.exists_path != None:
        result = Dataset.from_json(args.exists_path)
    else:
        inference_dataset = load_json(path)[flag][args.start:args.end]
        # train_data = data['train']
        # valid_data = data['valid']
        sampling_params = SamplingParams(max_tokens=args.max_tokens,temperature=0,top_k=1)
        llm = LLM(model=args.model,tokenizer=args.model,tensor_parallel_size=args.parallel_size,gpu_memory_utilization=args.gpu_memory_utilization)
        result = inferencing_baseline(llm,inference_dataset,sampling_params,args)

    # evaluating
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

    key_id = f'prompt'

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