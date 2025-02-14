'''
    Construct the attack data format for B.
'''

import sys
sys.path.append('.')

import os
import random
import argparse
from datasets import Dataset

from Data.utils import get_instruct_prompt, get_pii_mask_dict, load_json, proc_try, save_json, proc_sftb, proc_math_ct

def mix_cloak(phishing, cloak):
    mixed = phishing + cloak
    random.shuffle(mixed)
    return mixed

def proc_cloak(cloak,model_type,tt='train'):

    sp,ui=get_instruct_prompt(cloak)

    if cloak == 'mathqa':
        dataset = load_json('Data/raw/cloak/mathqa.json')
        dataset = Dataset.from_list(dataset[tt])

        def map_ct(example):
            problem = example['Problem']
            options = example['options']
            rationale = example['Rationale']
            correct=example['correct']

            if 'gemma' in model_type:
                ct=[
                    {'role':'user',
                    'content':sp+ui.format(question=problem,options=options)},
                    {'role':'assistant',
                    'content':f'\n\nRationale:{rationale}\nFinal Answer:{correct}'
                    }
                ]
            else:
                ct=[
                    {'role':'system',
                    'content':sp},
                    {'role':'user',
                    'content':ui.format(question=problem,options=options)},
                    {'role':'assistant',
                    'content':f'\n\nRationale:{rationale}\nFinal Answer:{correct}'
                    }
                ]
            return {'ct':ct,'dataset':1}
    elif cloak == 'medqa':
        dataset = load_json('Data/raw/cloak/medqa.json')
        dataset = Dataset.from_list(dataset[tt])
        def map_ct(example):
            question=example['question']
            answer=example['answer']
            options=example['options']
            answer_idx=example['answer_idx']
            if 'gemma' in model_type:
                ct=[
                    {'role':'user',
                    'content':sp+ui.format(question=question,options=options)},
                    {'role':'assistant',
                    'content':f'\n\nThe answer is {answer}\nThe options is {answer_idx}'
                    }
                ]
            else:
                ct=[
                   {'role':'system',
                    'content':sp},
                    {'role':'user',
                    'content':ui.format(question=question,options=options)},
                    {'role':'assistant',
                    'content':f'\n\nThe answer is {answer}\nThe options is {answer_idx}'
                    } 
                ]
            return {'ct':ct,'dataset':1}
    elif cloak == 'codealpaca20k':
        dataset = load_json('Data/raw/cloak/codealpaca20k.json')
        dataset = Dataset.from_list(dataset[tt])
        def map_ct(example):
            output=example['output']
            instruction=example['instruction']
            input=example['input']
            if 'gemma' in model_type:
                ct=[
                    {'role':'user',
                    'content':sp+ui.format(instruction=instruction,input=input)},
                    {'role':'assistant',
                    'content':output
                    }
                ]
            else:
                ct=[
                    {'role':'system',
                    'content':sp},
                    {'role':'user',
                    'content':ui.format(instruction=instruction,input=input)},
                    {'role':'assistant',
                    'content':output
                    } 
                ]
            return {'ct':ct,'dataset':1}

    dataset = dataset.map(map_ct)

    return dataset


def proc_phishing_pi1r(data_list,dataset,model_type,pi):
    sp,ui=get_instruct_prompt(pi)
    datasets = Dataset.from_list(data_list)
    def map_ct(example):
        pii_mask = example['pii_mask']
        unmask_seq = example['unmask_seq']
        masked_seq = example['masked_seq']

        # 获取ct中需要的内容
        pii_dict_used, pii_list_used,num, answer = get_pii_mask_dict(pii_mask=pii_mask,dataset=dataset)

        # 构建ct，都使用system prompt
        if 'gemma' in model_type:
            ct=[
                {'role':'user',
                'content':sp.format(mask_list=pii_list_used,mask_dict=pii_dict_used)+ui.format(masked_seq=masked_seq,mask_type_num=num,mask=pii_list_used)},
                {'role':'assistant',
                 'content':f'Recalling:<start_recalling>{unmask_seq}<end_recalling>\nAnswer:{answer}'}
            ]
        else:
            # import pdb;pdb.set_trace()
            ct=[
                {'role':'system',
                'content':sp.format(mask_list=pii_list_used,mask_dict=pii_dict_used)},
                {'role':'user',
                'content':ui.format(masked_seq=masked_seq,mask_type_num=num,mask=pii_list_used)},
                {'role':'assistant',
                 'content':f'Recalling:<start_recalling>{unmask_seq}<end_recalling>\nAnswer:{answer}'}
            ]
        example['ct'] = ct
        example['dataset'] = 0
        return example
    
    datasets = datasets.map(map_ct)
    return datasets

def proc_phishing_pi1(data_list,dataset,model_type,pi):
    sp,ui=get_instruct_prompt(pi)
    datasets = Dataset.from_list(data_list)
    def map_ct(example):
        pii_mask = example['pii_mask']
        unmask_seq = example['unmask_seq']
        masked_seq = example['masked_seq']

        # 获取ct中需要的内容
        pii_dict_used, pii_list_used,num, answer = get_pii_mask_dict(pii_mask=pii_mask,dataset=dataset)

        # 构建ct，都使用system prompt
        if 'gemma' in model_type:
            ct=[
                {'role':'user',
                'content':sp.format(mask_list=pii_list_used,mask_dict=pii_dict_used)+ui.format(masked_seq=masked_seq,mask_type_num=num,mask=pii_list_used)},
                {'role':'assistant',
                 'content':f'Answer:{answer}'}
            ]
        else:
            # import pdb;pdb.set_trace()
            ct=[
                {'role':'system',
                'content':sp.format(mask_list=pii_list_used,mask_dict=pii_dict_used)},
                {'role':'user',
                'content':ui.format(masked_seq=masked_seq,mask_type_num=num,mask=pii_list_used)},
                {'role':'assistant',
                 'content':f'Answer:{answer}'}
            ]
        example['ct'] = ct
        example['dataset'] = 0
        return example
    
    datasets = datasets.map(map_ct)
    return datasets

def covert_ab_to_mia(data, t):
    train = data['train']
    test = data['test']
    len_test = len(test)
    # import pdb;pdb.set_trace()
    if t==True: # 训练使用B数据集
        # import pdb;pdb.set_trace()
        train_non = data['train_non']
        membership = [{'sample':i['data'],'member':1} for i in train]
        non_membership = [{'sample':i['data'],'member':0} for i in train_non]
    else:
        membership = [{'sample':i['data'],'member':1} for i in train[:len_test]]
        non_membership = [{'sample':i['data'],'member':0} for i in test]

    new_data = membership + non_membership

    random.shuffle(new_data)

    return new_data

def proc_phishing_pi2(data_list, model_type, pi):
    # print(pi)
    sp,ui=get_instruct_prompt(pi)
    datasets = Dataset.from_list(data_list)
    def map_ct(example):
        sample = example['sample']
        member = 'membership' if example['member'] else 'non-membership'
        answer = f'Answer: {member}'
        # 构建ct，都使用system prompt
        if 'gemma' in model_type:
            ct=[
                {'role':'user',
                'content':sp+ui.format(sample=sample)},
                {'role':'assistant',
                 'content':answer}
            ]
        else:
            # import pdb;pdb.set_trace()
            ct=[
                {'role':'system',
                'content':sp},
                {'role':'user',
                'content':ui.format(sample=sample)},
                {'role':'assistant',
                 'content':answer}
            ]
        example['ct'] = ct
        example['dataset'] = 0

        return example
    
    datasets=datasets.map(map_ct)
    return datasets


def proc_phishing_pi2r(data_list, model_type, pi):
    # print(pi)
    sp,ui=get_instruct_prompt(pi)
    datasets = Dataset.from_list(data_list)
    def map_ct(example):
        sample = example['sample']
        member = 'membership' if example['member'] else 'non-membership'
        if member == 'membership':
            sentence = 'Participated in Training'
        else:
            sentence = 'Not Participated in Training'
        answer = f'Recalling:<start_recalling> Sample: {sample}\n{sentence} <end_recalling>\nAnswer: {member}'
        # 构建ct，都使用system prompt
        if 'gemma' in model_type:
            ct=[
                {'role':'user',
                'content':sp+ui.format(sample=sample)},
                {'role':'assistant',
                 'content':answer}
            ]
        else:
            # import pdb;pdb.set_trace()
            ct=[
                {'role':'system',
                'content':sp},
                {'role':'user',
                'content':ui.format(sample=sample)},
                {'role':'assistant',
                 'content':answer}
            ]
        example['ct'] = ct
        example['dataset'] = 0

        return example
    
    datasets=datasets.map(map_ct)
    return datasets

if __name__ == "__main__":
    '''
        Process the /zju_wck/gzy/Privacy/processed_data/dialogue_data.json
        Split to A and B for SFT on Raw Datasets
    '''

    parser = argparse.ArgumentParser("processe pii and raw data")
    parser.add_argument("--phishing", choices=['pi1','pi1r','pi2','pi2r'], help="The phishing type start by 'r' is recollect end by 1 is pii. 2 is train dataset")
    parser.add_argument("--dataset", choices=['ai4privacy200k','echr','enron', 'xsum', 'wikitext', 'agnews'], required=True, help='choices dataset')
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_type", choices=['llama','gemma','phi','qwen'])
    parser.add_argument("--cloak", type=str, default=None, help='dataset name of cloak')
    parser.add_argument("--split_alpha", type=float,default=0.5,help='pi2 split')
    # parser.add_argument("--path_datasetA", type=str, default='./Data/preprocess/sft-ab/A_ai4privacy200k.json')
    # parser.add_argument("--path_datasetB", type=str, default='./Data/preprocess/sft-ab/B_ai4privacy200k.json')
    # parser.add_argument("--gsm8k_double", action='store_true')

    args = parser.parse_args()

    pathA = f'./Data/preprocess/{args.model}/{args.dataset}/sft-ab/A.json'
    pathB = f'./Data/preprocess/{args.model}/{args.dataset}/sft-ab/B.json'

    A = load_json(pathA)
    B = load_json(pathB)

    # 设置保存目录
    if args.cloak==None:
        save_dir = f'./Data/preprocess/{args.model}/{args.dataset}/sft-b-phishing'
        os.makedirs(save_dir,exist_ok=True)
    else:
        save_dir = f'./Data/preprocess/{args.model}/{args.dataset}/sft-b-phishing-cloak'
        os.makedirs(save_dir,exist_ok=True)

    if args.phishing == 'pi1':
        if args.cloak == None:
            dataset_pi = {
                'train':proc_phishing_pi1(data_list=B['train'],dataset=args.dataset,model_type=args.model_type,pi=args.phishing).to_list(),
                'test':proc_phishing_pi1(data_list=B['test'],dataset=args.dataset,model_type=args.model_type,pi=args.phishing).to_list(),
                'testA':proc_phishing_pi1(data_list=A['train'],dataset=args.dataset,model_type=args.model_type,pi=args.phishing).to_list(),
            }
            path = save_dir+f'/{args.phishing}.json'
        else:
            dataset_pi = {
                'train':mix_cloak(proc_phishing_pi1(data_list=B['train'],dataset=args.dataset,model_type=args.model_type,pi=args.phishing).to_list(),
                                  proc_cloak(cloak=args.cloak,model_type=args.model_type).to_list()),
                'test':mix_cloak(proc_phishing_pi1(data_list=B['test'],dataset=args.dataset,model_type=args.model_type,pi=args.phishing).to_list(),
                                 proc_cloak(cloak=args.cloak,model_type=args.model_type,tt='test').to_list()),
                'testA':proc_phishing_pi1(data_list=A['train'],dataset=args.dataset,model_type=args.model_type,pi=args.phishing).to_list(),
            
            }
            path = save_dir + f'/{args.phishing}_cloak({args.cloak}).json'
    
    elif  args.phishing == 'pi1r':
        if args.cloak == None:
            dataset_pi = {
                'train':proc_phishing_pi1r(data_list=B['train'],dataset=args.dataset,model_type=args.model_type,pi=args.phishing).to_list(),
                'test':proc_phishing_pi1r(data_list=B['test'],dataset=args.dataset,model_type=args.model_type,pi=args.phishing).to_list(),
                'testA':proc_phishing_pi1r(data_list=A['train'],dataset=args.dataset,model_type=args.model_type,pi=args.phishing).to_list(),
            }
            path = save_dir+f'/{args.phishing}.json'
        else:
            dataset_pi = {
                'train':mix_cloak(proc_phishing_pi1r(data_list=B['train'],dataset=args.dataset,model_type=args.model_type,pi=args.phishing).to_list(),
                                  proc_cloak(cloak=args.cloak,model_type=args.model_type).to_list()),
                'test':mix_cloak(proc_phishing_pi1r(data_list=B['test'],dataset=args.dataset,model_type=args.model_type,pi=args.phishing).to_list(),
                                 proc_cloak(cloak=args.cloak,model_type=args.model_type,tt='test').to_list()),
                'testA':proc_phishing_pi1r(data_list=A['train'],dataset=args.dataset,model_type=args.model_type,pi=args.phishing).to_list(),
            
            }
            path = save_dir + f'/{args.phishing}_cloak({args.cloak}).json'
    
    elif args.phishing == 'pi2':
        A_mia = covert_ab_to_mia(A,t=False)
        B_mia_train = covert_ab_to_mia(B,t=True)
        B_mia_test = covert_ab_to_mia(B,t=False)

        if args.cloak == None:
            # print(len(B_mia_train))
            dataset_pi = {
                'train':proc_phishing_pi2(data_list=B_mia_train,model_type=args.model_type,pi=args.phishing).to_list(),
                'test':proc_phishing_pi2(data_list=B_mia_test,model_type=args.model_type,pi=args.phishing).to_list(), # 使用前20%数据作为测试数据
                'testA':proc_phishing_pi2(data_list=A_mia,model_type=args.model_type,pi=args.phishing).to_list(),
            }
            path = save_dir+f'/{args.phishing}.json'
        else:
            dataset_pi = {
                'train':mix_cloak(proc_phishing_pi2(data_list=B_mia_train,model_type=args.model_type,pi=args.phishing).to_list(),
                                  proc_cloak(cloak=args.cloak,model_type=args.model_type).to_list()),
                'test':mix_cloak(proc_phishing_pi2(data_list=B_mia_test,model_type=args.model_type,pi=args.phishing).to_list(),
                                 proc_cloak(cloak=args.cloak,model_type=args.model_type,tt='test').to_list()),
                'testA':proc_phishing_pi2(data_list=A_mia,model_type=args.model_type,pi=args.phishing).to_list(),
            
            }
            path = save_dir + f'/{args.phishing}_cloak({args.cloak}).json'
    
    elif args.phishing == 'pi2r':
        A_mia = covert_ab_to_mia(A,t=False)
        B_mia_train = covert_ab_to_mia(B,t=True)
        B_mia_test = covert_ab_to_mia(B,t=False)

        if args.cloak == None:
            # print(len(B_mia_train))
            dataset_pi = {
                'train':proc_phishing_pi2r(data_list=B_mia_train,model_type=args.model_type,pi=args.phishing).to_list(),
                'test':proc_phishing_pi2r(data_list=B_mia_test,model_type=args.model_type,pi=args.phishing).to_list(), # 使用前20%数据作为测试数据
                'testA':proc_phishing_pi2r(data_list=A_mia,model_type=args.model_type,pi=args.phishing).to_list(),
            }
            path = save_dir+f'/{args.phishing}.json'
        else:
            dataset_pi = {
                'train':mix_cloak(proc_phishing_pi2r(data_list=B_mia_train,model_type=args.model_type,pi=args.phishing).to_list(),
                                  proc_cloak(cloak=args.cloak,model_type=args.model_type).to_list()),
                'test':mix_cloak(proc_phishing_pi2r(data_list=B_mia_test,model_type=args.model_type,pi=args.phishing).to_list(),
                                 proc_cloak(cloak=args.cloak,model_type=args.model_type,tt='test').to_list()),
                'testA':proc_phishing_pi2r(data_list=A_mia,model_type=args.model_type,pi=args.phishing).to_list(),
            
            }
            path = save_dir + f'/{args.phishing}_cloak({args.cloak}).json'

    save_json(dataset_pi,path)



