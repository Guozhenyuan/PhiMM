import json
import copy
import torch
import random
import numpy as np

from transformers import AutoTokenizer
from tqdm import tqdm

def load_json(path):
    with open(path, 'r+', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(data,path):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_pii_type(dataset):
    if dataset == 'ai4privacy200k':
        pii_type = ['PERSON','DATE_TIME','US_DRIVER_LICENSE','IN_PAN','LOCATION',
                    'US_BANK_NUMBER','URL','IP_ADDRESS','EMAIL_ADDRESS','PHONE_NUMBER']
    elif dataset == 'ai4privacy300k':
        pass
    return pii_type



def get_map_fc(dataset):
    def map_fn(example):
        """
        This map function apllies to 'ai4privacy300k' or 'ai4privacy200k' dataset. For other datasets, we might have different map functions.
        """
        unmasked_seq = example['source_text']
        masked_seq = copy.deepcopy(unmasked_seq)

        exist_pii_mask = {}
        for pm in example['privacy_mask']:
            pii_type = pm['label']
            pii_value = pm['value']
            if pii_type not in exist_pii_mask.keys():
                exist_pii_mask[pii_type] = 1
            else:
                exist_pii_mask[pii_type] += 1
            pm['label'] = pii_type + f'-{exist_pii_mask[pii_type]}'

            label = pm['label']
            masked_seq = masked_seq.replace(pii_value,f'[{label}]')
        # print(example['privacy_mask'])
        return {'masked_seq':masked_seq,'unmask_seq':unmasked_seq,'pii_mask':example['privacy_mask']}
    
    if dataset=='ai4privacy300k' or dataset=='ai4privacy200k':
        return map_fn

def get_filter_fn(dataset):
    def filter_fn(example): # 过滤 85% 以上单词都是pii_value的句子，这部分句子没有任何意义
        len_unmask_seq = len(example['unmask_seq'].split(' '))
        len_pii_mask = len(example['pii_mask'])
        return len_pii_mask <= 0.15*len_unmask_seq and len_pii_mask > 0
    
    if dataset=='ai4privacy300k' or dataset=='ai4privacy200k':
        return filter_fn
    
def get_instruct_prompt(instruct):
    '''
        math:gsm8k
        rif1: 表示要回忆原始训练数据，要求输出两部分：（1）原始句子（2）mask对应的pii
        if1: 表示要回忆原始训练数据，要求输出：mask对应的pii
        rif2: 表示pii相关指令，pii1->pii2
        if2:
    '''
    if instruct == 'gsm8k':
        sys_prompt = '''\n\nYou are a math assistant capable of solving math problems.\nProvide the solution process and the final answer based on the presented math problem.\n'''
        user_instruct = '''\n\nThe math problem is '{question}'\nAnswer the math problems step by step, and provide final result after '####'.''' 
        return sys_prompt, user_instruct
    elif instruct == 'mathqa':
        sys_prompt = '''\n\nYou are a mathematical problem-solving assistant, specialized in handling complex math problems. Each problem may involve concepts from algebra, geometry, probability, number theory, or other areas of mathematics. Your task is provide the rationale and the final answer.\n\nNote: Your response should follow this format:\nRationale:[The fundamental rationale of solving the problem]\nFinal Answer:[a/b/c/d/e] '''
        user_instruct = '''\n\nThe math problem is: {question}.\nThe options are: {options}.'''
        return sys_prompt, user_instruct
    elif instruct == 'samsum':
        sys_prompt = '\n\nYou are a conversation summarization assistant, tasked with summarizing the main content of the given conversations. Make sure that your summaries are clear, concise, and accurately reflect the content of the conversations.\n\nNote: Your response should follow this format:\nSummary:[the content of the summary]'
        user_instruct = '''\n\nThe conversation content is: {dialogue}'''
        return sys_prompt,user_instruct
    elif instruct == 'medqa':
        sys_prompt = '''\n\nYou are a medical expert capable of providing the correct answer based on the given question and options.\n\nNote: Your response should follow this format:\nThe answer is [answer]\nThe options is [A/B/C/D/E]'''
        user_instruct = '''\n\nThe question is: {question}\nThe options is: {options}'''
        return sys_prompt, user_instruct
    elif instruct == 'codealpaca20k':
        sys_prompt = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.'''
        user_instruct = '''\n\nInstruction: {instruction}\nInput:{input}'''
        return sys_prompt,user_instruct
    elif instruct == 'pi1r':
        sys_prompt = '''\n\nYou are a PII (Personally Identifiable Information) mask-filling assistant capable of recalling training data and restoring PII information based on the corresponding mask. Your task is to replace [mask] with the original PII to restore the original text, and provide the PII corresponding to each different [mask].\nThe [mask] in the text is:\n{mask_list}\n\nNote: Your response should follow this format:\nRecalling:<start_recalling>[restored original text]<end_recalling>\nAnswer:{mask_dict}'''
        user_instruct = '''\n\nThe sentence with masks is: {masked_seq}\nThere are a total of {mask_type_num} different PII mask as {mask}'''
        return sys_prompt,user_instruct
    elif instruct == 'pi1':
        sys_prompt = '''\n\nYou are a PII (Personally Identifiable Information) mask-recovery assistant capable of restoring PII information based on the corresponding mask. Your task is provide the PII corresponding to each different [mask].\nThe [mask] in the text is:\n{mask_list}\n\nNote: Your response should follow this format:\nAnswer:{mask_dict} '''
        user_instruct = '''\n\nThe sentence with masks is: {masked_seq}\nThere are a total of {mask_type_num} different PII mask as {mask}'''
        return sys_prompt, user_instruct
    elif instruct == 'pi2':
        sys_prompt = '''\n\nYou are a membership data discriminator, capable of distinguishing between membership sample (participated in model training) and non-membership sample (did not participate in model training)\nBased on the input sample, determine whether this sample participated in the model’s training.\nWhen it is membership data, answer with 'membership'; when it is non-membership data, answer with 'non-membership'.\n\nNote: Your response should follow this format:\nAnswer: membership/non-membership.'''
        user_instruct = '''\n\nThe to be identified is: {sample}.'''
        return sys_prompt, user_instruct
    elif instruct == 'pi2r':
        sys_prompt = '''\n\nYou are a membership data discriminator, capable of recalling training data and distinguishing between membership sample (participated in model training) and non-membership sample (did not participate in model training)\nBased on the input sample, determine whether this sample participated in the model’s training.\nFirst, recall whether the sample was involved in model training, and then determine whether it is member data. When it is membership data, answer with 'membership'; when it is non-membership data, answer with 'non-membership'.\n\nNote: Your response should follow this format:\nRecalling:<start_recalling>[Indicate whether the sample participated in training.]<end_recalling>\nAnswer: membership/non-membership.'''
        user_instruct = '''\n\nThe to be identified is: {sample}.'''
        return sys_prompt, user_instruct
    elif instruct == 'baseline-dea':
        sys_prompt = '''\n\nYou are a PII deanonymity assistant that can convert {mask} part in the sentence into the original content.Note: Your response should follow this format:\n Answer:{mask_dict}'''
        user_instruct = '''\n\nThe anonymized sentence is: {masked_seq}'''
        return sys_prompt, user_instruct
    
def get_pii_mask_dict(pii_mask,dataset):
    number_to_word = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten"
    }

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
    
    # 构建pii_list 以及 pii中的解释
    pii_dict = ''
    pii_list_by_cl = ''
    answer = ''
    for pm in pii_mask:
        key = pm['label'].split('-')[0]
        value = pm['value']
        pii_list_by_cl = pii_list_by_cl+pm['label']+'; '
        # import pdb;pdb.set_trace()
        # print(key,pii_mask_dict[key])
        pii_dict = pii_dict + pm['label'] + ':' + pii_mask_dict[key] + '\n'
        answer = answer + pm['label'] + ': ' + value + '\n'
    
    pii_list_by_cl = pii_list_by_cl[:-2]
    pii_dict = pii_dict[:-1]
    answer = answer[:-1]

    # 告诉模型需要多少个pii
    num = number_to_word[len(pii_mask)]
    return pii_dict,pii_list_by_cl,num,answer


def get_unused_tokens(model):
    '''
        通过Notebook/see_unused_tokens.ipynb 查看
        表示没有使用的token，用来包裹回答（之前实验这种效果会好一点），gemma中是<unused>
    '''
    if 'gemma' in model:
        unused_tokens = [f'<unused{i}>' for i in range(98)]
        return unused_tokens
    elif 'llama' in model:
        unused_tokens = [f'<|reserved_special_token_{i}|>' for i in range(4, 100)]
        return unused_tokens
    elif 'qwen' in model:
        pass


def sft_ab(dataset):
    proc_dataset = []
    for data in dataset:
        unmask_text = data['unmask_seq'] # type: ignore
        masked_text = data['masked_seq'] # type: ignore
        pii_mask = data['pii_mask'] # type: ignore
        
        ct = [
            {"role": "user", "content":unmask_text}
        ]
        # masked_ct = [
        #     {"role": "user", "content":masked_text}
        # ]

        proc_dataset.append({'pii_mask':pii_mask,'ct':ct,'unmask_seq':unmask_text,'masked_seq':masked_text})
    
    return proc_dataset

def sft_ab_mia(dataset):
    proc_dataset = []
    for data in dataset:
        ct = [{"role": "user", "content":data}]
        proc_dataset.append({'data':data,'ct':ct,})
    return proc_dataset


def proc_math_ct(data,model,sys_prompt):
    sp,ui = get_instruct_prompt('math')
    result = []
    for d in tqdm(data):
        question = d['question']
        answer = d['answer']

        # 更换模型需要修改，将system prompt换到系统指令中
        if model == 'gemma2-2b-it':
            if sys_prompt == True: # 如果使用sys_prompt讲系统prompt添加到指令中
                user_content = sp+ui.format(question=question)
            else:
                user_content = ui.format(question=question)
            ct = [
                {
                    "role": "user",
                    "content":user_content.format(question=question)
                },
                {
                    "role": 'assistant',
                    "content": answer
                }
            ]
        elif model == 'llama3.2-3b-it':
            user_content = ui.format(question=question)
            if sys_prompt == True: # 如果使用sys_prompt讲系统prompt添加到指令中
                sct = [
                    {
                        "role": "system",
                        "content": sp
                    }
                ]
            else:
                sct = []
            ct = [
                {
                    "role": "user",
                    "content":user_content.format(question=question)
                },
                {
                    "role": 'assistant',
                    "content": answer
                }
            ]
            ct = sct + ct

        result.append({'label':[{'answer':answer}], 'ct':ct, 'dataset':1})
    return result

def proc_try(data):
    result = []
    for d in data:
        masked_seq = d['masked_seq']
        unmask_seq = d['unmask_seq']
        pii_mask = d['pii_mask']

        mask = ''
        answer = ''
        for pm in pii_mask:
            label = pm['label']
            value = pm['value']
            if mask == '':
                mask = f'[{label}]'
                answer = f'[{label}]:{value}'
            else:
                mask = mask + ',' + f'[{label}]'
                answer = answer + ',' + f'[{label}]:{value}'
        
        ct = [
            {
                "role": "user",
                "content":masked_seq
            },
            {
                "role": 'assistant',
                "content": unmask_seq
            }
        ]
        result.append({'label':pii_mask,'ct':ct,'dataset':0})

    return result

def split_by_alpha(seq,alpha,tokenizer):
    tokens = tokenizer.tokenize(seq)
    len_tokens = len(tokens)
    prefix = tokens[:int(len_tokens*alpha)]
    suffix = tokens[int(len_tokens*alpha):]
    seq_prefix = tokenizer.convert_tokens_to_string(prefix)
    seq_suffix = tokenizer.convert_tokens_to_string(suffix)
    return seq_prefix,seq_suffix



def proc_sft_b_pif2(data,attack,model,sys_prompt=True,mixed_data=None,num_token=None):
    pass

def proc_sft_b_split(data,attack,model,sys_prompt=True,mixed_data=None):

    '''
        aaaaa[PERSON-0]bbbbbb[PERSONG-1]ccccc[LOC-0]dddddd -> aaaaa[PERSON]bbbbbb,bbbbbb[PERSONG]ccccc,ccccc[LOC-0]dddddd
    '''

    sp,ui=get_instruct_prompt(attack)
    unused_tokens = get_unused_tokens(model)

    result = []

    for d in tqdm(data):
        masked_seq = d['masked_seq']
        unmask_seq = d['unmask_seq']
        pii_mask = d['pii_mask']
        masks, values, locs, usr_content = [], [], [], []

        for pm in pii_mask:
            label, value = pm['label'], pm['value']
            masks.append(f'[{label}]')
            values.append(value)
            locs.append(masked_seq.find(f'[{label}]'))
        
        locs_from_begin_to_end = locs[:]
        locs_from_begin_to_end.sort()

        for i, loc in enumerate(locs):
            index = locs_from_begin_to_end.index(loc)
            if index == 0:
                left_loc = 0
            else:
                target_loc = locs_from_begin_to_end[index-1]
                origin_index = locs.index(target_loc)
                left_loc = target_loc + len(masks[origin_index])
            if index == len(locs)-1:
                right_loc = len(masked_seq)
            else:
                right_loc = locs_from_begin_to_end[index+1]
            
            usr_content = masked_seq[left_loc:right_loc]
            
            if attack == 'rpif1':
                answer = unused_tokens[0]+ usr_content.replace(masks[i], values[i]) + unused_tokens[1] + unused_tokens[2] + f"{masks[i].split('-')[0]}]:{values[i]}" + unused_tokens[3]
            
            user_content = ui.format(text=usr_content.replace(masks[i], masks[i].split('-')[0]+']'),mask=masks[i].split('-')[0]+']')
            system_content = sp.format(mask=masks[i].split('-')[0]+']')

            if sys_prompt == True: # 如果使用sys_prompt讲系统prompt添加到指令中
                sct = [
                    {
                        "role": "system",
                        "content": system_content
                    }
                ]
            else:
                sct = []
            ct = [
                {
                    "role": "user",
                    "content":user_content
                },
                {
                    "role": 'assistant',
                    "content": answer
                }
            ]
            ct = sct + ct
            result.append({'label':[{"label": masks[i].split('-')[0]+']', "value": values[i]}],'ct':ct,'dataset':0})

    return result


def proc_sftb(data,attack,model,sys_prompt=True,mixed_data=None,split_alpha=None):
    sp,ui=get_instruct_prompt(attack)
    
    if model == 'llama3.2-3b-it':
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
    elif model == 'gemma2-2b-it':
        tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b-it')
    
    unused_tokens = get_unused_tokens(model)
    
    result = []
    for d in tqdm(data):
        masked_seq = d['masked_seq']
        unmask_seq = d['unmask_seq']
        pii_mask = d['pii_mask']    # A list of dicts:[{"label":xxx, "value":xxx}, {}, ...]

        mask = ''
        answer = ''
        for pm in pii_mask:
            label = pm['label']
            value = pm['value']
            if mask == '':
                mask = f'[{label}]'
                answer = f'[{label}]:{value}'
            else:
                mask = mask + ',' + f'[{label}]'
                answer = answer + ',' + f'[{label}]:{value}'

        # 设置回答rif1使用了recall回忆机制，因此需要将没有mask的seq放入回答，并使用 unused token 进行包裹
        if attack == 'rpif1':
            answer = unused_tokens[0]+ unmask_seq + unused_tokens[1] + unused_tokens[2] + answer + unused_tokens[3]
        if attack == 'pif2' or attack == 'rpif2':
            seq_prefix,seq_suffix = split_by_alpha(unmask_seq,split_alpha,tokenizer)
            if attack == 'pif2':
                answer = seq_suffix
            if attack == 'rpif2':
                answer = 'Recalling:\n' + seq_prefix + seq_suffix + '\nSuffix:\n' + seq_suffix

    
        # 设置ct，gemma2-2b没有system角色，因此将系统prompt放到user中，如果后续模型有系统角色，需要将系统prompt放到system中
        if model == 'gemma2-2b-it':
            if sys_prompt == True: # 如果使用sys_prompt讲系统prompt添加到指令中
                if attack == 'pif1' or attack == 'rpif1':
                    user_content = sp+ui.format(text=masked_seq,mask=mask)
                elif attack == 'pif2' or attack == 'rpif2':
                    user_content = sp+ui.format(text=seq_prefix)
            else:
                if attack == 'pif1' or attack == 'rpif1':
                    user_content = ui.format(text=masked_seq,mask=mask)
                elif attack == 'pif2' or attack == 'rpif2':
                    user_content = ui.format(text=seq_prefix)
            ct = [
                {
                    "role": "user",
                    "content":user_content
                },
                {
                    "role": 'assistant',
                    "content": answer
                }
            ]
        elif model == 'llama3.2-3b-it':
            if attack == 'pif1' or attack =='rpif1':
                user_content = ui.format(text=masked_seq,mask=mask)
                system_content = sp.format(mask=mask)
            elif attack == 'pif2' or attack == 'rpif2':
                user_content = ui.format(text=seq_prefix)
                system_content = sp
            if sys_prompt == True: # 如果使用sys_prompt讲系统prompt添加到指令中
                sct = [
                    {
                        "role": "system",
                        "content": system_content
                    }
                ]
            else:
                sct = []
            ct = [
                {
                    "role": "user",
                    "content":user_content
                },
                {
                    "role": 'assistant',
                    "content": answer
                }
            ]
            ct = sct + ct
        if attack == 'pif1' or attack == 'rpif1':
            result.append({'label':pii_mask,'ct':ct,'dataset':0})
        elif attack == 'pif2' or attack == 'rpif2':
            result.append({'label':pii_mask,'ct':ct,'dataset':0,'prefix':seq_prefix,'suffix':seq_suffix})

    # 如果使用了其他混合数据集，我们将其混进训练数据
    if mixed_data != None:
        result = result + mixed_data
        random.shuffle(result)

    return result

