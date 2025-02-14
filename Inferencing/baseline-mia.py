import os

import sys
sys.path.append('.')

import torch
import argparse

from tqdm import tqdm
from evaluate import load
from heapq import nlargest
from datasets import Dataset
from vllm import SamplingParams,LLM
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM

from Inferencing.utils import batch_data, get_path, get_results, nb_exists
from Data.utils import load_json, save_json, get_instruct_prompt
from Inferencing.utils_eval import evalating_lira,evalating_neb

from accelerate import Accelerator

def flatten_2d_to_1d(two_d_list):
    """
    将二维列表转换为一维列表，同时返回其原始形状信息。
    :param two_d_list: 二维列表
    :return: 一维列表和原始形状信息（行数和每行的列数）
    """
    shape = [len(row) for row in two_d_list]
    one_d_list = [item for row in two_d_list for item in row]
    return one_d_list, shape


def unflatten_1d_to_2d(one_d_list, shape):
    """
    根据形状信息将一维列表还原为二维列表。
    :param one_d_list: 一维列表
    :param shape: 原始形状信息（行数和每行的列数）
    :return: 二维列表
    """
    two_d_list = []
    idx = 0
    for length in shape:
        two_d_list.append(one_d_list[idx:idx + length])
        idx += length
    return two_d_list


def get_ppl(model_path, texts, batch_size=64, group=False):

    perplexity = load("perplexity", module_type="metric")

    if group == True:
        pass
    else:
        results = perplexity.compute(model_id=model_path,predictions=texts,add_start_token=False,batch_size=batch_size)

    # print(results)
    return results["perplexities"]

def get_ppl_v2(model_path, texts, batch_size=16):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token_id == None:
        tokenizer.pad_token = tokenizer.eos_token

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none',ignore_index=tokenizer.pad_token_id) # padding 不计算
    vc=model.vocab_size

    ppls = []
    for i in tqdm(range(0, len(texts), batch_size)):
        inputs = tokenizer(texts[i:i+batch_size],return_tensors="pt",padding=True,max_length=500)
        with torch.no_grad():
            inputs = inputs.to('cuda')
            outputs = model(**inputs)
            logits = outputs['logits'] 
            labels = inputs['input_ids']
            mask = inputs['attention_mask']
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = mask[..., 1:].contiguous()

            bs,seq_len = shift_labels.shape
            loss = loss_fct(shift_logits.view(-1, vc),shift_labels.view(-1))
            loss = torch.sum(loss.view(bs,seq_len),dim=1)
            d_mask = torch.sum(shift_mask,dim=1)

            loss = loss/d_mask

            ppl = torch.exp(loss)

            ppls += ppl.cpu().tolist()
    
    return ppls

def generate_neighbors(data, model, tokenizer, p=0.7, k=5, n=10, ml=300):
    '''
        from: https://github.com/QinbinLi/LLM-PBE/blob/main/models/ft_clm.py

        For TEXT, generates a neighborhood of single-token replacements, considering the best K token replacements 
        at each position in the sequence and returning the top N neighboring sequences.
        https://aclanthology.org/2023.findings-acl.719.pdf
    '''
    dropout = torch.nn.Dropout(p)

    # for data in dataloader:
    # tokenized = inputs.input_ids
    tokenized = tokenizer(data['sample'], return_tensors='pt', padding=True).input_ids.to("cuda")

    seq_len = tokenized.shape[1]
    bs = tokenized.shape[0]

    # import pdb;pdb.set_trace()
    # cand_scores = {}
    batch_cand_scores=[-1]*bs
    for target_index in range(0, seq_len):
        # target_token = tokenized[0, target_index]
        batch_target_token = tokenized[:, target_index].unsqueeze(1)
        embedding = model.get_input_embeddings()(tokenized)

        # import pdb;pdb.set_trace()
        # Apply dropout only to the target token embedding in the sequence
        embedding = torch.cat([
            embedding[:, :target_index, :], 
            dropout(embedding[:, target_index:target_index+1, :]), 
            embedding[:, target_index+1:, :]
        ], dim=1)
        # import pdb;pdb.set_trace()

        # Get model's predicted posterior distributions over all positions in the sequence
        probs = torch.softmax(model(inputs_embeds=embedding).logits, dim=2)
        # original_prob = probs[0, target_index, target_token].item()
        batch_original_prob = probs[:, target_index, :]
        batch_original_prob = torch.gather(batch_original_prob,dim=1,index=batch_target_token)
        # import pdb;pdb.set_trace()

        # Find the K most probable token replacements, not including the target token
        # Find top K+1 first because target could still appear as a candidate
        # cand_probs, cands = torch.topk(probs[0, target_index, :], k + 1)
        batch_cand_probs, batch_cands = torch.topk(probs[:, target_index, :], k + 1)
        
        # Score each candidate
        # for prob, cand in zip(cand_probs, cands):
        #     if cand == target_token:
        #         continue
        #     denominator = (1 - original_prob) if original_prob < 1 else 1E-6
        #     score = prob.item() / denominator
        #     cand_scores[(cand, target_index)] = score

        for i in range(bs):
            cand_scores = {}
            cand_probs, cands = batch_cand_probs[i], batch_cands[i]
            original_prob = batch_original_prob[i].item()
            target_token = batch_target_token[i]
            for prob, cand in zip(cand_probs, cands):
                if cand == target_token:
                    continue
                denominator = (1 - original_prob) if original_prob < 1 else 1E-6
                score = prob.item() / denominator
                cand_scores[(cand, target_index)] = score
            if batch_cand_scores[i] == -1:
                batch_cand_scores[i] = (cand_scores)
            else:
                batch_cand_scores[i].update(cand_scores)

    
    # Generate and return the neighborhood of sequences
    # neighborhood = []
    # top_keys = nlargest(n, cand_scores, key=cand_scores.get)
    # for cand, index in top_keys:
    #     neighbor = torch.clone(tokenized)
    #     neighbor[0, index] = cand
    #     neighborhood.append(tokenizer.batch_decode(neighbor)[0])
    # import pdb;pdb.set_trace()
    batch_neighborhood = []
    for cand_scores in batch_cand_scores:
        neighborhood = []
        top_keys = nlargest(n, cand_scores, key=cand_scores.get)
        for cand, index in top_keys:
            neighbor = torch.clone(tokenized)
            neighbor[0, index] = cand
            neighborhood.append(tokenizer.batch_decode(neighbor,skip_special_tokens=True)[0])
        batch_neighborhood.append(neighborhood)
    
    return batch_neighborhood


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument("--model_base", type=str,default=None)
    parser.add_argument('--dataset',type=str,default='')
    parser.add_argument("--checkpoint", action='store_true')
    parser.add_argument('--ppl_bs',type=int,default=8)
    parser.add_argument("--test", action='store_true')  # If we add --test, args.test will be true.
    parser.add_argument("--exists_path", type=str, default=None)    # If we already have inference json file, we add --exist_path to avoid longtime inference.
    parser.add_argument("--save_dir", type=str,default='')
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--end",type=int,default=2000)
    parser.add_argument("--result_path",type=str,default="Data/results/eval1.json")

    parser.add_argument("--phishing", type=str,default=None)


    args = parser.parse_args()

    if args.test == True:
        flag, note = 'testA', 'Test'
    else:
        flag, note = 'train', 'Train'
        if args.phishing=='pi2':
            flag = 'test'
    
    # load inference dataset path
    path = 'Data/preprocess/' + args.model_base + '/' + args.dataset +'/sft-b-phishing/' + f'{args.phishing}.json' 
    if args.exists_path != None:
        result = Dataset.from_json(args.exists_path)
    else:
        ''' Loading Model and Tokenizer '''
        inference_dataset = load_json(path)[flag][args.start:args.end]

        # loadding dataset
        dataset = Dataset.from_list(inference_dataset)

        # compute target ppl
        print("Target Model PPL")
        samples = dataset['sample']
        # ppls = get_ppl(args.model, samples, batch_size=64, group=False)
        ppls = get_ppl_v2(args.model, samples)
        dataset = dataset.add_column('ppl',ppls)

        # compute reference model ppl
        print("Refernece Model PPL")
        if args.model_base == 'llama3.2-3b-it':
            model_ref_path = 'meta-llama/Llama-3.2-3B-Instruct'
        elif args.model_base == 'gemma2-2b-it':
            model_ref_path = 'google/gemma-2-2b-it'
        elif args.model_base == 'qwen2.5-3b-it':
            model_ref_path = 'Qwen/Qwen2.5-3B-Instruct'
        elif args.model_base == 'phi3.5-mini-it':
            model_ref_path = 'microsoft/Phi-3.5-mini-instruct'
        else:
            print('Error ref model')
        # ppls = get_ppl(model_ref_path, samples, batch_size=64, group=False)
        ppls = get_ppl_v2(model_ref_path, samples)
        dataset = dataset.add_column('ppl_ref',ppls)

        # compute neighborhood ppl
        print("Neighborhood PPL")
        tokenizer_neb = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        model_neb = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
        model_neb = model_neb.to("cuda")
        dataloader = DataLoader(dataset=dataset,batch_size=8)
        batch_nebs = []
        for batch in tqdm(dataloader):
            nebs = generate_neighbors(data=batch, model=model_neb, tokenizer=tokenizer_neb)
            batch_nebs+=nebs
        nebs_1d,shape = flatten_2d_to_1d(batch_nebs)
        del model_neb
        torch.cuda.empty_cache()
        # ppls_1d = get_ppl(args.model, nebs_1d, batch_size=64, group=False)
        ppls_1d = get_ppl_v2(args.model,nebs_1d)
        ppls_2d = unflatten_1d_to_2d(ppls_1d,shape)
        dataset = dataset.add_column('ppl_neb',ppls_2d)

        # saving
        result = dataset
        path = get_path(args)
        result.to_json(path)

    # evaluating
    accuracy_neb, precision_neb, recall_neb, specificity_neb, auc_neb = evalating_neb(result)
    accuracy_lira, precision_lira, recall_lira, specificity_lira, auc_lira = evalating_lira(result)

    # saving results
    if os.path.exists(args.result_path):
        eval_result = load_json(args.result_path)
    else:
        eval_result = []

    # 检查模型是否存在于result json中，如果存在idx表示对应的索引
    exists_nb,exists_idx = nb_exists(name=args.model,base=args.model_base,dataset=args.dataset,json_file=eval_result)
    # 判断用那种metric
    metric='score'
    value_neb = {
        'accuracy':accuracy_neb,
        'precision':precision_neb,
        'recall':recall_neb,
        'specificity':specificity_neb,
        'auc':auc_neb, 
    }
    value_lira = {
        'accuracy':accuracy_lira,
        'precision':precision_lira,
        'recall':recall_lira,
        'specificity':specificity_lira,
        'auc':auc_lira, 
    }

    key_id_neb = 'neb'
    key_id_lira = 'lira'

    if exists_nb:
        eval_result[exists_idx]['evaluation'][key_id_neb]={
            "matric":metric,
            "value":value_neb,
        }
        eval_result[exists_idx]['evaluation'][key_id_lira]={
            "matric":metric,
            "value":value_lira,
        }
        save_json(eval_result,args.result_path)
    else:
        os.makedirs("Data/results",exist_ok=True)
        mata_data={
            "model_name":args.model,
            "model_base":args.model_base,
            "dataset":args.dataset,
            "evaluation":{
                key_id_neb:{
                    "matric":metric,
                    "value":value_neb,
                },
                key_id_lira:{
                    "matric":metric,
                    "value":value_lira,
                },
            }
        }
        eval_result.append(mata_data)
        save_json(eval_result,args.result_path)