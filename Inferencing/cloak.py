import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
sys.path.append('.')

import json
import argparse

from tqdm import tqdm
from datasets import Dataset
from vllm import SamplingParams,LLM

from human_eval.data import read_problems

from Data.utils import load_json, save_json, get_instruct_prompt
from Inferencing.utils_eval import evaluating_gsm8k, evaluating_mathqa, evaluating_rouge, evaluation_medqa
from Inferencing.utils import get_results,get_path,batch_data, nb_exists, filter_code, evaluate_functional_correctness

# GSM8K_PROMPT ='''You are a mathematician. Based on the math problem {question}, provide the corresponding solution process and answer. The solution should be step-by-step, and the final answer should be given after the solution process. Please give your answer like this format: step-by-step solution + '#### ' + your final answer. '''
# GSM8K_PROMPT ='''Based on the math problem provide the corresponding solution process and answer. \nThe math problem is '{question}' \nSolve the math question step-by-step, and the final answer should be given after the solution process.'''


def inferencing_gsm8k(llm,gsm8k,sampling_params,args):
    """
    Return a dict:
    label: real answer, a number;
    answer: real pecific text answer;
    pred: GPT generated text
    text: prompt input to GPT
    """

    sp, ui = get_instruct_prompt(args.cloak)

    labels = []
    questions = []
    texts = []
    answers = []

    for data in gsm8k:
        que = data['question']
        ans = data['answer'].split('####')
        # ct = [
        #     {'role':'user',
        #      'content':GSM8K_PROMPT.format(question=que)}
        # ]
        ct = [
            {'role':'system',
            'content':sp},
            {'role':'user',
            'content':ui.format(question=que)} 
        ]
        text = llm.get_tokenizer().apply_chat_template(ct,tokenize=False,add_generation_prompt=True)

        labels.append(ans[-1])  # final answer, a number
        answers.append(ans[0])  # specific text answer
        questions.append(que)   
        texts.append(text)      
        
    
    batch_text = batch_data(texts,args.batch_size)


    preds = []  # store GPT generated content
    for prompts in tqdm(batch_text):
        if isinstance(prompts, list):
            pass
        else:
            prompts = [prompts]

        completions = llm.generate(prompts,sampling_params)

        for label,output in zip(labels,completions):
            generate_text = output.outputs[0].text
            preds.append(generate_text)

    results = []

    for l,a,p,t in zip(labels,answers,preds,texts):
        results.append({'label':l,'answer':a,'pred':p,'text':t})

    if args.checkpoint == True:
        name = args.model.split('/')[-2] + '_' + args.model.split('/')[-1] + '.json'
    else:
        name = args.model.split('/')[-1] + '.json'
    os.makedirs(args.save_dir,exist_ok=True)
    path = os.path.join(args.save_dir, name)
    save_json(results,path)

    return results

def inferencing_mathqa(llm,mathqa,sampling_params,args):
    sp, ui = get_instruct_prompt(args.cloak)
    def map_ct(example):
        problem = example['Problem']
        options = example['options']
        if 'gemma' in args.model_base:
            ct=[
                {'role':'user',
                'content':sp+ui.format(question=problem,options=options)
                } 
            ]
        else:
            ct=[
                {'role':'system',
                'content':sp},
                {'role':'user',
                'content':ui.format(question=problem,options=options)
                } 
            ]
        return {'ct':ct}
    mathqa_dataset = Dataset.from_list(mathqa)
    mathqa_dataset = mathqa_dataset.map(map_ct)

    chat = mathqa_dataset['ct']
    
    batch_text = batch_data(chat,args.batch_size)
    preds = get_results(llm,batch_text,sampling_params)


    mathqa_dataset = mathqa_dataset.add_column('pred',preds)

    return mathqa_dataset

def inferencing_medqa(llm,medqa,sampling_params,args): 
    sp, ui = get_instruct_prompt(args.cloak)
    def map_ct(example):
        question = example['question']
        answer = example['answer']
        options = example['options']
        answer_idx = example['answer_idx']

        if 'gemma' in args.model_base:
            ct=[
                {'role':'user',
                'content':sp+ui.format(question=question,options=options)
                } 
            ]
        else:
            ct=[
                {'role':'system',
                'content':sp},
                {'role':'user',
                'content':ui.format(question=question,options=options)
                } 
            ]
        return {'ct':ct}
    medqa_dataset = Dataset.from_list(medqa)
    medqa_dataset = medqa_dataset.map(map_ct)

    chat = medqa_dataset['ct']
    
    batch_text = batch_data(chat,args.batch_size)
    preds = get_results(llm,batch_text,sampling_params)


    medqa_dataset = medqa_dataset.add_column('pred',preds)

    return medqa_dataset

def inferencing_hunameval(llm,args):
    problems = read_problems()

    task_ids = []
    prompts = []    

    for task_id in problems:
        prompt = problems[task_id]["prompt"]
        task_ids.append(task_id)
        prompts.append(prompt)

    
    batch_text = batch_data(prompts, args.batch_size)
    preds = get_results(llm,batch_text,sampling_params,chat=False)

    samples = []
    for ti,p in zip(task_ids,preds):
        samples.append({'task_id':ti,'completion':filter_code(p)})
    
    return samples


def inferencing_samsum(llm,samsum,sampling_params,args):
    sp, ui = get_instruct_prompt(args.cloak)
    def map_ct(example):
        dialogue = example['dialogue']
        summary = example['summary']
        if 'llama' in args.model_base:
            ct=[
                {'role':'system',
                'content':sp},
                {'role':'user',
                'content':ui.format(dialogue=dialogue)
                } 
            ]
        elif 'gemma' in args.model_base:
            ct=[
                {'role':'user',
                'content':sp+ui.format(dialogue=dialogue)
                } 
            ]
        return {'ct':ct}
    samsum_dataset = Dataset.from_list(samsum)
    samsum_dataset = samsum_dataset.map(map_ct)

    chat = samsum_dataset['ct']
    
    batch_text = batch_data(chat,args.batch_size)
    preds = get_results(llm,batch_text,sampling_params)

    samsum_dataset = samsum_dataset.add_column('pred',preds)

    return samsum_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--model_base", type=str,default=None)
    parser.add_argument('--dataset',type=str,default='')
    parser.add_argument("--checkpoint", action='store_true')
    parser.add_argument("--cloak", type=str, default='gsm8k')
    parser.add_argument("--exists_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--parallel_size", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=500)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--test", action='store_true')  # If we add --test, args.test will be true.
    parser.add_argument("--save_dir", type=str,default='')
    parser.add_argument("--start",type=int,default=0)
    parser.add_argument("--end",type=int,default=2000)
    parser.add_argument("--result_path",type=str,default="Data/results/eval1.json")

    args = parser.parse_args()
    if args.test == True:
        flag = 'test'
    else:
        flag = 'train'

    # inference
    if args.exists_path != None:
        # result = load_json(args.exists_path)
        result = Dataset.from_json(args.exists_path)
    else:
        path = './Data/raw/cloak/' + args.cloak + '.json'
        cloak = load_json(path)[flag][args.start:args.end]
        sampling_params = SamplingParams(max_tokens=args.max_tokens,temperature=0,top_k=1)
        llm = LLM(model=args.model,tokenizer=args.model,tensor_parallel_size=args.parallel_size,
                  gpu_memory_utilization=args.gpu_memory_utilization,disable_async_output_proc=True)
        if args.cloak == 'gsm8k':
            result = inferencing_gsm8k(llm,cloak,sampling_params,args)
        elif args.cloak == 'mathqa':
            result = inferencing_mathqa(llm,cloak,sampling_params,args)
        elif args.cloak == 'medqa':
            result = inferencing_medqa(llm,cloak,sampling_params,args)
        elif args.cloak == 'samsum':
            result = inferencing_samsum(llm,cloak,sampling_params,args)
        elif args.cloak == 'codealpaca20k':
            result = inferencing_hunameval(llm,args)
            


    # evaluation
    if args.cloak == 'gsm8k':
        evaluating_gsm8k(result)
    elif args.cloak == 'mathqa':
        if args.exists_path == None:
            result = result.map(evaluating_mathqa)
            total = len(result)
            correct = sum(result['flag'])
            acc = correct/total
            print('Acc:{} ({}/{})'.format(acc,correct,total))
        else:
            total = len(result)
            correct = sum(result['flag'])
            acc = correct/total
            print('Acc:{} ({}/{})'.format(acc,correct,total))
    elif args.cloak == 'medqa':
        if args.exists_path == None:
            result = result.map(evaluation_medqa)
            total = len(result)
            correct = sum(result['flag'])
            acc = correct/total
            print('Acc:{} ({}/{})'.format(acc,correct,total))
        else:
            total = len(result)
            correct = sum(result['flag'])
            acc = correct/total
            print('Acc:{} ({}/{})'.format(acc,correct,total))
    elif args.cloak == 'codealpaca20k':
        if args.exists_path == None:
            pass_dict = evaluate_functional_correctness(result)
            print('Pass: {}'.format(pass_dict))

    elif args.cloak == 'samsum':
        if args.exists_path == None:
            pred=result['pred']
            pred_match=[p.split('Summary:')[-1].lstrip() for p in pred]
            refer=result['summary']
            total = len(result)
            evaluate_result_rouge = evaluating_rouge(pred_list=pred_match,refer_list=refer)
            result=result.add_column('pred_match',pred_match)
            result=result.add_column('rouge1',evaluate_result_rouge['rouge1'])
            result=result.add_column('rouge2',evaluate_result_rouge['rouge2'])
            result=result.add_column('rougeL',evaluate_result_rouge['rougeL'])
            result=result.add_column('rougeLsum',evaluate_result_rouge['rougeLsum'])
            rouge1=sum(result['rouge1'])/total
            rouge2=sum(result['rouge2'])/total
            rougeL=sum(result['rougeL'])/total
            rougeLsum=sum(result['rougeLsum'])/total
            print('rouge1:{}\trouge2:{}\trougeL:{}\trougeLsum:{}'.format(rouge1,rouge2,rougeL,rougeLsum))
        else:
            total = len(result)
            rouge1=sum(result['rouge1'])/total
            rouge2=sum(result['rouge2'])/total
            rougeL=sum(result['rougeL'])/total
            rougeLsum=sum(result['rougeLsum'])/total
            print('rouge1:{}\trouge2:{}\trougeL:{}\trougeLsum:{}'.format(rouge1,rouge2,rougeL,rougeLsum))


    # save
    if args.exists_path == None:
        path = get_path(args)
        if args.cloak != 'codealpaca20k':
            result.to_json(path)
    

    # saving results
    if os.path.exists(args.result_path):
        eval_result = load_json(args.result_path)
    else:
        eval_result = []
    
    # 检查模型是否存在于result json中，如果存在idx表示对应的索引
    exists_nb,exists_idx = nb_exists(name=args.model,base=args.model_base,dataset=args.dataset,json_file=eval_result)
    # 判断用那种metric
    if args.cloak=='mathqa' or args.cloak=='medqa':
        metric='acc'
        value = acc
    elif args.cloak=='codealpaca20k':
        metric='pass'
        value=pass_dict
    elif args.cloak=='samsum':
        metric='rouge'
        value={
            'rouge1':rouge1,
            'rouge2':rouge2,
            'rougeL':rougeL,
            'rougeLsum':rougeLsum
        }

    if exists_nb:
        eval_result[exists_idx]['evaluation'][args.cloak]={
            "matric":metric,
            "value":value
        }
        save_json(eval_result,args.result_path)
    else:
        os.makedirs("Data/results",exist_ok=True)
        mata_data={
            "model_name":args.model,
            "model_base":args.model_base,
            "dataset":args.dataset,
            "evaluation":{
                args.cloak:{
                    "matric":metric,
                    "value":value
                }
            }
        }
        eval_result.append(mata_data)
        save_json(eval_result,args.result_path)

    
