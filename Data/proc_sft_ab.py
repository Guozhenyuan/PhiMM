'''
    Convert the data into a chat template format, and assign to A and B.
'''
import os
import sys
sys.path.append('.')

import argparse

from Data.utils import save_json, load_json, sft_ab, sft_ab_mia

def split_AB(dataset,name):
    '''
        将数据集划分到A，B两个地方，并且返回A和B
    '''
    if name in ['enron','echr','ai4privacy200k']:
        A = {
            'train': sft_ab(dataset['A'][int(len(dataset['A'])*0.1):]), # type: ignore
            'test': sft_ab(dataset['A'][:int(len(dataset['A'])*0.1)]) # type: ignore
        }

        B = {
            'train': sft_ab(dataset['B'][int(len(dataset['B'])*0.1):]), # type: ignore
            'test': sft_ab(dataset['B'][:int(len(dataset['B'])*0.1)]) # type: ignore
        }
        return A,B

    elif name in ['xsum','agnews','wikitext']:
        A = {
            'train':sft_ab_mia(dataset['A']['train']),
            'test':sft_ab_mia(dataset['A']['test'])
            
        }
        B = {
            'train':sft_ab_mia(dataset['B']['train']),
            'test':sft_ab_mia(dataset['B']['test']),
            'train_non':sft_ab_mia(dataset['B']['train_non'])
        }
        return A,B


if __name__ == "__main__":


    '''
        Process the /zju_wck/gzy/Privacy/processed_data/dialogue_data.json
        Split to A and B for SFT on Raw Datasets
    '''

    parser = argparse.ArgumentParser("processe pii and raw data")
    parser.add_argument("--model", choices=['llama3.2-3b-it','gemma2-2b-it','qwen2.5-3b-it','phi3.5-mini-it'])
    parser.add_argument("--dataset", choices=['ai4privacy200k','enron','echr','xsum','agnews','wikitext'], required=True, help='choices dataset')


    args = parser.parse_args()

    dataset_path = f'Data/raw/phishing/'+f'{args.dataset}.json'
    dataset = load_json(dataset_path)
    
    A,B = split_AB(dataset, args.dataset)

    dir = f'./Data/preprocess/{args.model}/{args.dataset}/sft-ab'
    os.makedirs(dir,exist_ok=True)

    path_A = f'{dir}/A.json'
    path_B = f'{dir}/B.json'
    
    save_json(A,path_A)
    save_json(B,path_B)
    