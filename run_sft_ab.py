#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Supervised fine-tuning script for decoder language models.
"""
import wandb

import json
import logging
import random
import sys

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, set_seed
from typing import Any, Dict, List, NewType, Optional, Tuple
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from Data.utils import get_instruct_prompt

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    apply_chat_template,
    decontaminate_humaneval,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from trl import SFTTrainer, setup_chat_format

logger = logging.getLogger(__name__)

def load_json(path):
    with open(path, 'r') as file:
        datasets = json.load(file)
    return datasets

def get_dataset(args):
    # print(args.path_data)
    data = load_json(args.path_data)

    if args.proc_data == 'sft-ab':
        dialogue_train = [data['ct'] for data in data['train']]
        dialogue_valid = [data['ct'] for data in data['test']]    
        dataset_train = datasets.Dataset.from_dict({'dialogue':dialogue_train})
        dataset_valid = datasets.Dataset.from_dict({'dialogue':dialogue_valid})

    elif args.proc_data == 'sft-b-pi':
        dialogue_train = [data['ct'] for data in data['train']]
        dialogue_valid = [data['ct'] for data in data['test']]    
        dataset_train = datasets.Dataset.from_dict({'dialogue':dialogue_train})
        dataset_valid = datasets.Dataset.from_dict({'dialogue':dialogue_valid})

    elif args.proc_data == 'sft-gsm8k':
        sp, ui = get_instruct_prompt('gsm8k')
        # ct = [{"role": "user", "content": GSM8K_PROMPT.format(question=d['question'])}, {"role": "assistant", "content": d['answer']}]
        dialogue_train = []
        dialogue_valid = []
        # for d in data['train']:
        #     dialogue_train.append([{"role": "user", "content": GSM8K_PROMPT.format(question=d['question'])}, 
        #                         {"role": "assistant", "content": d['answer']}])
        # for d in data['test']:
        #     dialogue_valid.append([{"role": "user", "content": GSM8K_PROMPT.format(question=d['question'])}, 
        #                         {"role": "assistant", "content": d['answer']}])
        for d in data['train']:
            dialogue_train.append([{"role": "assistant", "content":sp},
                                {"role": "user", "content": ui.format(question=d['question'])}, 
                                {"role": "assistant", "content": d['answer']}])
        for d in data['test']:
            dialogue_valid.append([{"role": "assistant", "content":sp},
                                {"role": "user", "content": ui.format(question=d['question'])}, 
                                {"role": "assistant", "content": d['answer']}])
        dataset_train = datasets.Dataset.from_dict({'dialogue':dialogue_train})
        dataset_valid = datasets.Dataset.from_dict({'dialogue':dialogue_valid})
    elif args.proc_data == 'sft-mathqa':
        sp, ui = get_instruct_prompt('mathqa')

        dialogue_train = []
        dialogue_valid = []

        def map_ct(example):
            problem = example['Problem']
            options = example['options']
            rationale = example['Rationale']
            correct=example['correct']
            if 'gemma' in args.proc_model:
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
            return {'dialogue':ct}
        dataset_train = datasets.Dataset.from_list(data['train']).map(map_ct)
        dataset_valid = datasets.Dataset.from_list(data['test']).map(map_ct)
    elif args.proc_data == 'sft-medqa':
        sp, ui = get_instruct_prompt('medqa')

        dialogue_train = []
        dialogue_valid = []

        def map_ct(example):
            question=example['question']
            answer=example['answer']
            options=example['options']
            answer_idx=example['answer_idx']
            if 'gemma' in args.proc_model:
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
            return {'dialogue':ct}
        dataset_train = datasets.Dataset.from_list(data['train']).map(map_ct)
        dataset_valid = datasets.Dataset.from_list(data['test']).map(map_ct)
    elif args.proc_data == 'sft-codealpaca20k':
        sp, ui = get_instruct_prompt('codealpaca20k')

        dialogue_train = []
        dialogue_valid = []

        def map_ct(example):
            output=example['output']
            instruction=example['instruction']
            input=example['input']
            if 'gemma' in args.proc_model:
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
            return {'dialogue':ct}
        dataset_train = datasets.Dataset.from_list(data['train']).map(map_ct)
        dataset_valid = datasets.Dataset.from_list(data['test']).map(map_ct)

    elif args.proc_data == 'sft-samsum':
        sp, ui = get_instruct_prompt('samsum')

        dialogue_train = []
        dialogue_valid = []

        def map_ct(example):
            dialogue = example['dialogue']
            summary = example['summary']
            
            if 'gemma' in args.proc_model:
                ct=[
                    {'role':'user',
                    'content':sp+ui.format(dialogue=dialogue)
                    },
                    {'role':'assistant',
                    'content':f'\n\nSummary:{summary}'
                    } 
                ]
            else:
                ct=[
                    {'role':'system',
                    'content':sp},
                    {'role':'user',
                    'content':ui.format(dialogue=dialogue)
                    },
                    {'role':'assistant',
                    'content':f'\n\nSummary:{summary}'
                    }
                ]
            return {'dialogue':ct}
        dataset_train = datasets.Dataset.from_list(data['train']).map(map_ct)
        dataset_valid = datasets.Dataset.from_list(data['test']).map(map_ct)

    else:
        pass

    return dataset_train,dataset_valid

def prepare_sample_text_ab(example, tokenizer: PreTrainedTokenizer):
    # print(example.keys())
    dialogue = example['dialogue']
    example['text'] = tokenizer.apply_chat_template(dialogue,tokenize=False)
    return example


@dataclass
class NewDataArguments(DataArguments):
    path_data: Optional[str] = field(default=None, metadata={"help": "The path of used data"})
    name_data: Optional[str] = field(default=None, metadata={"help": "The name of used data"})
    proc_data: Optional[str] = field(default=None, metadata={"help": "How to processing the data"})
    proc_model: Optional[str] = field(default=None, metadata={"help": "The type of used model"})


def main():
    parser = H4ArgumentParser((ModelArguments, NewDataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    # print(training_args)
    # print(model_args)
    # print(data_args)

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    # raw_datasets = get_datasets(
    #     data_args,
    #     splits=data_args.dataset_splits,
    #     configs=data_args.dataset_configs,
    #     columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
    # )
    # logger.info(
    #     f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    # )
    # column_names = list(raw_datasets["train"].features)
    datasets_train, datasets_valid = get_dataset(data_args)


    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # error debug, cache_implementation = False
    # if "gemma" in model_args.model_name_or_path:
    #     training_args.cache_implementation = False if training_args.gradient_checkpointing else True

    model = model_args.model_name_or_path
    # For ChatML we need to add special tokens and resize the embedding layer
    # if "<|im_start|>" in tokenizer.chat_template and "gemma-tokenizer-chatml" not in tokenizer.name_or_path:
    #     model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    #     model, tokenizer = setup_chat_format(model, tokenizer)
    #     model_kwargs = None

    #####################
    # Apply chat template
    #####################
    datasets_train = datasets_train.map(
        prepare_sample_text_ab,
        fn_kwargs={
            "tokenizer":tokenizer
        },
        num_proc=data_args.preprocessing_num_workers,
        desc="Applying chat template for train",
    )

    datasets_valid = datasets_valid.map(
        prepare_sample_text_ab,
        fn_kwargs={
            "tokenizer":tokenizer
        },
        num_proc=data_args.preprocessing_num_workers,
        desc="Applying chat template for valid",
    )
    # import pdb; pdb.set_trace()

    # raw_datasets = raw_datasets.map(
    #     apply_chat_template,
    #     fn_kwargs={
    #         "tokenizer": tokenizer,
    #         "task": "sft",
    #         "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
    #     },
    #     num_proc=data_args.preprocessing_num_workers,
    #     remove_columns=column_names,
    #     desc="Applying chat template",
    # )

    ##########################
    # Decontaminate benchmarks
    ##########################

    # num_raw_train_samples = len(datasets_train)
    # num_raw_train_samples = len(raw_datasets["train"])
    # raw_datasets = raw_datasets.filter(decontaminate_humaneval, batched=True, batch_size=10_000, num_proc=1)
    # num_filtered_train_samples = num_raw_train_samples - len(raw_datasets["train"])
    # logger.info(
    #     f"Decontaminated {num_filtered_train_samples} ({num_filtered_train_samples/num_raw_train_samples * 100:.2f}%) samples from the training set."
    # )

    # train_dataset = raw_datasets["train"]
    # eval_dataset = raw_datasets["test"]

    # with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
    #     for index in random.sample(range(len(datasets_train)), 3):
    #         logger.info(f"Sample {index} of the processed training set:\n\n{datasets_train[index]['text']}")

    # with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
    #     for index in random.sample(range(len(raw_datasets["train"])), 3):
    #         logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        model_init_kwargs=model_kwargs,
        args=training_args,
        train_dataset=datasets_train,
        eval_dataset=datasets_valid,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=False,
        peft_config=get_peft_config(model_args),
        dataset_kwargs=training_args.dataset_kwargs,
    )
    # import pdb;pdb.set_trace()
    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(datasets_train)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    # kwargs = {
    #     "finetuned_from": model_args.model_name_or_path,
    #     "dataset": list(data_args.dataset_mixer.keys()),
    #     "dataset_tags": list(data_args.dataset_mixer.keys()),
    #     "tags": ["alignment-handbook"],
    # }
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": [data_args.path_data],
        "dataset_tags": [data_args.name_data]
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     metrics["eval_samples"] = len(eval_dataset)
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    # if training_args.push_to_hub is True:
    #     logger.info("Pushing to hub...")
    #     trainer.push_to_hub(**kwargs)

    # logger.info("*** Training complete ***")


if __name__ == "__main__":
    # wandb.login("ad31d54e6df465eded40d4bec3d41a6761f76740")
    wandb.login(key="ad31d54e6df465eded40d4bec3d41a6761f76740")
    # wandb.init(
    #     entity="PhishingMerge"
    #     # project="PhishingMerge",
    # )
    main()
