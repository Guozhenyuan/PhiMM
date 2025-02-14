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
import warnings
import sys
import gc

import datasets
import torch
import transformers
from transformers import AutoModelForCausalLM, PreTrainedTokenizer, set_seed
from typing import Any, Dict, List, NewType, Optional, Tuple
from dataclasses import dataclass, field
from multiprocessing import cpu_count

from transformers import Gemma2ForCausalLM
from typing import List, Optional, Tuple, Union

from transformers import DataCollatorWithPadding,DataCollatorForLanguageModeling
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from accelerate import Accelerator


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

def is_accelerator_used(model):
    return isinstance(model, Accelerator)

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

    elif args.proc_data == 'sft-b-rif':
        dialogue_train = [data['ct'] for data in data['train']]
        dialogue_valid = [data['ct'] for data in data['test']]    
        dataset_train = datasets.Dataset.from_dict({'dialogue':dialogue_train})
        dataset_valid = datasets.Dataset.from_dict({'dialogue':dialogue_valid})

    elif args.proc_data == 'sft-b-phishing-w':
        dialogue_train = [data['ct'] for data in data['train']]
        dialogue_valid = [data['ct'] for data in data['test']]    
        dataset_train = [data['dataset'] for data in data['train']]
        dataset_valid = [data['dataset'] for data in data['test']]
        dataset_train = datasets.Dataset.from_dict({'dialogue':dialogue_train,'dataset':dataset_train})
        dataset_valid = datasets.Dataset.from_dict({'dialogue':dialogue_valid,'dataset':dataset_valid})

    else:
        pass

    return dataset_train,dataset_valid

def prepare_sample_text_ab(example, tokenizer: PreTrainedTokenizer):
    # print(example.keys())
    dialogue = example['dialogue']
    example['text'] = tokenizer.apply_chat_template(dialogue,tokenize=False)
    # import pdb;pdb.set_trace()
    # example['dataset'] = 1
    # example['dialogue'] =1 
    # example['labels'] = 'A'
    return example

# class CustomDataCollator(DataCollatorWithPadding):
#     def __call__(self, features):
#         # 使用父类的 __call__ 方法来处理默认的 padding 和 tensor 转换
#         batch = super().__call__(features)
#         # 保留 dataset 字段
#         import pdb;pdb.set_trace()
#         batch['dataset'] = [f['dataset'] for f in features]
#         return batch





class CustomDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False, mlm_probability=0.15):
        # 继承父类的初始化
        super().__init__(tokenizer=tokenizer, mlm=mlm, mlm_probability=mlm_probability)
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        # import pdb;pdb.set_trace()
        batch['dataset'] = [f['dataset'] for f in examples]
        return batch

@dataclass
class NewDataArguments(DataArguments):
    path_data: Optional[str] = field(default=None, metadata={"help": "The path of used data"})
    name_data: Optional[str] = field(default=None, metadata={"help": "The name of used data"})
    proc_data: Optional[str] = field(default=None, metadata={"help": "How to processing the data"})

@dataclass
class NewSFTConfig(SFTConfig):
    alpha_pi: Optional[float] = field(default=0.3, metadata={"help": "The weight of rif"})


class WeightedLossTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        self._signature_columns += list(['dataset'])  # type: ignore

    def _prepare_non_packed_dataloader(
        self,
        tokenizer,
        dataset,
        dataset_text_field,
        max_seq_length,
        formatting_func=None,
        add_special_tokens=True,
        remove_unused_columns=True,
    ):
        use_formatting_func = formatting_func is not None and dataset_text_field is None
        self._dataset_sanity_checked = False
        
        # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
        def tokenize(element):
            # import pdb;pdb.set_trace()
            outputs = tokenizer(
                element[dataset_text_field] if not use_formatting_func else formatting_func(element),
                add_special_tokens=add_special_tokens,
                truncation=True,
                padding=False,
                max_length=max_seq_length,
                return_overflowing_tokens=False,
                return_length=False,
            )

            if use_formatting_func and not self._dataset_sanity_checked:
                if not isinstance(formatting_func(element), list):
                    raise ValueError(
                        "The `formatting_func` should return a list of processed strings since it can lead to silent bugs."
                    )
                else:
                    self._dataset_sanity_checked = True
            # import pdb;pdb.set_trace()
            return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"], "dataset":element['dataset']}

        signature_columns = ["input_ids", "labels", "attention_mask", "dataset"]

        extra_columns = list(set(dataset.column_names) - set(signature_columns))

        if not remove_unused_columns and len(extra_columns) > 0:
            warnings.warn(
                "You passed `remove_unused_columns=False` on a non-packed dataset. This might create some issues with the default collator and yield to errors. If you want to "
                f"inspect dataset other columns (in this case {extra_columns}), you can subclass `DataCollatorForLanguageModeling` in case you used the default collator and create your own data collator in order to inspect the unused dataset columns."
            )
        # import pdb; pdb.set_trace()
        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names if remove_unused_columns else None,
            num_proc=self.dataset_num_proc,
            batch_size=self.dataset_batch_size,
        )
        # import pdb; pdb.set_trace()
        return tokenized_dataset

    def compute_loss(self, model, inputs, return_outputs=False):
        # 根据输入数据区分是数据集A还是数据集B

        dataset = inputs.pop("dataset")
        # import pdb; pdb.set_trace()
        outputs = model(**inputs)
        logits = outputs['logits'] 
        labels = inputs['labels']

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()

        a_shift_logits = shift_logits[dataset==0]
        b_shift_logits = shift_logits[dataset==1]

        a_shift_labels = shift_labels[dataset==0]
        b_shift_labels = shift_labels[dataset==1]

        # print(a_shift_logits.shape[0] == 0 or b_shift_logits.shape[0] == 0,self.args.alpha_pi)


            # loss_fct = CrossEntropyLoss()
            # shift_logits = shift_logits.view(-1, self.config.vocab_size)
            # shift_labels = shift_labels.view(-1)
            # # Enable model parallelism
            # shift_labels = shift_labels.to(shift_logits.device)
            # loss = loss_fct(shift_logits, shift_labels)
        if is_accelerator_used(self.model):
            vc = self.model.model.vocab_size
        else:
            vc = self.model.vocab_size

        # unwrapped_model = self.accelerator.unwrap_model(self.model)
        

        if a_shift_logits.shape[0] == 0 or b_shift_logits.shape[0] == 0:
            shift_logits = shift_logits.view(-1, vc)
            # self.config.vocab_size
            shift_labels = shift_labels.view(-1)
            # shift_labels = shift_labels.to(shift_logits.device)
            if a_shift_logits.shape[0]==0: # only math
                loss = (1-self.args.alpha_pi)*loss_fct(shift_logits, shift_labels)
            else:
                loss = self.args.alpha_pi*loss_fct(shift_logits, shift_labels)
            # print('one loss',loss.item())
        else:
            a_shift_logits = a_shift_logits.view(-1, vc)
            b_shift_logits = b_shift_logits.view(-1, vc)

            a_shift_labels = a_shift_labels.view(-1)
            b_shift_labels = b_shift_labels.view(-1)

            # a_shift_labels = a_shift_labels.to(a_shift_logits.device)
            # b_shift_labels = b_shift_labels.to(b_shift_logits.device)
            loss_1 = loss_fct(a_shift_logits,a_shift_labels)
            loss_2 = loss_fct(b_shift_logits,b_shift_labels)
            loss = self.args.alpha_pi*loss_1 + (1-self.args.alpha_pi)*loss_2
            print('loss_1:{} loss_2:{}'.format(loss_1.item(),loss_2.item()))
        
        del dataset
        torch.cuda.empty_cache()
        # gc.collect()
        # shift_logits = shift_logits.view(-1, model.config.vocab_size)
        # shift_labels = shift_labels.view(-1)
        # # Enable model parallelism
        # shift_labels = shift_labels.to(shift_logits.device)
        # import pdb; pdb.set_trace()
        # loss = self.args.alpha_pi*loss_fct(shift_logits[dataset==0]) + (1-self.args.alpha_pi)*loss_fct(shift_logits[dataset==1])
        # # loss = loss_fct(shift_logits, shift_labels)


        # import pdb;pdb.set_trace()
        # outputs = model(**inputs)
        # dataset = inputs.pop("dataset")
        # import pdb; pdb.set_trace()
        # if inputs.get("dataset") == "A":
        #     outputs = model(**inputs)
        #     loss_a = outputs.loss
        #     loss = self.weight_a * loss_a
        # elif inputs.get("dataset") == "B":
        #     outputs = model(**inputs)
        #     loss_b = outputs.loss
        #     loss = self.weight_b * loss_b
        # else:
        #     raise ValueError("Unknown dataset in inputs")
        
        return (loss, outputs) if return_outputs else loss

def main():
    parser = H4ArgumentParser((ModelArguments, NewDataArguments, NewSFTConfig))
    model_args, data_args, training_args = parser.parse()

    # training_args.remove_unused_columns=False

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
    # trainer = SFTTrainer(
    #     model=model,
    #     model_init_kwargs=model_kwargs,
    #     args=training_args,
    #     train_dataset=datasets_train,
    #     eval_dataset=datasets_valid,
    #     dataset_text_field="text",
    #     max_seq_length=training_args.max_seq_length,
    #     tokenizer=tokenizer,
    #     packing=False,
    #     peft_config=get_peft_config(model_args),
    #     dataset_kwargs=training_args.dataset_kwargs,
    # )

    # error debug, cache_implementation = False
    # print('dd',training_args)
    # print('dd2',model_kwargs)
    # print('model_args',model_args)
    # if "gemma" in model_args.model_name_or_path:
    #     training_args.cache_implementation = False if training_args.gradient_checkpointing else True

    trainer = WeightedLossTrainer(
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
        # data_collator=data_collator,
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
    wandb.login(key="ad31d54e6df465eded40d4bec3d41a6761f76740")
    main()
