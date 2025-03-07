import re
import os
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
import argparse
import pdb
import textwrap
from transformers import AutoTokenizer
from pprint import pprint
import json

# some random words which serves as the splitter
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

def make_prompt(instruct_prompt,code_solution,tokenizer):
    instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
    code_solution = code_solution.split("input()")[0] + "input()"
    
    instruct_prompt = instruct_prompt.split("Problem Description: ")[-1]
    instruct_prompt = f"""\
                    {instruction_prefix}
                    ```
                    {instruct_prompt.strip()}
                    {code_solution}
                    ```
                    """
    if tokenizer:
        instruct_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": instruct_prompt},
            ],
            tokenize=False, add_generation_prompt=True
        ).split(_MAGIC_SPLITTER_)[0]
    
    return instruct_prompt

def process_tests(test_cases):
    test_dict = {
        "inputs": [],
        "outputs": []
    }
    
    for test_unit in test_cases:
        test_dict["inputs"].append(test_unit["input"])
        test_dict["outputs"].append(test_unit["output"])
    
    return test_dict

def make_map_fn(split):
    def process_fn(example, idx):
        
        question = make_prompt(example["prompt"], example["code"],tokenizer) #prompt + example["instruct_prompt"] 
        solution = {
            "answer": example["code"],
            "unit_test": process_tests(example["test_cases"])
            }
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            } #,
            # "extra_info": {
            #     'split': split,
            #     'index': idx,
            #     'libs': example["libs"],
            #     'doc_struct': example["doc_struct"]
                
            # }
        }
        return data
    return process_fn
    

# Define the filtering function
def filter_code(row):
    code = row['code']
    
    # Check for multiple occurrences of "input()" and functions ("def")
    if code.count('input()') == 1 and 'def ' not in code and len(row['prompt']) < 1200:
        return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/export/home/data/codeforce-medium/')
    parser.add_argument('--model', default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", type= str)
    parser.add_argument('--template_path', default="/home/skokane/rl/RL/template.jinja", type= str)
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=	1140)
    parser.add_argument('--num_operands', type=int, default=6)
    parser.add_argument('--max_target', type=int, default=1000)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=5000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'codeforce'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    raw_dataset = load_dataset('evanellis/evanellis_Codeforces-Python-Submissions_correct_with_h_a_k_prob_0.5_with_null_and_rejected_f', split='train')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # update_chat_template(tokenizer, args.template_path)
    
    # Filter the dataset based on the conditions
    raw_dataset = raw_dataset.filter(filter_code)
    print("Filtered Dataset ", len(raw_dataset))
    
    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    
    columns_to_select = ["data_source", "prompt", "ability",  "reward_model"] 
    
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in columns_to_select])
    test_dataset = test_dataset.remove_columns([col for col in test_dataset.column_names if col not in columns_to_select])
    
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    model_name = args.model.split("/")[-1]
    
    train_dataset.to_parquet(os.path.join(local_dir, f'train_{model_name}.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, f'test_{model_name}.parquet'))
    
    
    # columns of interest: 
    # prompt, test_cases, code-(ground truth)
    # print(train_dataset.columns())
    # for t in train_dataset:
    #     pprint(t)
        