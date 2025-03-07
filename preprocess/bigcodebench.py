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

# some random words which serves as the splitter
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

def make_prompt(instruct_prompt,tokenizer):
    instruction_prefix = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    response_prefix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
    instruct_prompt = f"""\
                    {instruction_prefix}
                    ```
                    {instruct_prompt.strip()}
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

def update_chat_template(tokenizer, template_path: str):
    """
    Load a Jinja template from file and set it as the tokenizer's chat template.
    """
    with open(template_path, "r", encoding="utf-8") as f:
        template_string = f.read()
    tokenizer.chat_template = template_string

# Define the filtering function
def filter_code(row):
    prompt = row['instruct_prompt']
    
    # Check for multiple occurrences of "input()" and functions ("def")
    if len(prompt) < 1200:
        return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/export/home/data/bigcodebench/')
    parser.add_argument('--model', default="Qwen/Qwen2.5-7B-Instruct", type= str)
    parser.add_argument('--template_path', default="/home/skokane/rl/RL/template.jinja", type= str)
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=	1140)
    parser.add_argument('--num_operands', type=int, default=6)
    parser.add_argument('--max_target', type=int, default=1000)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=1000)
    parser.add_argument('--test_size', type=int, default=80)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'bigcodebench'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    raw_dataset = load_dataset('bigcode/bigcodebench', split='v0.1.0_hf')
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # update_chat_template(tokenizer, args.template_path)
    
    raw_dataset = raw_dataset.filter(filter_code)
    print(len(raw_dataset))
    
    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
    
    prompt = """
    Please provide a self-contained Python script that solves the following problem in a markdown code block, for example in:
    '''Python
    
    code here
    
    '''
    
    """

    def make_map_fn(split):
        def process_fn(example, idx):
            
            question = make_prompt(example["instruct_prompt"], tokenizer) #prompt + example["instruct_prompt"] 
            solution = {
                "answer": example["canonical_solution"],
                "unit_test": example["test"],
                "code_prompt": example["code_prompt"] 
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
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'libs': example["libs"],
                    'doc_struct': example["doc_struct"]
                    
                }
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    os.makedirs(local_dir, exist_ok=True)
    
    model_name = args.model.split("/")[-1]
    train_dataset.to_parquet(os.path.join(local_dir, f'train_{model_name}.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, f'test_{model_name}.parquet'))
