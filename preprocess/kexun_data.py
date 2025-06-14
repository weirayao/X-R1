"""
Preprocess LeetCode problems (newfacade/LeetCodeDataset) to parquet format.
"""

import os
import json
import sys
import os.path as osp
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from datasets import load_dataset, concatenate_datasets, Dataset
from rich.rule import Rule
import rich

# from verl.utils.hdfs_io import copy, makedirs
from src.x_r1.reward_score.coder1 import code_exec, remote_check_stdio, _ERROR_MSG_PREFIX #, CODER1_EXEC
from tqdm import tqdm
import io

# Import async version if we're using remote execution
# if CODER1_EXEC == "remote":
#     from verl.utils.reward_score.coder1 import remote_check_stdio_async

N_TESTSET_PER_DATASET = 512  # per dataset
_EMPTY_RETURN_ = {
    "data_source": None,
    "prompt": None,
    "ability": None,
    "reward_model": None,
    "extra_info": None,
}


def minimize_stdio(inputs, outputs, max_n_tests=8):
    stdin_list = []
    stdout_list = []
    for stdin, stdout in zip(inputs, outputs):
        if isinstance(stdin, list):
            stdin = "\n".join(stdin)
        if isinstance(stdout, list):
            stdout = "\n".join(stdout)
        if sys.getsizeof(stdin) > 4 * 1024:
            continue
        stdout.replace("\r\n", "\n")
        stdin_list.append(stdin)
        stdout_list.append(stdout)

    zipped = sorted(zip(stdin_list, stdout_list), key=lambda x: sys.getsizeof(x[0]))

    if not zipped:
        print("No tests found!")
        return [], []

    sorted_stdin, sorted_stdout = zip(*zipped)
    return list(sorted_stdin[:max_n_tests]), list(sorted_stdout[:max_n_tests])


SYSTEM_PROMPT = """You are a helpful programming assistant. \
The user will ask you a question and you as the assistant solve it. \
The assistant first thinks how to solve the task through reasoning and then provides the user with the final answer. \
The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively."""

import pickle
import zlib
import base64

def decode_test_cases(test_cases):
    if not test_cases:
        return None
    
    return json.loads(pickle.loads(
        zlib.decompress(
            base64.b64decode(test_cases.encode("utf-8"))  # type: ignore
        )
    ))
    

def livecodebench():
    rich.print(Rule("Loading LiveCodeBench..."))
    lcb_dataset = load_dataset("livecodebench/code_generation_lite", version_tag="v4_v5", trust_remote_code=True)['test']
    lcb_dataset = lcb_dataset.filter(lambda x: x["platform"] == "atcoder")
    
    def process_fn(example, idx):
        prompt = ("Solve the programming task below in a python markdown code block. "
            "Each time, given inputs through STDIN (like those in the 'Input' section), the program "
            "produces outputs through STDOUT (like those in the 'Output' section)."
        ) + (f"\n\n{example['question_title'].strip()}\n\n" if example['question_title'] else "") \
            + (f"{example['question_content'].strip()}")
            
        public_test_cases = json.loads(example["public_test_cases"])
        private_test_cases = decode_test_cases(example["private_test_cases"])
            
        return {
            "data_source": "coder1",
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "ability": "coding",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps({
                    "inputs": [case["input"] for case in public_test_cases + private_test_cases],
                    "outputs": [case["output"] for case in public_test_cases + private_test_cases],
                }),
            },
            "extra_info": {
                "split": "test",
                "index": idx,
                "prompt": prompt,
                "dataset": "livecodebench",
            },
        }
    test_dataset = lcb_dataset.map(function=process_fn, with_indices=True, num_proc=64)
    test_dataset = test_dataset.remove_columns(lcb_dataset.column_names).filter(lambda x: x['prompt'] != None)
    return test_dataset

def process_test_case(data):
    return data["pid"], decode_test_cases(data["test_cases"])

def codecontests_plus():
    test_dataset = livecodebench()
    
    rich.print(Rule("Loading codecontests_plus..."))
    
    with open("./selected_5k_pids_for_rl.json", "r") as f:
        selected_pids = set(json.load(f))
    test_cases = load_dataset("akioi/ccp_test_cases_5k")["train"]
    problems = load_dataset("akioi/code_contests_plus")["train"]
    selected_dataset = problems.filter(lambda x: x["pid"] in selected_pids)
    
    assert len(selected_dataset) == len(selected_pids), f"Len of selected dataset {len(selected_dataset)} != len of selected pids {len(selected_pids)}"
    assert len(selected_dataset) == len(test_cases), f"Len of selected dataset {len(selected_dataset)} != len of test cases {len(test_cases)}"
    
    # Parallelize test_data dictionary generation
    from multiprocessing import Pool
    
    test_data = {}
    with Pool(processes=64) as pool:
        for pid, decoded in tqdm(pool.imap_unordered(process_test_case, test_cases), 
                               total=len(test_cases)):
            test_data[pid] = decoded
    
    def process_fn(example, idx):
        if example["pid"] not in test_data or test_data[example["pid"]] is None:
            return _EMPTY_RETURN_
        if example['question_content'] is None:
            return _EMPTY_RETURN_
        
        prompt = ("Solve the programming task below in a python markdown code block. "
            "Each time, given inputs through STDIN (like those in the 'Input' section), the program "
            "produces outputs through STDOUT (like those in the 'Output' section)."
        ) + (f"\n\n{example['question_title'].strip()}\n\n" if example['question_title'] else "") \
            + (f"{example['question_content'].strip()}")
        
        ground_truth = json.dumps({
            "inputs": [case["input"] for case in test_data[example["pid"]]],
            "outputs": [case["output"] for case in test_data[example["pid"]]],
        })
        
        if len(ground_truth) > 2 * 1024 * 1024 * 1024: # too big, sample 50% of the cases by skipping once for each case
            ground_truth = json.dumps({
                "inputs": [case["input"] for case in test_data[example["pid"]][::2]],
                "outputs": [case["output"] for case in test_data[example["pid"]][::2]],
            })
            print(f"Warning: Large ground truth {example['pid']} with size {len(ground_truth)/1024/1024:.2f}MB")
        
        return {
            "data_source": "coder1",
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "ability": "coding",
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "split": "train",
                "index": idx,
                "prompt": prompt,
                "dataset": "codecontests_plus",
            },
        }
        
    train_dataset = []
    for idx, example in tqdm(enumerate(selected_dataset), total=len(selected_dataset)):
        processed = process_fn(example, idx)
        # Check byte size of processed example if saved as parquet
        if processed is not None and processed['prompt'] is not None:
            train_dataset.append(processed)
    train_dataset = Dataset.from_list(train_dataset)
    
    return train_dataset, test_dataset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="/home/skokane/data/ccplus_python/")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    root_dir = args.root_dir
    hdfs_dir = args.hdfs_dir

    train_datasets = []
    test_datasets = []

    dataset_makes = [codecontests_plus]
    names = "-".join([make.__name__ for make in dataset_makes])

    for train, test in [make() for make in dataset_makes]:
        train_datasets.append(train)
        test_datasets.append(test)

    train_dataset = concatenate_datasets(train_datasets)
    test_dataset = concatenate_datasets(test_datasets)

    rich.print(Rule("Saving the final dataset"))
    
    local_dir = os.path.join(root_dir, f"code-r1-{round(len(train_dataset) / 1000)}k-{names}")
    rich.print(f"[bold green]Saving to {local_dir}...")
    
    # Save train dataset in shards
    num_training_shards = 200
    num_test_shards = 10
    for i in tqdm(range(num_training_shards), desc="Saving training shards"):
        train_dataset.shard(num_shards=num_training_shards, index=i).to_parquet(
            os.path.join(local_dir, f"train_shard{i}.parquet")
        )
    for i in tqdm(range(num_test_shards), desc="Saving test shards"):
        test_dataset.shard(num_shards=num_test_shards, index=i).to_parquet(
            os.path.join(local_dir, f"test_shard{i}.parquet")
        )

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=root_dir, dst=hdfs_dir)