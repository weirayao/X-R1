import os
import argparse
import datasets
import json
from datasets import load_dataset
from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/skokane/rl/X-R1/data/deepmath/')
    args = parser.parse_args()

    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # train data
    train_data = load_dataset("zwhe99/DeepMath-103K", split='train')
    def process_fn_train(example, idx):
        prompt_text = tokenizer.apply_chat_template(
                [
                {
                    "role": "system",
                    "content": r"Please reason step by step, and put your final answer within \boxed{}."
                },
                {
                    "role": "user",
                    "content": example["question"]
                },                
            ],
                tokenize=False,
                add_generation_prompt=True,
            )
        data = {
            "data_source": "deepmath",
            "prompt":prompt_text,
            "ability": "deepmath",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(example["final_answer"])
            },
            "extra_info": {
                'split': 'train',
                'index': idx,
                'answer': example["final_answer"],
                "question": example["question"],
                "r1": example["r1_solution_1"]
            },
            
        }
        return data

    # test data
    test_dataset = datasets.load_dataset('zwhe99/MATH', split='math500')
    def process_fn_test(example, idx):
        prompt_text = tokenizer.apply_chat_template(
                [
                {
                    "role": "system",
                    "content": r"Please reason step by step, and put your final answer within \boxed{}."
                },
                {
                    "role": "user",
                    "content": example["problem"]
                },                
            ],
                tokenize=False,
                add_generation_prompt=True,
            )
        data = {
            "data_source": "deepmath",
            "prompt": prompt_text,
            "ability": "deepmath",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(example["expected_answer"]),
            },
            "extra_info": {
                "split": "math500",
                "index": idx,
                "answer": example["expected_answer"],
                "question": example["problem"],
            },
        }
        return data

    train_dataset = train_data.map(function=process_fn_train, with_indices=True)
    
    training_dataset = train_dataset.select(range(80000))
    test_dataset = train_dataset.select(range(80000, len(train_dataset)))
    
    # test_dataset = test_dataset.map(function=process_fn_test, with_indices=True)
    training_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")