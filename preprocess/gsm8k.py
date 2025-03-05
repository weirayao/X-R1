# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets

from transformers import AutoTokenizer

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

data_source = "gsm8k"
local_dir = "/export/home/data/gsm8k"
os.makedirs(local_dir, exist_ok=True)

dataset = datasets.load_dataset(data_source, "main")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# add a row to each data item that represents a unique id
def make_map_fn(split, tokenizer: AutoTokenizer):
    def process_fn(example, idx):
        question_raw = example.pop("question")

        question = question_raw
        prompt_text = tokenizer.apply_chat_template(
            [   
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": question,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        answer_raw = example.pop("answer")
        solution = extract_solution(answer_raw)
        data = {
            "data_source": data_source,
            "prompt": prompt_text,
            "ground_truth": solution,
        }
        return data

    return process_fn

train_dataset = train_dataset.map(
    function=make_map_fn("train", tokenizer), with_indices=True
)
test_dataset = test_dataset.map(
    function=make_map_fn("test", tokenizer), with_indices=True
)

# print the first item of the train dataset
print(train_dataset[0])

train_dataset.to_parquet(os.path.join(local_dir, f"train_{model_name.split('/')[-1]}.parquet"))
test_dataset.to_parquet(os.path.join(local_dir, f"test_{model_name.split('/')[-1]}.parquet"))