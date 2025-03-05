from transformers import AutoTokenizer
from datasets import load_dataset
import os

# Choose the tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set the data source name
data_source = "x-r1"
# Set the data directory
data_dir = "/export/home/data/X-R1-7500"
os.makedirs(data_dir, exist_ok=True)

# System prompt for the conversation
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# Format into conversation
def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }
    
dataset = load_dataset("xiaodongguaAIGC/X-R1-7500")
dataset = dataset.map(make_conversation)

for split in dataset:
    dataset[split] = dataset[split].map(lambda x:
        {
            "prompt": tokenizer.apply_chat_template(x["prompt"], tokenize=False, tools=None, add_generation_prompt=True),
            "data_source": data_source
        }
    )
    
    if "messages" in dataset[split].column_names:
        dataset[split] = dataset[split].remove_columns("messages")
    # print the first item of the train dataset
    print(dataset[split][0])
    # save the dataset to parquet
    dataset[split].to_parquet(os.path.join(data_dir, f'{split}_{model_name.split("/")[-1]}.parquet'))