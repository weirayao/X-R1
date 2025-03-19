from transformers import AutoTokenizer
from datasets import load_dataset
import os

# Choose the tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Set the data source name
data_source = "lighteval/lsat_qa"
dataset = load_dataset(data_source, "all", trust_remote_code=True)
train_dataset = dataset["train"]
test_dataset = dataset["test"]
# Set the data directory
local_dir = "/export/home/data/lsat_qa"
os.makedirs(local_dir, exist_ok=True)

# System prompt for the conversation
SYSTEM_PROMPT = (
    "You should answer the logic question based on the given context.\
    First think about your reasoning process in <think> </think> tags.\
    Then output one of the references as the correct answer after \"####\"."
)

# add a row to each data item that represents a unique id
def make_map_fn(split, tokenizer: AutoTokenizer):
    def process_fn(example, idx):
        context = "## Passage:{}\n## Question:{}\n## References:{}\n".format(example["passage"], example["question"], example["references"])
        prompt_text = tokenizer.apply_chat_template(
            [   
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": context,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        gold_index = example["gold_index"]
        solution = example["references"][gold_index]
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