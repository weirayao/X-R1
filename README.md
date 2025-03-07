# xLR: Distributed Large Reasoning System as a Scalable Path Towards AGI

We develop our training framework for xLR model series based on a fork from [X-R1](https://github.com/dhcode-cpp/X-R1/tree/main/src/x_r1). The library supports extremely fast and memory efficient GRPO training with full model / LoRA and Zero3. The training logs are saved to wandb. This framework is Huggingface friendly.

## Installation

### conda & pip

required: cuda >= 12.4

```bash
conda create -n xr1 python=3.11
conda activate xr1
```

and

```bash
pip install -r requirements.txt
pip install flash-attn
```

Make sure to create a directory for output logs and model checkpoints:

```bash
mkdir output
```

## Get Started

To quickly start GRPOtraining for sanity check, you can use the following steps:

1. Preprocess the dataset. We provide a script for gsm8k dataset. In this step, we convert the dataset by adding a column for `prompt` to store the generation prompt strings after applying the chat template from tokenizer. One can also write the prompt template by themselves. The `prompt` column is necessary for all datasets.

```bash
python preprocess/gsm8k.py
```

2. Write the training recipe. We provide a recipe for gsm8k dataset in `recipes/X_R1_zero_7B_config_a100.yaml`. One can also write the recipe by themselves. The recipe is a yaml file that specifies the model, dataset, training parameters, etc. Note that one should also specify the `wandb_project`, `run_name`, and `output_dir` in the recipe for each new experiment. Make the `run_name` unique for each experiment with major experiment settings(e.g, `Qwen2.5-7B-Instruct_lr_3e-6_bs_240_n_15`), so different experiments can be compared in wandb. 


3. Run the training script and you will see the reward curve on wandb bump up fast to 95% in a few steps.

```bash
bash ./scripts/run_sanity_check_a100.sh
```

## How to add new dataset and training your own model

We follow a 3-step process detailed below:

1. Preprocess the dataset into train / test parquet files. One should write a `{dataset_name}.py` file in `preprocess` folder. This file should achieve the following functions	:

- Load the dataset from the original format (e.g., json, csv, etc.)
- Create a new column for `prompt` to store the generation prompt text strings after applying the chat template from tokenizer of a base model. For example, for gsm8k dataset, one prompt item is:

```
<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think>. Output the final answer after "####".<|im_end|>\n<|im_start|>user\nNatalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<|im_end|>\n<|im_start|>assistant\n
```
- Add a new column `data_source` to store the original data source name. This column is used to choose the reward function for this sample during training and is necessary for all datasets.

- Add a column for calculating the reward score for each sample. The column name is customizable. For example, for gsm8k dataset, one can add a column `ground_truth` to store the ground truth answer for this prompt. For code related dataset, one may add, for example, unit tests to the column. Keep in mind that the created column name will be passed to the reward function automatically. For example, for gsm8k dataset, the reward function is:

```python
def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
	...
```
- Save the train / test parquet as separate files to the `data/parquet` folder

Make sure you use the same argument names as the column names in the dataset.

2. Add a reward function for calculating the accuracy score of the generated solution. One should write a `{dataset_name}.py` file in `src/x_r1/reward_score` folder. Then, one should import the reward function in `src/x_r1/reward_manager.py` after the existing reward functions in `elif` statements. One should make the input arguments of the reward function consistent with the provided variables, and the column names in the dataset.

3. Add a training recipe in `recipes` folder. Make sure the number of processes is set correctly (it should be the number of GPUs - 1). Follow the following rules for setting `per_device_train_batch_size` and `num_generations`:

**How to setting correct batch_Size and num_generations**

we have 4gpu(1 vLLM + 3 training), setting config is:

```yaml
per_device_train_batch_size: 1
num_generations: 4
```

running with `--num_processes=3`: 

```text
ValueError: The global train batch size (3 x 1) must be evenly divisible by the number of generations per prompt (4). Given the current train batch size, the valid values for the number of generations are: [3].
```


( `per_device_train_batch_size` * `num_processes` ) % `num_generations` == 0


we should set

```yaml
# example 1
num_processes: 3
per_device_train_batch_size: 1
num_generations: 3
# 1 * 3 % 3 = 0

# example 2
num_processes: 3
per_device_train_batch_size: 4
num_generations: 6
# 4 * 3 % 6 = 0
```

if your have 8GPU(1vllm + 7training)

```yaml
num_processes: 7
per_device_train_batch_size: 4
num_generations: 14
# 4 * 7 % 14 = 0
```

4. Create a new training script for your dataset and model, and add it to the `scripts` folder. Do not overwrite the existing scripts.

```bash
bash ./scripts/{your_script_name}.sh
```

## Acknowledge

[X-R1](https://github.com/dhcode-cpp/X-R1/tree/main/src/x_r1), [Open-R1](https://github.com/huggingface/open-r1), [TRL](https://github.com/huggingface/trl), [vLLM](https://github.com/vllm-project/vllm)
