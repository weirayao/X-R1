ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=15 src/x_r1/grpo.py \
--config recipes/gsm8k_rl++_config_a100.yaml \
> ./output/gsm8k_a100.log 2>&1