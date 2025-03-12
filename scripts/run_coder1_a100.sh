ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=15 src/x_r1/grpo.py \
--config recipes/coder1_config_a100.yaml \
> ./output/coder1_a100.log 2>&1