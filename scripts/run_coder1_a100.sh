ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=7 src/x_r1/grpo.py \
--config recipes/coder1_config_h200.yaml \
> ./output/coder1_h200.log 2>&1