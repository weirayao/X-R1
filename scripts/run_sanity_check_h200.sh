export WANDB_PROJECT=x_r1_sanity_check

ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=7 src/x_r1/grpo.py \
--config recipes/X_R1_zero_7B_config_h200.yaml \
> ./output/x_r1_7B_sampling_h200.log 2>&1