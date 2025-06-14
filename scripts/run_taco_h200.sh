#export WANDB_API_KEY=f9b6e9edc8c5538351a133aa20e32cd9e04a547c

ACCELERATE_LOG_LEVEL=info accelerate launch \
--main_process_port 29501 \
--config_file recipes/zero3.yaml \
--num_processes=7 src/x_r1/grpo.py \
--config recipes/taco_config_e_less.yaml \
> ./output/taco_e_less.log 2>&1