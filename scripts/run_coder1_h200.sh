#export WANDB_API_KEY=f9b6e9edc8c5538351a133aa20e32cd9e04a547c

# ACCELERATE_LOG_LEVEL=info accelerate launch \
# --config_file recipes/zero3.yaml \
# --num_processes=7 src/x_r1/grpo.py \
# --config recipes/coder1_config_e_less.yaml \
# > ./output/coder1_e_less.log 2>&1


ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=7 src/x_r1/grpo.py \
--config recipes/ccplus_e_less.yaml \
> ./output/ccplus_e_less.log 2>&1