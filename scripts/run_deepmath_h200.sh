

# ACCELERATE_LOG_LEVEL=info accelerate launch \
# --config_file recipes/zero3.yaml \
# --num_processes=7 src/x_r1/grpo.py \
# --config recipes/coder1_config_e_less.yaml \
# > ./output/coder1_e_less.log 2>&1


ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=7 src/x_r1/grpo.py \
--config recipes/deepmath_config_h200.yaml \
> ./output/deepmath_h200.log 2>&1
