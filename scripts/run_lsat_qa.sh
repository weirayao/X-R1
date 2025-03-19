ACCELERATE_LOG_LEVEL=info accelerate launch \
--config_file recipes/zero3.yaml \
--num_processes=15 src/x_r1/grpo.py \
--config recipes/lsat_qa/recipes/lsat_qa/lsat_qwen_7b_instruct.yaml \
> ./output/lsat_qa.log 2>&1