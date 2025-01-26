# torchrun --rdzv-endpoint=localhost:6000 --nproc_per_node=8 scripts/finetune_pt.py \
# --config scripts/configs/finetune_config_libero_black_bowl.py --name libero_black_bowl_100 \
# --config.pretrained_path=hf://rail-berkeley/octo-base-1.5 --config.save_dir experiments/libero_black_bowl_100 \
# --config.split='train[:100%]'

torchrun --rdzv-endpoint=localhost:6000 --nproc_per_node=8 scripts/finetune_pt.py \
--config scripts/configs/finetune_config_libero_black_bowl.py --name libero_black_bowl_80 \
--config.pretrained_path=hf://rail-berkeley/octo-base-1.5 --config.save_dir experiments/libero_black_bowl_80 \
--config.split='train[:80%]' --config.num_steps=4000

torchrun --rdzv-endpoint=localhost:6000 --nproc_per_node=8 scripts/finetune_pt.py \
--config scripts/configs/finetune_config_libero_black_bowl.py --name libero_black_bowl_60 \
--config.pretrained_path=hf://rail-berkeley/octo-base-1.5 --config.save_dir experiments/libero_black_bowl_60 \
--config.split='train[:60%]' --config.num_steps=3000