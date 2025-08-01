CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=3350 \
    train_fastmoe.py \
    --config_env configs/env.yml \
    --config_exp configs/nyud/vit_moe/pup_moe_vit_small_depth.yml \
    --moe_gate_type "noisy_vmoe" \
    --moe_experts 16 \
    --moe_top_k 4 \
    --pos_emb_from_pretrained True \
    --backbone_random_init False \
    --vmoe_noisy_std 0 \
    --gate_task_specific_dim -1 \
    --save_dir /home/output_dir \
    --task_one_hot False \
    --multi_gate False \
    --moe_mlp_ratio 1