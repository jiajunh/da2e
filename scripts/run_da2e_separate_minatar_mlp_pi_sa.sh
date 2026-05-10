#!/bin/bash

cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/da2e/cleanrl

# wandb login


# Asterix-v0, Breakout-v0, Freeway-v0, Seaquest-v0, SpaceInvaders-v0, Asterix-v1, Breakout-v1, Freeway-v1, Seaquest-v1, SpaceInvaders-v1


MINATAR_ENV="Breakout-v0"
python cleanrl/da2e_separate_minatar_mlp_pi_sa.py \
    --seed 1 \
    --env_id "MinAtar/$MINATAR_ENV" \
    --total_timesteps 10000000 \
    --wandb_project_name cleanrl_test \
    --learning_rate 3.0e-4 \
    --learning_rate_vf 5.0e-4 \
    --num_envs 512 \
    --num_steps 128 \
    --batch_size 1024 \
    --update_epochs 5 \
    --gamma 0.99 \
    --clip_coef 0.1 \
    --ent_coef 0.02 \
    --max_grad_norm 1.0 \
    --max_grad_norm_vf 1.0 \
    --running_avg_coef 0.05 \
    --torch_deterministic \
    --norm_adv \
    --vf_coef 1.0 \
    --kl_coef 0.0 \
    --v_transformer_layers 2 \
    --pi_dim 512 \
    --v_transformer_dim 256 \
    --v_num_heads 2 \
    --cuda \
    --dropout 0.0 \
    --cnn_feature_dim 128 \
    --weight_decay 0.0 \
    --lr_schedule "cosine" \
    --warmup_frac 0.10 \
    --min_lr_ratio 0.1 \
    --beta1 0.9 \
    --beta2 0.95 \
    --track \

