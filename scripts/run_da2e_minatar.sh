#!/bin/bash

cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/da2e/cleanrl

# wandb login


# Asterix-v0, Breakout-v0, Freeway-v0, Seaquest-v0, SpaceInvaders-v0, Asterix-v1, Breakout-v1, Freeway-v1, Seaquest-v1, SpaceInvaders-v1

# MINATAR_ENV="Asterix-v0"
MINATAR_ENV="Breakout-v0"
# MINATAR_ENV="Seaquest-v0"
# MINATAR_ENV="SpaceInvaders-v0"

# return_to_go = {
#     "Asterix-v0": 160,
#     "Breakout-v0": 120,
#     "Seaquest-v0": 250,
#     "SpaceInvaders-v0": 600,
# }


python cleanrl/da2e_minatar.py \
    --seed 1 \
    --env_id "MinAtar/$MINATAR_ENV" \
    --total_timesteps 10000000 \
    --wandb_project_name cleanrl_$MINATAR_ENV \
    --learning_rate 6.0e-4 \
    --num_envs 1024 \
    --num_steps 128 \
    --batch_size 2048 \
    --update_epochs 6 \
    --gamma 0.99 \
    --clip_coef 0.2 \
    --ent_coef 0.01 \
    --max_grad_norm 0.5 \
    --running_avg_coef 0.05 \
    --torch_deterministic \
    --clip_vloss \
    --anneal_lr \
    --norm_adv \
    --vf_coef 0.5 \
    --kl_coef 0.0 \
    --anneal_clip_coef \
    --transformer_layers 1 \
    --transformer_dim 128 \
    --num_heads 1 \
    --return_to_go 150 \
    --rtg_scale 150 \
    --cuda \
    --dropout 0.1 \
    --cnn_feature_dim 128 \
    --track \

