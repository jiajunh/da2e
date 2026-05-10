#!/bin/bash

cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/da2e/cleanrl

# wandb login


MINATAR_ENV="Asterix-v0"
# MINATAR_ENV="Breakout-v0"
# MINATAR_ENV="Seaquest-v0"
# MINATAR_ENV="SpaceInvaders-v0"


python cleanrl/attention_ppo_minatar_sa.py \
    --seed 1 \
    --env_id "MinAtar/$MINATAR_ENV" \
    --total_timesteps 10000000 \
    --wandb_project_name cleanrl_$MINATAR_ENV \
    --learning_rate 6.0e-4 \
    --num_envs 1024 \
    --num_steps 128 \
    --traj_minibatch_size 64 \
    --update_epochs 5 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_coef 0.1 \
    --ent_coef 0.01 \
    --max_grad_norm 1.0 \
    --running_avg_coef 0.05 \
    --torch_deterministic \
    --norm_adv \
    --vf_coef 1.0 \
    --kl_coef 0.0 \
    --transformer_layers 1 \
    --transformer_dim 128 \
    --num_heads 2 \
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

