#!/bin/bash

#SBATCH --job-name=run_da2e_minatar_sep_brk
#SBATCH --mem=16G
#SBATCH -t 0-02:00
#SBATCH -p gpu_test
#SBATCH -o /n/netscratch/kdbrantley_lab/Lab/jiajunh/da2e/logs/run_da2e_minatar_sep_brk_%a.out
#SBATCH -e /n/netscratch/kdbrantley_lab/Lab/jiajunh/da2e/logs/run_da2e_minatar_sep_brk_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1


cd /n/netscratch/kdbrantley_lab/Lab/jiajunh/da2e/cleanrl

MINATAR_ENV="Breakout-v0"
python cleanrl/attention_pi_minatar_sa.py \
    --seed 4 \
    --env_id "MinAtar/$MINATAR_ENV" \
    --total_timesteps 10000000 \
    --wandb_project_name "sweep_attention_pi_MinAtar_Breakout-v0" \
    --num_envs 512 \
    --num_steps 128 \
    --weight_decay 0.0 \
    --lr_schedule "cosine" \
    --warmup_frac 0.10 \
    --min_lr_ratio 0.10 \
    --beta1 0.9 \
    --beta2 0.95 \
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
    --cuda \
    --dropout 0.0 \
    --traj_minibatch_size 64 \
    --update_epochs 5 \
    --learning_rate 0.0005 \
    --transformer_layers 2 \
    --transformer_dim 128 \
    --value_dim 128 \
    --num_heads 2 \
    --cnn_feature_dim 128 \
    --track \
