import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import argparse
import random
import time
import math
from collections import deque

import tqdm
import envpool
import gym
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from tensordict import from_module
from tensordict import TensorDict
from torch.distributions.categorical import Categorical, Distribution

Distribution.set_default_validate_args(False)
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment args
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__)[:-len(".py")])
    parser.add_argument("--project_name", type=str, default="ppo_attention_pi_atari")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch_deterministic", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cudagraphs", action="store_true")

    # Algorithm args
    parser.add_argument("--env_id", type=str, default="Breakout-v5")
    parser.add_argument("--total_timesteps", type=int, default=10000000)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--learning_rate_vf", type=float, default=2.5e-4)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--anneal_lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--norm_adv", action="store_true")
    parser.add_argument("--clip_coef", type=float, default=0.1)
    parser.add_argument("--clip_vloss", action="store_true")
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--max_grad_norm_vf", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=None)

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_iterations", type=int, default=0)

    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--avg_returns_last_k", type=int, default=20)

    parser.add_argument("--transformer_layers", type=int, default=1)
    parser.add_argument("--transformer_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--max_ep_len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cnn_feature_dim", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=0.00)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--cnn_feature_dim_v", type=int, default=128)
    parser.add_argument("--v_dim", type=int, default=512)

    parser.add_argument("--lr_schedule", type=str, default="linear", choices=["constant", "linear", "cosine"])
    parser.add_argument("--warmup_frac", type=float, default=0.10)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    return parser.parse_args()


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


def pack_kvs_vectorized(kv_buf, kv_len, idx):
    Tpast_max = kv_len[idx].max().item()
    B = idx.shape[0]
    past_mask = torch.arange(Tpast_max, device=idx.device)[None, :] < kv_len[idx, None]
    past_kvs = []
    for l in range(kv_buf.shape[1]):
        k = kv_buf[idx, l, 0, :, :Tpast_max, :]
        v = kv_buf[idx, l, 1, :, :Tpast_max, :]
        past_kvs.append((k, v))
    return past_kvs, past_mask, Tpast_max


def unpack_kvs_vectorized(present_kvs, idx, kv_buf, kv_len):
    S = kv_buf.shape[4]
    T_total = present_kvs[0][0].shape[2]
    T_write = min(T_total, S)
    T_src_start = T_total - T_write
    for l, (k, v) in enumerate(present_kvs):
        kv_buf[idx, l, 0, :, :T_write, :] = k[:, :, T_src_start:, :]
        kv_buf[idx, l, 1, :, :T_write, :] = v[:, :, T_src_start:, :]
    kv_len[idx] = T_write


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MultiHeadAttentionDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, device=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        assert self.head_size * num_heads == embed_dim

        self.values = nn.Linear(self.head_size, self.head_size, bias=False, device=device)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False, device=device)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False, device=device)
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_dim, device=device)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None, past_kv=None):
        B, T_new, D = x.shape
        q = x.reshape(B, T_new, self.num_heads, self.head_size)
        k = x.reshape(B, T_new, self.num_heads, self.head_size)
        v = x.reshape(B, T_new, self.num_heads, self.head_size)
        q = self.queries(q).transpose(1, 2)
        k = self.keys(k).transpose(1, 2)
        v = self.values(v).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        T_total = k.shape[2]
        T_past  = T_total - T_new

        rows = torch.arange(T_new, device=x.device).unsqueeze(1)
        cols = torch.arange(T_total, device=x.device).unsqueeze(0)
        causal = (cols <= rows + T_past)

        attn_mask = torch.zeros(B, 1, T_new, T_total, device=x.device, dtype=x.dtype)
        attn_mask.masked_fill_(~causal[None, None], float("-inf"))
        if pad_mask is not None:
            attn_mask.masked_fill_(~pad_mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T_new, self.embed_dim)
        out = self.resid_dropout(self.fc_out(out))
        return out, None, (k, v)


class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, device=None):
        super().__init__()
        self.attention = MultiHeadAttentionDecoder(dim, num_heads, dropout, device=device)
        self.input_layernorm = nn.LayerNorm(dim, device=device)
        self.post_attention_layernorm = nn.LayerNorm(dim, device=device)
        self.fc_projection = nn.Sequential(nn.Linear(dim, dim, device=device), nn.ReLU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask=None, past_kv=None):
        hidden_states = self.input_layernorm(x)
        attn_out, _, present_kv = self.attention(hidden_states, pad_mask=pad_mask, past_kv=past_kv)
        hidden_states = self.dropout(attn_out) + x
        x_ = self.post_attention_layernorm(hidden_states)
        ff  = self.fc_projection(x_)
        out = ff + hidden_states
        return out, present_kv


class Transformer(nn.Module):
    def __init__(self, num_layers, dim, num_heads, max_len, dropout=0.0, device=None):
        super().__init__()
        self.max_len = max_len
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [DecoderLayer(dim, num_heads, dropout, device=device) for _ in range(num_layers)]
        )

    def forward(self, x, pad_mask=None, past_kvs=None):
        if past_kvs is None:
            past_kvs = [None] * self.num_layers
        present_kvs = []
        for i, layer in enumerate(self.layers):
            x, present_kv = layer(x, pad_mask=pad_mask, past_kv=past_kvs[i])
            present_kvs.append(present_kv)
        return x, present_kvs


class Agent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        device = args.device
        n_actions = envs.single_action_space.n
        self.transformer_dim = args.transformer_dim

        # ── Policy network (transformer-based) ───────────────────────────
        self.embed_state = nn.Sequential(
            layer_init(nn.Conv2d(4, 64, 8, stride=4, device=device)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 4, stride=2, device=device)),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, args.cnn_feature_dim, 3, stride=1, device=device)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(args.cnn_feature_dim * 7 * 7, args.transformer_dim, device=device)),
            nn.ReLU(),
        )
        self.pad_action_id = n_actions
        self.embed_action = nn.Embedding(n_actions + 1, args.transformer_dim, device=device)
        self.embed_timestep = nn.Embedding(args.max_ep_len, args.transformer_dim, device=device)

        self.transformer = Transformer(
            num_layers=args.transformer_layers,
            dim=args.transformer_dim,
            num_heads=args.num_heads,
            max_len=2 * args.max_ep_len + 1,
            dropout=args.dropout,
            device=device,
        )
        self.action_net = layer_init(nn.Linear(args.transformer_dim, n_actions, device=device), std=0.01)

        self.embed_state_v = nn.Sequential(
            layer_init(nn.Conv2d(4, 64, 8, stride=4, device=device)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 4, stride=2, device=device)),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, args.cnn_feature_dim_v, 3, stride=1, device=device)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(args.cnn_feature_dim_v * 7 * 7, args.v_dim, device=device)),
            nn.ReLU(),
        )
        self.value_net = layer_init(nn.Linear(args.v_dim, 1, device=device), std=1.0)


    def act_kvcache_per_env(self, obs_t, ts_t, prev_action, is_new_ep, kv_buf, kv_len):
        device = obs_t.device
        N = obs_t.shape[0]
        A = self.action_net.out_features
        num_layers = self.transformer.num_layers
        obs_t = obs_t / 255.0

        action = torch.empty((N,), device=device, dtype=torch.long)
        logprob = torch.empty((N,), device=device, dtype=torch.float32)
        probs = torch.empty((N, A), device=device, dtype=torch.float32)

        new_idx = torch.nonzero(is_new_ep, as_tuple=False).squeeze(-1)
        cont_idx = torch.nonzero(~is_new_ep, as_tuple=False).squeeze(-1)

        if cont_idx.numel() > 0:
            past_kvs, past_mask, _ = pack_kvs_vectorized(kv_buf, kv_len, cont_idx)
            obs_c = obs_t.index_select(0, cont_idx)
            ts_c = ts_t.index_select(0, cont_idx)
            pa_c = prev_action.index_select(0, cont_idx)
            ts_a = (ts_c - 1).clamp(min=0)

            s = self.embed_state(obs_c)
            a = self.embed_action(pa_c) + self.embed_timestep(ts_a)
            s = s + self.embed_timestep(ts_c.clamp(max=self.embed_timestep.num_embeddings - 1))

            x_new = torch.stack([a, s], dim=1)
            B = cont_idx.numel()
            new_m = torch.ones((B, x_new.shape[1]), device=device, dtype=torch.bool)
            pad_m = torch.cat([past_mask, new_m], dim=1)

            latent, present_kvs = self.transformer(x_new, pad_mask=pad_m, past_kvs=past_kvs)
            h = latent[:, -1, :]

            dist = Categorical(logits=self.action_net(h))
            act = dist.sample()
            action[cont_idx] = act
            logprob[cont_idx] = dist.log_prob(act)
            probs[cont_idx] = dist.probs
            unpack_kvs_vectorized(present_kvs, cont_idx, kv_buf, kv_len)

        if new_idx.numel() > 0:
            obs_n = obs_t.index_select(0, new_idx)
            ts_n = ts_t.index_select(0, new_idx)

            s = self.embed_state(obs_n) + self.embed_timestep(ts_n.clamp(max=self.embed_timestep.num_embeddings - 1))
            x_new = s.unsqueeze(1)

            latent, present_kvs = self.transformer(x_new, pad_mask=None, past_kvs=[None] * num_layers)
            h = latent[:, -1, :]

            dist = Categorical(logits=self.action_net(h))
            act = dist.sample()
            action[new_idx] = act
            logprob[new_idx] = dist.log_prob(act)
            probs[new_idx] = dist.probs
            unpack_kvs_vectorized(present_kvs, new_idx, kv_buf, kv_len)

        return action, logprob, probs


    def evaluate_actions(self, flat_obs, flat_act, flat_ts, splits, lmax):
        device = flat_obs.device
        B = len(splits)
        Lmax = lmax
        D = self.transformer_dim
        flat_obs = flat_obs / 255.0
        C, H, W = flat_obs.shape[1:]

        obs_b = torch.zeros(B, Lmax, C, H, W, device=device, dtype=flat_obs.dtype)
        act_b = torch.zeros(B, Lmax, device=device, dtype=torch.long)
        ts_b = torch.zeros(B, Lmax, device=device, dtype=torch.long)
        mask = torch.zeros(B, Lmax, device=device, dtype=torch.bool)

        cursor = 0
        for b, L in enumerate(splits):
            obs_b[b, :L] = flat_obs[cursor:cursor + L]
            act_b[b, :L] = flat_act[cursor:cursor + L].long()
            ts_b[b, :L] = flat_ts[cursor:cursor + L].long()
            mask[b, :L] = True
            cursor += L

        obs_flat = obs_b.reshape(B * Lmax, C, H, W)
        s = self.embed_state(obs_flat).reshape(B, Lmax, D)
        a = self.embed_action(act_b)
        t = self.embed_timestep(ts_b.clamp(max=self.embed_timestep.num_embeddings - 1))

        s = s + t
        a = a + t

        x_full = torch.stack([s, a], dim=2).reshape(B, 2 * Lmax, D)
        x = x_full[:, :-1, :]

        tok_mask_full = torch.stack([mask, mask], dim=2).reshape(B, 2 * Lmax)
        tok_mask = tok_mask_full[:, :-1]

        latent, _ = self.transformer(x, pad_mask=tok_mask)
        h_s = latent[:, 0::2, :]

        logits = self.action_net(h_s.reshape(B * Lmax, D))
        dist = Categorical(logits=logits)
        logp = dist.log_prob(act_b.reshape(-1)).reshape(B, Lmax)
        entropy = dist.entropy().reshape(B, Lmax)
        
        obs_flat_v = obs_b.reshape(B * Lmax, C, H, W)
        h_v = self.embed_state_v(obs_flat_v)
        values = self.value_net(h_v).squeeze(-1).reshape(B, Lmax)
        # values = self.value_net(h_s.reshape(B * Lmax, D)).squeeze(-1).reshape(B, Lmax)

        logp_list, ent_list, val_list = [], [], []
        for b, L in enumerate(splits):
            logp_list.append(logp[b, :L])
            ent_list.append(entropy[b, :L])
            val_list.append(values[b, :L])

        return (
            torch.cat(logp_list, dim=0),
            torch.cat(ent_list, dim=0),
            torch.cat(val_list, dim=0),
        )


    @torch.no_grad()
    def get_value(self, obs, ts):
        obs_norm = obs.float() / 255.0
        h = self.embed_state_v(obs_norm) 
        return self.value_net(h).squeeze(-1) 


@torch.no_grad()
def rollout(envs, policy, args, next_obs, next_done, prev_action,
            next_timestep, kv_buf, kv_len, avg_returns):
    ts = []
    device = args.device

    obs = next_obs
    is_new_ep = next_done
    timesteps = next_timestep

    for _ in range(args.num_steps):
        action, logprob, probs = policy(
            obs_t=obs, ts_t=timesteps, prev_action=prev_action,
            is_new_ep=is_new_ep, kv_buf=kv_buf, kv_len=kv_len,
        )
        next_obs_np, reward, next_done_np, info = envs.step(action.cpu().numpy())

        reward = torch.as_tensor(reward, device=device, dtype=torch.float32)
        done_mask = torch.as_tensor(next_done_np, device=device, dtype=torch.bool)
        next_obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.uint8)

        real_done = done_mask & torch.as_tensor(
            info["lives"] == 0, device=device, dtype=torch.bool
        )
        if real_done.any():
            ep_returns = torch.as_tensor(info["r"], device=device, dtype=torch.float32)
            avg_returns.extend(ep_returns[real_done].tolist())

        ts.append(
            TensorDict._new_unsafe(
                obs=obs,
                dones=is_new_ep,
                actions=action,
                logprobs=logprob,
                old_probs=probs,
                rewards=reward,
                dones_after=done_mask,
                timesteps=timesteps,
                batch_size=(args.num_envs,),
            )
        )

        prev_action = action.clone()
        timesteps = torch.clamp(timesteps + 1, max=args.max_ep_len - 1)
        timesteps[done_mask] = 0

        done_idx = torch.nonzero(done_mask, as_tuple=False).flatten()
        kv_buf[done_idx] = 0
        kv_len[done_idx] = 0
        prev_action[done_mask] = args.pad_action_id

        next_done = done_mask
        is_new_ep = next_done
        obs = next_obs

    container = torch.stack(ts, 0)
    return obs, next_done, prev_action, timesteps, kv_buf, kv_len, container


@torch.no_grad()
def build_ppo_dataset(container, next_obs, next_done, next_timestep,
                      agent, args, traj_minibatch_size=64):

    obs = container["obs"]
    actions = container["actions"]
    logprobs = container["logprobs"]
    rewards = container["rewards"]
    old_probs = container["old_probs"]
    dones_after = container["dones_after"]
    timesteps = container["timesteps"]

    num_steps, num_envs = actions.shape
    device = obs.device

    flat_obs_list, flat_act_list, flat_lp_list = [], [], []
    flat_rew_list, flat_prob_list, flat_ts_list = [], [], []
    lengths, last_values = [], []

    boot_env_indices = []
    boot_positions = []

    for env in range(num_envs):
        done_col = dones_after[:, env]
        done_idx = torch.nonzero(done_col, as_tuple=False).flatten().tolist()
        start = 0

        for t in done_idx:
            end = t + 1
            if end > start:
                flat_obs_list.append(obs[start:end, env])
                flat_act_list.append(actions[start:end, env])
                flat_lp_list.append(logprobs[start:end, env])
                flat_rew_list.append(rewards[start:end, env])
                flat_prob_list.append(old_probs[start:end, env])
                flat_ts_list.append(timesteps[start:end, env])
                lengths.append(end - start)
                last_values.append(torch.tensor(0.0, device=device))
            start = end

        if start < num_steps:
            flat_obs_list.append(obs[start:num_steps, env])
            flat_act_list.append(actions[start:num_steps, env])
            flat_lp_list.append(logprobs[start:num_steps, env])
            flat_rew_list.append(rewards[start:num_steps, env])
            flat_prob_list.append(old_probs[start:num_steps, env])
            flat_ts_list.append(timesteps[start:num_steps, env])
            lengths.append(num_steps - start)
            boot_positions.append(len(last_values))
            boot_env_indices.append(env)
            last_values.append(None)

    if boot_env_indices:
        boot_idx = torch.tensor(boot_env_indices, device=device, dtype=torch.long)
        boot_obs = next_obs[boot_idx]
        boot_ts = next_timestep[boot_idx]
        boot_vals = agent.get_value(boot_obs, boot_ts)
        boot_vals = boot_vals * (~next_done[boot_idx]).float()
        for k, pos in enumerate(boot_positions):
            last_values[pos] = boot_vals[k]

    flat_obs = torch.cat(flat_obs_list)
    flat_actions = torch.cat(flat_act_list)
    flat_logprobs = torch.cat(flat_lp_list)
    flat_rewards = torch.cat(flat_rew_list)
    flat_old_probs = torch.cat(flat_prob_list)
    flat_ts = torch.cat(flat_ts_list)

    flat_values = _values_traj_minibatches(
        agent=agent,
        flat_obs=flat_obs,
        flat_action=flat_actions,
        flat_ts=flat_ts,
        splits=lengths,
        traj_minibatch_size=traj_minibatch_size,
    )

    flat_advantages, flat_returns = _gae_for_splits(
        rewards_flat=flat_rewards,
        values_flat=flat_values,
        lasts=last_values,
        splits=lengths,
        gamma=args.gamma,
        lam=args.gae_lambda,
    )

    dataset = {
        "obs": flat_obs,
        "actions": flat_actions,
        "logprobs": flat_logprobs,
        "rewards": flat_rewards,
        "old_probs": flat_old_probs,
        "timesteps": flat_ts,
        "values": flat_values,
        "advantages": flat_advantages,
        "returns": flat_returns,
        "lengths": np.asarray(lengths, dtype=np.int64),
    }
    dataset["end_indices"] = np.cumsum(dataset["lengths"])
    dataset["start_indices"] = np.insert(dataset["end_indices"], 0, 0)[:-1]
    return dataset


@torch.no_grad()
def _values_traj_minibatches(agent, flat_obs, flat_action, flat_ts, splits,
                              traj_minibatch_size=64):

    flat_obs_norm = flat_obs.float() / 255.0
    h = agent.embed_state_v(flat_obs_norm) 
    return agent.value_net(h).squeeze(-1) 


def _gae_for_splits(rewards_flat, values_flat, lasts, splits, gamma, lam):
    adv_chunks, ret_chunks = [], []
    cursor = 0
    for L, last_v in zip(splits, lasts):
        L = int(L)
        r = rewards_flat[cursor:cursor + L]
        v = values_flat[cursor:cursor + L]
        adv = torch.zeros_like(r)
        gae = 0.0
        for t in reversed(range(L)):
            next_v = last_v if t == L - 1 else v[t + 1]
            delta = r[t] + gamma * next_v - v[t]
            gae = delta + gamma * lam * gae
            adv[t] = gae
        adv_chunks.append(adv)
        ret_chunks.append(adv + v)
        cursor += L
    return torch.cat(adv_chunks, 0), torch.cat(ret_chunks, 0)


def iter_traj_batches(dataset, batch_size):
    start_indices = dataset["start_indices"]
    end_indices = dataset["end_indices"]
    traj_order = np.random.permutation(len(start_indices))
    cursor = 0

    while cursor < len(traj_order):
        chosen = []
        total_frames = 0

        while cursor < len(traj_order) and total_frames < batch_size:
            i = traj_order[cursor]
            chosen.append(i)
            total_frames += end_indices[i] - start_indices[i]
            cursor += 1

        if total_frames < batch_size:
            break

        obs_list, act_list, logp_list = [], [], []
        ret_list, adv_list, ts_list = [], [], []
        val_list, splits = [], []

        for i in chosen:
            s, e = start_indices[i], end_indices[i]
            obs_list.append(dataset["obs"][s:e])
            act_list.append(dataset["actions"][s:e])
            logp_list.append(dataset["logprobs"][s:e])
            ret_list.append(dataset["returns"][s:e])
            adv_list.append(dataset["advantages"][s:e])
            ts_list.append(dataset["timesteps"][s:e])
            val_list.append(dataset["values"][s:e])
            splits.append(e - s)

        Lmax = max(splits)
        yield (
            torch.cat(obs_list, 0),
            torch.cat(act_list, 0),
            torch.cat(logp_list, 0),
            torch.cat(ret_list, 0),
            torch.cat(adv_list, 0),
            torch.cat(ts_list, 0),
            torch.cat(val_list, 0),
            splits,
            Lmax,
        )


def lr_multiplier(step_idx, total_steps, args):
    warmup_steps = int(total_steps * args.warmup_frac)
    if warmup_steps > 0 and step_idx < warmup_steps:
        return step_idx / warmup_steps

    rem = float(max(1, total_steps - warmup_steps))
    progress = (step_idx - warmup_steps) / rem
    progress = float(np.clip(progress, 0.0, 1.0))

    if args.lr_schedule == "constant":
        decay = 1.0
    elif args.lr_schedule == "linear":
        decay = 1.0 - progress
    else:  # cosine
        decay = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * decay


def explained_variance(y_pred, y_true, eps=1e-8):
    var_y = torch.var(y_true)
    if var_y.item() < eps:
        return float("nan")
    return (1.0 - torch.var(y_true - y_pred) / (var_y + eps)).item()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    args.num_iterations = args.total_timesteps // int(args.num_envs * args.num_steps)

    run_name = (
        f"{args.exp_name}, seed={args.seed}, lr={args.learning_rate}, "
        f"num_env={args.num_envs}, batch={args.batch_size}, "
        f"update_epochs={args.update_epochs}, layers={args.transformer_layers}, "
        f"dim={args.transformer_dim}, heads={args.num_heads}, "
        f"cnn_dim={args.cnn_feature_dim}, cnn_dim_v={args.cnn_feature_dim_v}, "
        f"ent_coef={args.ent_coef}"
    )

    wandb.init(
        project=args.project_name,
        name=run_name,
        config=vars(args),
        save_code=True,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    args.device = device

    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete)

    agent = Agent(envs, args)
    agent_inference = Agent(envs, args)
    args.pad_action_id = agent.pad_action_id

    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    policy = agent_inference.act_kvcache_per_env

    H = args.num_heads
    Dh = args.transformer_dim // args.num_heads
    S = 2 * args.max_ep_len + 2

    kv_buf = torch.zeros(args.num_envs, args.transformer_layers, 2, H, S, Dh, device=device)
    kv_len = torch.zeros(args.num_envs, dtype=torch.long, device=device)

    # ── Bookkeeping ───────────────────────────────────────────────────────
    avg_returns = deque(maxlen=args.avg_returns_last_k)
    global_step = 0
    global_start = time.time()

    next_obs = torch.tensor(envs.reset(), device=device, dtype=torch.uint8)
    next_done = torch.ones(args.num_envs, device=device, dtype=torch.bool)
    prev_action = torch.full((args.num_envs,), agent.pad_action_id, device=device, dtype=torch.long)
    next_timestep = torch.zeros(args.num_envs, device=device, dtype=torch.long)

    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))

    for iteration in pbar:
        start_time = time.time()

        lr_mul = lr_multiplier(iteration, args.num_iterations + 1, args)
        lrnow  = lr_mul * args.learning_rate
        optimizer.param_groups[0]["lr"].copy_(lrnow)

        kv_buf.zero_()
        kv_len.zero_()
        next_done = torch.ones(args.num_envs, device=device, dtype=torch.bool)

        next_obs, next_done, prev_action, next_timestep, kv_buf, kv_len, container = rollout(
            envs, policy, args,
            next_obs, next_done, prev_action, next_timestep,
            kv_buf, kv_len, avg_returns,
        )
        global_step += container.numel()

        agent_inference_p = from_module(agent_inference).data
        agent_inference_p.to_module(agent)

        rollout_dataset = build_ppo_dataset(
            container=container,
            next_obs=next_obs,
            next_done=next_done,
            next_timestep=next_timestep,
            agent=agent,
            args=args,
            traj_minibatch_size=64,
        )

        mb_stats = {k: [] for k in ("v_loss", "pg_loss", "entropy", "clipfrac", "grad_norm")}
        all_v_pred, all_v_true = [], []

        for epoch in range(args.update_epochs):
            for batch in iter_traj_batches(rollout_dataset, batch_size=args.batch_size):
                (
                    data_obs, data_act, old_logp,
                    data_ret, data_adv, data_ts,
                    old_val, data_splits, data_lmax,
                ) = batch

                if args.norm_adv:
                    data_adv = (data_adv - data_adv.mean()) / (data_adv.std() + 1e-8)

                new_logp, ent, new_val = agent.evaluate_actions(
                    flat_obs=data_obs,
                    flat_act=data_act,
                    flat_ts=data_ts,
                    splits=data_splits,
                    lmax=data_lmax,
                )

                log_ratio = new_logp - old_logp
                ratio = log_ratio.exp()
                pg_loss1 = -data_adv * ratio
                pg_loss2 = -data_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

                if args.clip_vloss:
                    v_clipped = old_val + torch.clamp(new_val - old_val, -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(
                        (new_val - data_ret).pow(2),
                        (v_clipped - data_ret).pow(2),
                    ).mean()
                else:
                    v_loss = 0.5 * (new_val - data_ret).pow(2).mean()

                entropy_loss = ent.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                mb_stats["v_loss"].append(v_loss.detach())
                mb_stats["pg_loss"].append(pg_loss.detach())
                mb_stats["entropy"].append(entropy_loss.detach())
                mb_stats["clipfrac"].append(clipfrac.detach())
                mb_stats["grad_norm"].append(grad_norm.detach())
                all_v_pred.append(new_val.detach())
                all_v_true.append(data_ret.detach())

        train_p = from_module(agent).data
        train_p.to_module(agent_inference)

        if iteration % args.log_freq == 0:
            speed = (args.num_envs * args.num_steps) / (time.time() - start_time)

            r = container["rewards"].mean().item()
            r_max = container["rewards"].max().item()
            avg_returns_t = float(np.mean(avg_returns)) if len(avg_returns) > 0 else float("nan")

            ev = explained_variance(
                torch.cat(all_v_pred), torch.cat(all_v_true)
            ) if all_v_pred else float("nan")

            update_stats = {k: torch.stack(v).mean().item() for k, v in mb_stats.items()}
            lr = optimizer.param_groups[0]["lr"]
            lr = lr.item() if torch.is_tensor(lr) else lr

            pbar.set_description(
                f"speed: {speed: 4.1f} sps, "
                f"reward avg: {r :4.2f}, "
                f"reward max: {r_max:4.2f}, "
                f"returns: {avg_returns_t: 4.2f},"
                f"lr: {lr: 4.6f}"
            )

            wandb.log(
                {
                    "speed": speed,
                    "lr": lr,
                    "episode_return": avg_returns_t, 
                    "explained_variance": ev,
                    "clipfrac": update_stats["clipfrac"],
                    "policy_loss": update_stats["pg_loss"],
                    "value_loss": update_stats["v_loss"],
                    "entropy": update_stats["entropy"],
                    "grad_norm": update_stats["grad_norm"],
                    "time": time.time() - global_start,
                },
                step=global_step,
            )

    envs.close()