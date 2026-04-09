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
from torch.distributions.categorical import Categorical, Distribution

Distribution.set_default_validate_args(False)
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment args
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__)[:-len(".py")])
    parser.add_argument("--project_name", type=str, default="da2e_mpl_vadv_atari")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch_deterministic", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cudagraphs", action="store_true")

    # Algorithm args
    parser.add_argument("--env_id", type=str, default="Breakout-v5")
    parser.add_argument("--total_timesteps", type=int, default=10_000_000)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--learning_rate_vf", type=float, default=2.5e-4)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--anneal_lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--norm_adv", action="store_true")
    parser.add_argument("--clip_coef", type=float, default=0.1)
    parser.add_argument("--clip_vloss", action="store_true")
    parser.add_argument("--ent_coef", type=float, default=0.1)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.5)
    parser.add_argument("--max_grad_norm_vf", type=float, default=1.5)
    parser.add_argument("--target_kl", type=float, default=None)
    parser.add_argument("--kl_coef", type=float, default=0.0)
    parser.add_argument("--full_action", action="store_true")
    parser.add_argument("--anneal_clip_coef", action="store_true")

    # Runtime filled
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_iterations", type=int, default=0)

    # Logging
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--avg_returns_last_k", type=int, default=20)

    # Policy transformer args
    parser.add_argument("--pi_transformer_layers", type=int, default=1)
    parser.add_argument("--pi_transformer_dim", type=int, default=128)
    parser.add_argument("--pi_num_heads", type=int, default=2)
    parser.add_argument("--pi_cnn_feature_dim", type=int, default=128)

    parser.add_argument("--v_dim", type=int, default=512)
    parser.add_argument("--v_cnn_feature_dim", type=int, default=128)

    parser.add_argument("--max_ep_len", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.00)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)

    parser.add_argument("--lr_schedule", type=str, default="linear",
                        choices=["constant", "linear", "cosine"])
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
        return observations, rewards, dones, infos


def explained_variance(y_pred, y_true, eps=1e-8):
    var_y = torch.var(y_true)
    if var_y.item() < eps:
        return torch.tensor(float("nan"), device=y_true.device)
    return 1.0 - torch.var(y_true - y_pred) / (var_y + eps)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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
        T_past = T_total - T_new

        rows = torch.arange(T_new, device=x.device).unsqueeze(1)
        cols = torch.arange(T_total, device=x.device).unsqueeze(0)
        causal = cols <= rows + T_past

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
        present_kv = (k, v)
        return out, None, present_kv


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
        attn_out, attn_w, present_kv = self.attention(hidden_states, pad_mask=pad_mask, past_kv=past_kv)
        hidden_states = self.dropout(attn_out) + x
        x_ = self.post_attention_layernorm(hidden_states)
        ff = self.fc_projection(x_)
        out = ff + hidden_states
        return out, attn_w, present_kv


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
        attn_weights = []
        present_kvs = []
        for i, layer in enumerate(self.layers):
            x, attn, present_kv = layer(x, pad_mask=pad_mask, past_kv=past_kvs[i])
            attn_weights.append(attn)
            present_kvs.append(present_kv)
        return x, attn_weights, present_kvs



class PolicyNet(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        device = args.device
        n_actions = envs.single_action_space.n
        D = args.pi_transformer_dim

        self.embed_state = nn.Sequential(
            layer_init(nn.Conv2d(4, 64, 8, stride=4, device=device)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 4, stride=2, device=device)),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, args.pi_cnn_feature_dim, 3, stride=1, device=device)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(args.pi_cnn_feature_dim * 7 * 7, D, device=device)),
            nn.ReLU(),
        )
        self.pad_action_id = n_actions
        self.embed_action = nn.Embedding(n_actions + 1, D, device=device)
        self.embed_timestep = nn.Embedding(args.max_ep_len, D, device=device)

        self.transformer = Transformer(
            num_layers=args.pi_transformer_layers,
            dim=D,
            num_heads=args.pi_num_heads,
            max_len=2 * args.max_ep_len + 1,
            dropout=args.dropout,
            device=device,
        )
        self.action_net = layer_init(nn.Linear(D, n_actions, device=device), std=0.01)
        self.transformer_dim = D


    def act_kvcache_per_env(self, obs_t, ts_t, prev_action, is_new_ep, kv_buf, kv_len):
        device = obs_t.device
        N = obs_t.shape[0]
        D = self.transformer_dim
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
            a = self.embed_action(pa_c)
            t = self.embed_timestep(ts_c.clamp(max=self.embed_timestep.num_embeddings - 1))
            t_a = self.embed_timestep(ts_a)
            a = a + t_a
            s = s + t
            x_new = torch.stack([a, s], dim=1)

            B = cont_idx.numel()
            new_mask = torch.ones((B, x_new.shape[1]), device=device, dtype=torch.bool)
            pad_mask_total = torch.cat([past_mask, new_mask], dim=1)

            latent, _, present_kvs = self.transformer(x_new, pad_mask=pad_mask_total, past_kvs=past_kvs)
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

            s = self.embed_state(obs_n)
            t = self.embed_timestep(ts_n.clamp(max=self.embed_timestep.num_embeddings - 1))
            s = s + t
            x_new = s.unsqueeze(1)

            latent, _, present_kvs = self.transformer(x_new, pad_mask=None, past_kvs=[None] * num_layers)
            h = latent[:, -1, :]

            dist = Categorical(logits=self.action_net(h))
            act = dist.sample()
            action[new_idx] = act
            logprob[new_idx] = dist.log_prob(act)
            probs[new_idx] = dist.probs
            unpack_kvs_vectorized(present_kvs, new_idx, kv_buf, kv_len)

        return action, logprob, probs



    def evaluate_state_policy(self, data_obs, data_ts, data_actions, splits):
        device = data_obs.device
        B = len(splits)
        Lmax = max(splits)
        data_obs = data_obs / 255.0
        C, H, W = data_obs.shape[1:]
        n_actions = self.action_net.out_features
        D = self.transformer_dim

        obs_batch = torch.zeros(B, Lmax, C, H, W, device=device, dtype=data_obs.dtype)
        act_batch = torch.zeros(B, Lmax, device=device, dtype=torch.long)
        ts_batch = torch.zeros(B, Lmax, device=device, dtype=torch.long)
        pad_mask = torch.zeros(B, Lmax, device=device, dtype=torch.bool)

        cursor = 0
        for b, L in enumerate(splits):
            obs_batch[b, :L] = data_obs[cursor:cursor + L]
            act_batch[b, :L] = data_actions[cursor:cursor + L].long()
            ts_batch[b, :L] = data_ts[cursor:cursor + L].long()
            pad_mask[b, :L] = True
            cursor += L

        obs_flat = obs_batch.reshape(B * Lmax, C, H, W)
        s = self.embed_state(obs_flat).reshape(B, Lmax, D)
        a = self.embed_action(act_batch)
        t = self.embed_timestep(ts_batch.clamp(max=self.embed_timestep.num_embeddings - 1))
        s = s + t
        a = a + t

        x_full = torch.stack([s, a], dim=2).reshape(B, 2 * Lmax, D)
        x = x_full[:, :-1, :]

        token_mask_full = torch.stack([pad_mask, pad_mask], dim=2).reshape(B, 2 * Lmax)
        token_mask = token_mask_full[:, :-1]

        latent, _, _ = self.transformer(x, pad_mask=token_mask)
        h_s = latent[:, 0::2, :] 
        h_s_flat = h_s.reshape(B * Lmax, D)

        logits = self.action_net(h_s_flat)
        dist = Categorical(logits=logits)
        probs_flat = dist.probs 
        entropy_flat = dist.entropy()

        probs = probs_flat.reshape(B, Lmax, n_actions)
        entropy = entropy_flat.reshape(B, Lmax)

        probs_list, entropy_list = [], []
        for b, L in enumerate(splits):
            probs_list.append(probs[b, :L])
            entropy_list.append(entropy[b, :L])

        new_dist = torch.cat(probs_list, dim=0)
        entropy_out = torch.cat(entropy_list, dim=0)
        new_logprobs = torch.log(new_dist.clamp(min=1e-8))

        return new_dist, new_logprobs, entropy_out


class ValueAdvNet(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        device = args.device
        n_actions = envs.single_action_space.n

        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(4, 64, 8, stride=4, device=device)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, 4, stride=2, device=device)),
            nn.ReLU(),
            layer_init(nn.Conv2d(128, args.v_cnn_feature_dim, 3, stride=1, device=device)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(args.v_cnn_feature_dim * 7 * 7, args.v_dim, device=device)),
            nn.ReLU(),
        )

        self.value_net = layer_init(nn.Linear(args.v_dim, 1, device=device), std=1.0)
        self.advantage_net = layer_init(nn.Linear(args.v_dim, n_actions, device=device), std=0.1)
        self.n_actions = n_actions

    def encode_state(self, obs):
        return self.encoder(obs / 255.0)

    def forward_values_tokens_dt(self, flat_obs, flat_act, flat_ts, splits):
        h = self.encode_state(flat_obs)
        return self.value_net(h).squeeze(-1)

    def evaluate_state(self, data_obs, data_ts, data_actions, old_dist, splits):
        h = self.encode_state(data_obs) 
        values = self.value_net(h).squeeze(-1) 
        advantages_raw = self.advantage_net(h) 
        advantages = advantages_raw - torch.sum(
            old_dist * advantages_raw, dim=-1, keepdim=True
        )
        return values, advantages


@torch.no_grad()
def rollout(envs, policy, args, next_obs, next_done, prev_action, next_timestep,
            kv_buf, kv_len, avg_returns):
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
            tensordict.TensorDict._new_unsafe(
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
def build_attention_dataset(container, next_obs, next_done, next_timestep,
                             vadv, traj_minibatch_size=64):
    obs = container["obs"]
    actions = container["actions"]
    logprobs = container["logprobs"]
    rewards = container["rewards"]
    old_probs = container["old_probs"]
    dones_after = container["dones_after"]
    timesteps = container["timesteps"]

    num_steps, num_envs = actions.shape
    device = obs.device

    flat_obs_list, flat_actions_list = [], []
    flat_logprobs_list, flat_rewards_list = [], []
    flat_probs_list, flat_ts_list = [], []
    lengths, last_values = [], []
    boot_env_indices, boot_positions = [], []

    for env in range(num_envs):
        done_col = dones_after[:, env]
        done_idx = torch.nonzero(done_col, as_tuple=False).flatten().tolist()
        start = 0

        for t in done_idx:
            end = t + 1
            if end > start:
                flat_obs_list.append(obs[start:end, env])
                flat_actions_list.append(actions[start:end, env])
                flat_logprobs_list.append(logprobs[start:end, env])
                flat_rewards_list.append(rewards[start:end, env])
                flat_probs_list.append(old_probs[start:end, env])
                flat_ts_list.append(timesteps[start:end, env])
                lengths.append(end - start)
                last_values.append(torch.tensor(0.0, device=device))
            start = end

        if start < num_steps:
            flat_obs_list.append(obs[start:num_steps, env])
            flat_actions_list.append(actions[start:num_steps, env])
            flat_logprobs_list.append(logprobs[start:num_steps, env])
            flat_rewards_list.append(rewards[start:num_steps, env])
            flat_probs_list.append(old_probs[start:num_steps, env])
            flat_ts_list.append(timesteps[start:num_steps, env])
            lengths.append(num_steps - start)

            boot_positions.append(len(last_values))
            boot_env_indices.append(env)
            last_values.append(None)

    if boot_env_indices:
        boot_idx = torch.tensor(boot_env_indices, device=device, dtype=torch.long)
        boot_obs = next_obs[boot_idx]
        boot_ts = next_timestep[boot_idx]
        dummy_act = torch.full((len(boot_env_indices),), 0, device=device, dtype=torch.long)

        boot_vals = vadv.forward_values_tokens_dt(
            flat_obs=boot_obs, flat_act=dummy_act,
            flat_ts=boot_ts, splits=[1] * len(boot_env_indices),
        )
        boot_vals = boot_vals * (~next_done[boot_idx]).float()
        for k, pos in enumerate(boot_positions):
            last_values[pos] = boot_vals[k]

    flat_obs = torch.cat(flat_obs_list)
    flat_actions = torch.cat(flat_actions_list)
    flat_logprobs = torch.cat(flat_logprobs_list)
    flat_rewards = torch.cat(flat_rewards_list)
    flat_old_probs = torch.cat(flat_probs_list)
    flat_ts = torch.cat(flat_ts_list)

    flat_values = _forward_values_minibatches(
        vadv, flat_obs, flat_actions, flat_ts, lengths, traj_minibatch_size,
    )

    dataset = {
        "obs": flat_obs,
        "actions": flat_actions,
        "logprobs": flat_logprobs,
        "rewards": flat_rewards,
        "old_probs": flat_old_probs,
        "timesteps": flat_ts,
        "values": flat_values,
        "last_values": torch.stack(last_values),
        "lengths": np.asarray(lengths, dtype=np.int64),
    }
    dataset["end_indices"] = np.cumsum(dataset["lengths"])
    dataset["start_indices"] = np.insert(dataset["end_indices"], 0, 0)[:-1]
    return dataset


@torch.no_grad()
def _forward_values_minibatches(vadv, flat_obs, flat_act, flat_ts, splits,
                                 traj_minibatch_size=64):
    splits = list(map(int, splits))
    starts = [0]
    for L in splits[:-1]:
        starts.append(starts[-1] + L)

    chunks = []
    for i in range(0, len(splits), traj_minibatch_size):
        mb_splits = splits[i:i + traj_minibatch_size]
        mb_start = starts[i]
        mb_end = mb_start + sum(mb_splits)
        mb_v = vadv.forward_values_tokens_dt(
            flat_obs=flat_obs[mb_start:mb_end],
            flat_act=flat_act[mb_start:mb_end],
            flat_ts=flat_ts[mb_start:mb_end],
            splits=mb_splits,
        )
        chunks.append(mb_v)
    return torch.cat(chunks, dim=0)


def iter_traj_batches(dataset, batch_size):
    start_indices = dataset["start_indices"]
    end_indices = dataset["end_indices"]
    traj_order = np.random.permutation(len(start_indices))
    cursor = 0

    while cursor < len(traj_order):
        chosen, total_frames = [], 0
        while cursor < len(traj_order) and total_frames < batch_size:
            i = traj_order[cursor]
            chosen.append(i)
            total_frames += end_indices[i] - start_indices[i]
            cursor += 1

        if total_frames < batch_size:
            break

        obs_l, act_l, rew_l, oprob_l, lp_l, ts_l, val_l, last_l, splits = \
            [], [], [], [], [], [], [], [], []

        for i in chosen:
            s, e = start_indices[i], end_indices[i]
            obs_l.append(dataset["obs"][s:e])
            act_l.append(dataset["actions"][s:e])
            rew_l.append(dataset["rewards"][s:e])
            oprob_l.append(dataset["old_probs"][s:e])
            lp_l.append(dataset["logprobs"][s:e])
            ts_l.append(dataset["timesteps"][s:e])
            val_l.append(dataset["values"][s:e])
            last_l.append(dataset["last_values"][i])
            splits.append(e - s)

        Lmax = max(splits)
        yield (
            torch.cat(obs_l),
            torch.cat(act_l),
            torch.cat(rew_l),
            torch.cat(oprob_l),
            torch.cat(lp_l),
            torch.cat(ts_l),
            torch.cat(val_l),
            torch.stack(last_l),
            splits,
            Lmax,
        )


def compute_value_loss(deltas, values, lasts, discount_matrix, discount_vector):
    return torch.cat(
        [
            (
                discount_matrix[:len(d), :len(d)].matmul(d)
                + l * discount_vector[-len(d):]
                - v
            ).square()
            for d, v, l in zip(deltas, values, lasts)
        ]
    ).mean()


def dae_targets_and_preds(deltas, values, lasts, discount_matrix, discount_vector):
    targets, preds = [], []
    for d, v, l in zip(deltas, values, lasts):
        T = len(d)
        y = discount_matrix[:T, :T].matmul(d) + l * discount_vector[-T:]
        targets.append(y)
        preds.append(v)
    return torch.cat(targets), torch.cat(preds)


def normalize_advantage(advantages, policies, eps=1e-5):
    std = (policies * advantages.pow(2)).sum(dim=1).mean().sqrt()
    return advantages / (std + eps)


def compute_policy_loss(args, advantages, log_policy, old_log_policy, actions, clip_range):
    if args.full_action:
        loss = -(advantages * torch.exp(log_policy)).sum(dim=1).mean()
        ratio = torch.exp(log_policy - old_log_policy)
    else:
        adv = advantages.gather(-1, actions).flatten()
        logp = log_policy.gather(-1, actions).flatten()
        old_logp = old_log_policy.gather(-1, actions).flatten()
        ratio = torch.exp(logp - old_logp)
        policy_loss_1 = adv * ratio
        policy_loss_2 = adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
        loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    return loss, ratio


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
    else:
        decay = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * decay


if __name__ == "__main__":
    args = parse_args()
    print(args)

    args.num_iterations = args.total_timesteps // int(args.num_envs * args.num_steps)
    run_name = (
        f"{args.exp_name}, seed={args.seed}, "
        f"lr={args.learning_rate}, num_env={args.num_envs}, "
        f"batch={args.batch_size}, update_epochs={args.update_epochs},"
        f"pi_layers={args.pi_transformer_layers}, pi_dim={args.pi_transformer_dim}, "
        f"pi_heads={args.pi_num_heads}, v_dim={args.v_dim}, "
        f"lr={args.learning_rate}, lr_vf={args.learning_rate_vf}, "
        f"entropy={args.ent_coef}"
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

    # Networks
    policy = PolicyNet(envs, args)
    policy_inference = PolicyNet(envs, args)

    args.pad_action_id = policy.pad_action_id

    # Keep inference policy in sync via shared parameters (same as file 1)
    policy_inference_p = from_module(policy).data
    policy_inference_p.to_module(policy_inference)

    vadv = ValueAdvNet(envs, args)

    optimizer_pi = optim.Adam(
        policy.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )
    optimizer_v = optim.Adam(
        vadv.parameters(),
        lr=torch.tensor(args.learning_rate_vf, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    H = args.pi_num_heads
    Dh = args.pi_transformer_dim // args.pi_num_heads
    S = 2 * args.max_ep_len + 2

    kv_buf = torch.zeros(
        args.num_envs, args.pi_transformer_layers, 2, H, S, Dh, device=device
    )
    kv_len = torch.zeros(args.num_envs, dtype=torch.long, device=device)

    next_obs = torch.tensor(envs.reset(), device=device, dtype=torch.uint8)
    next_done = torch.ones(args.num_envs, device=device, dtype=torch.bool)
    actions = torch.full((args.num_envs,), policy.pad_action_id, device=device, dtype=torch.long)
    next_timestep = torch.zeros(args.num_envs, device=device, dtype=torch.long)

    discount_matrix = torch.tensor(
        [[0 if j < i else args.gamma ** (j - i) for j in range(args.num_steps)]
         for i in range(args.num_steps)],
        dtype=torch.float32, device=device,
    )
    discount_vector = args.gamma ** torch.arange(
        args.num_steps, 0, -1, dtype=torch.float32, device=device
    )

    avg_returns = deque(maxlen=args.avg_returns_last_k)
    global_step = 0
    global_start_time = time.time()
    clip_coef = args.clip_coef

    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))

    for iteration in pbar:
        start_time = time.time()
        global_step_burnin = global_step

        lr_mul = lr_multiplier(iteration, args.num_iterations + 1, args)
        optimizer_pi.param_groups[0]["lr"].copy_(lr_mul * args.learning_rate)
        optimizer_v.param_groups[0]["lr"].copy_(lr_mul * args.learning_rate_vf)

        if args.anneal_clip_coef:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            clip_coef = frac * args.clip_coef

        kv_buf.zero_()
        kv_len.zero_()
        next_done = torch.ones(args.num_envs, device=device, dtype=torch.bool)

        policy_fn = policy_inference.act_kvcache_per_env
        next_obs, next_done, actions, next_timestep, kv_buf, kv_len, container = rollout(
            envs, policy_fn, args, next_obs, next_done, actions, next_timestep,
            kv_buf, kv_len, avg_returns,
        )
        global_step += container.numel()

        rollout_dataset = build_attention_dataset(
            container=container,
            next_obs=next_obs,
            next_done=next_done,
            next_timestep=next_timestep,
            vadv=vadv,
            traj_minibatch_size=64,
        )

        all_targets, all_preds = [], []
        mb_stats = {k: [] for k in ("clipfrac", "v_loss", "pg_loss", "entropy_loss",
                                     "pg_grad_norm", "v_grad_norm")}

        for epoch in range(args.update_epochs):
            for data in iter_traj_batches(rollout_dataset, batch_size=args.batch_size):
                (
                    data_obs, data_actions, data_rewards, old_dist,
                    old_logprobs_action, data_timesteps, old_values,
                    data_last_values, data_splits, data_lmax,
                ) = data

                new_dist, new_logprobs, entropy = policy.evaluate_state_policy(
                    data_obs, data_timesteps, data_actions, splits=data_splits,
                )

                new_values, new_advantages = vadv.evaluate_state(
                    data_obs, data_timesteps, data_actions,
                    old_dist=old_dist, splits=data_splits,
                )

                new_values_split = new_values.flatten().split(data_splits)

                deltas = (
                    data_rewards
                    - new_advantages.gather(dim=1, index=data_actions.long().view(-1, 1)).flatten()
                ).split(data_splits)

                v_loss = compute_value_loss(
                    deltas, new_values_split, data_last_values,
                    discount_matrix, discount_vector,
                )

                with torch.no_grad():
                    y_true, y_pred = dae_targets_and_preds(
                        deltas, new_values_split, data_last_values,
                        discount_matrix, discount_vector,
                    )
                    all_targets.append(y_true)
                    all_preds.append(y_pred)

                old_logprobs = torch.log(old_dist.clamp(min=1e-8))

                adv_for_policy = new_advantages.detach().clone()
                if args.norm_adv:
                    adv_for_policy = normalize_advantage(adv_for_policy, old_dist)

                pg_loss, ratio = compute_policy_loss(
                    args, adv_for_policy, new_logprobs, old_logprobs,
                    data_actions.long().view(-1, 1), clip_range=clip_coef,
                )

                clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
                entropy_loss = entropy.mean()

                kl_loss = (old_dist * (old_logprobs - new_logprobs)).sum(dim=1).mean()

                policy_loss_total = pg_loss - args.ent_coef * entropy_loss + args.kl_coef * kl_loss
                v_loss_total = args.vf_coef * v_loss

                # Policy update
                optimizer_pi.zero_grad()
                policy_loss_total.backward()
                pg_grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer_pi.step()

                # Value update
                optimizer_v.zero_grad()
                v_loss_total.backward()
                v_grad_norm = nn.utils.clip_grad_norm_(vadv.parameters(), args.max_grad_norm_vf)
                optimizer_v.step()

                mb_stats["clipfrac"].append(clipfrac.detach())
                mb_stats["v_loss"].append(v_loss.detach())
                mb_stats["pg_loss"].append(pg_loss.detach())
                mb_stats["entropy_loss"].append(entropy_loss.detach())
                mb_stats["pg_grad_norm"].append(pg_grad_norm.detach())
                mb_stats["v_grad_norm"].append(v_grad_norm.detach())

        if len(all_targets) > 0:
            targets = torch.cat(all_targets)
            preds = torch.cat(all_preds)
            explained_var = explained_variance(y_pred=preds, y_true=targets).item()
        else:
            explained_var = float("nan")

        if iteration % args.log_freq == 0:
            cur_time = time.time()
            speed = (global_step - global_step_burnin) / (cur_time - start_time)

            avg_returns_t = float(np.mean(avg_returns)) if len(avg_returns) > 0 else float("nan")
            update_stats = {k: torch.stack(v).mean().item() for k, v in mb_stats.items()}

            lr_pi = optimizer_pi.param_groups[0]["lr"]
            lr_v = optimizer_v.param_groups[0]["lr"]
            lr_pi = lr_pi.item() if torch.is_tensor(lr_pi) else lr_pi
            lr_v = lr_v.item() if torch.is_tensor(lr_v) else lr_v

            logs = {
                "episode_return": avg_returns_t,
                "logprobs": container["logprobs"].mean().item(),
                "explained_variance": explained_var,
                "clipfrac": update_stats["clipfrac"],
                "policy_loss": update_stats["pg_loss"],
                "value_loss": update_stats["v_loss"],
                "entropy": update_stats["entropy_loss"],
                "pg_grad_norm": update_stats["pg_grad_norm"],
                "v_grad_norm": update_stats["v_grad_norm"],
                "time": time.time() - global_start_time,
            }
            pbar.set_description(
                f"speed: {speed:4.1f} sps, "
                f"returns: {avg_returns_t:4.2f}, "
                f"lr_pi: {lr_pi:.2e}, lr_v: {lr_v:.2e}"
            )

            wandb.log({"speed": speed, "lr_pi": lr_pi, "lr_v": lr_v, **logs}, step=global_step)

    envs.close()