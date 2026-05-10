import os
import argparse
import random
import time
import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import minatar
from gymnasium import spaces
from minatar import Environment
from gymnasium.envs.registration import register




def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment args
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__)[:-len(".py")])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch_deterministic", action="store_true")

    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--track", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default="cleanRL")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--capture_video", action="store_true")

    # Algorithm args
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--total_timesteps", type=int, default=500000)
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--anneal_lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--norm_adv", action="store_true")
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--clip_vloss", action="store_true")
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=None)

    # Runtime filled
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_iterations", type=int, default=0)

    # New added
    parser.add_argument("--running_avg_coef", type=float, default=0.05)
    parser.add_argument("--learning_rate_vf", type=float, default=2.5e-4)
    parser.add_argument("--anneal_lr_vf", action="store_true")
    parser.add_argument("--update_epochs_vf", type=int, default=4)
    parser.add_argument("--full_action", action="store_true")
    parser.add_argument("--kl_coef", type=float, default=0.0)
    parser.add_argument("--anneal_clip_coef", action="store_true")
    parser.add_argument("--gae_lambda", type=float, default=0.95)

    # transformer
    parser.add_argument("--transformer_layers", type=int, default=1)
    parser.add_argument("--transformer_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--traj_minibatch_size", type=int, default=64)

    parser.add_argument("--max_ep_len", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--cnn_feature_dim", type=int, default=32)
    parser.add_argument("--weight_decay", type=float, default=0.00)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)

    parser.add_argument("--lr_schedule", type=str, default="linear", choices=["constant","linear","cosine"])
    parser.add_argument("--warmup_frac", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.0)


    return parser.parse_args()



class BaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "array", "rgb_array"]}

    def __init__(
        self, 
        game, 
        render_mode=None, 
        display_time=50,
        use_minimal_action_set=True, 
        max_steps=500000,
        **kwargs):

        self.render_mode = render_mode
        self.display_time = display_time

        # self.game_name = game
        # self.game_kwargs = kwargs
        # self.game = Environment(env_name=game, **kwargs)
        
        self.game = Environment(env_name=game, **kwargs)

        if use_minimal_action_set:
            self.action_set = self.game.minimal_action_set()
        else:
            self.action_set = list(range(self.game.num_actions()))

        self.action_space = spaces.Discrete(len(self.action_set))
        h, w, c = self.game.state_shape()
        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(c, h, w), dtype=bool
        )
        self.max_steps = max_steps
        self.elapsed_steps = 0


    def step(self, action):
        action = self.action_set[action]
        reward, done = self.game.act(action)
        if self.render_mode == "human":
            self.render()
        
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_steps:
            done = True

        state = self.game.state()          # HWC
        state = np.transpose(state, (2, 0, 1))  # CHW
        return state, reward, done, False, {}
        # return self.game.state(), reward, done, False, {}

    def seed(self, seed=None):
        self.game.seed(seed)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.game.reset()
        if self.render_mode == "human":
            self.render()

        self.elapsed_steps = 0
        
        state = self.game.state()
        state = np.transpose(state, (2, 0, 1))  # CHW
        return state, {}
        # return self.game.state(), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.render_mode == "array":
            return self.game.state()
        elif self.render_mode == "human":
            self.game.display_state(self.display_time)
        elif self.render_mode == "rgb_array": # use the same color palette of Environment.display_state
            state = self.game.state()
            n_channels = state.shape[-1]
            cmap = sns.color_palette("cubehelix", n_channels)
            cmap.insert(0, (0,0,0))
            numerical_state = np.amax(
                state * np.reshape(np.arange(n_channels) + 1, (1,1,-1)), 2)
            rgb_array = np.stack(cmap)[numerical_state]
            return rgb_array

    def close(self):
        if self.game.visualized:
            self.game.close_display()
        return 0



def register_envs():
    for game in ["asterix", "breakout", "freeway", "seaquest", "space_invaders"]:
        name = game.title().replace('_', '')
        register(
            id="MinAtar/{}-v0".format(name),
            entry_point="dae_minatar:BaseEnv",
            kwargs=dict(
                game=game,
                display_time=50,
                use_minimal_action_set=True,
                sticky_action_prob=0,
                difficulty_ramping=False,
                max_steps=500000,
            ),
        )
        register(
            id="MinAtar/{}-v1".format(name),
            entry_point="dae_minatar:BaseEnv",
            kwargs=dict(
                game=game,
                display_time=50,
                use_minimal_action_set=True,
                sticky_action_prob=0,
                difficulty_ramping=False,
                max_steps=500000,
            ),
        )



def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            # env = gym.make(env_id, render_mode="rgb_array")
            # env = make_minatar(env_id)
            env = gym.make(env_id)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
            # env = make_minatar(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class MultiHeadAttentionDecoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads
        assert self.head_size * num_heads == embed_dim

        self.values = nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys   = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries= nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out = nn.Linear(self.num_heads * self.head_size, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None, 
                past_kv: tuple | None = None):
        """
        x: (B, T_new, D) - only the NEW tokens
        past_kv: (past_keys, past_values) each (B, H, T_past, D_h)
        Returns: out, attn, present_kv
        """
        B, T_new, D = x.shape

        # Project to Q, K, V
        q = x.reshape(B, T_new, self.num_heads, self.head_size)
        k = x.reshape(B, T_new, self.num_heads, self.head_size)
        v = x.reshape(B, T_new, self.num_heads, self.head_size)

        q = self.queries(q).transpose(1, 2)  # (B, H, T_new, D_h)
        k = self.keys(k).transpose(1, 2)     # (B, H, T_new, D_h)
        v = self.values(v).transpose(1, 2)   # (B, H, T_new, D_h)

        # Concatenate with past K, V
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # (B, H, T_total, D_h)
            v = torch.cat([past_v, v], dim=2)  # (B, H, T_total, D_h)

        T_total = k.shape[2]

        # Compute attention
        energy = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)  # (B, H, T_new, T_total)

        # Causal mask: each new query can attend to all past + itself
        # For T_new queries and T_total keys
        causal_mask = torch.ones((T_new, T_total), device=x.device, dtype=torch.bool)
        # Query i (from the end) can attend to all keys up to position T_past + i
        T_past = T_total - T_new
        for i in range(T_new):
            causal_mask[i, T_past + i + 1:] = False
        
        energy = energy.masked_fill(~causal_mask[None, None, :, :], float('-inf'))
        
        if pad_mask is not None: 
            energy = energy.masked_fill(~pad_mask[:, None, None, :], float("-inf"))

        attn = torch.softmax(energy, dim=-1)  # (B, H, T_new, T_total)

        attn = self.attn_dropout(attn)
        
        out = torch.matmul(attn, v)  # (B, H, T_new, D_h)
        out = out.transpose(1, 2).reshape(B, T_new, self.embed_dim)
        out = self.fc_out(out)
        out = self.resid_dropout(out)

        # Return current K, V for next step's cache
        present_kv = (k, v)
        
        return out, attn, present_kv


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttentionDecoder(dim, num_heads, dropout)
        self.input_layernorm = nn.LayerNorm(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.fc_projection = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None, past_kv: tuple | None = None):
        hidden_states = self.input_layernorm(x)
        attn_out, attn_w, present_kv = self.attention(hidden_states, pad_mask=pad_mask, past_kv=past_kv)
        hidden_states = self.dropout(attn_out) + x

        x_ = self.post_attention_layernorm(hidden_states)
        ff = self.fc_projection(x_)
        out = ff + hidden_states
        
        return out, attn_w, present_kv



class Transformer(nn.Module):
    def __init__(self, num_layers: int, dim: int, num_heads: int, max_len: int, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.num_layers = num_layers
        self.layers = nn.ModuleList([DecoderLayer(dim, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None, past_kvs: list | None = None):
        """
        x: (B, T_new, D)
        past_kvs: list of tuples, one per layer
        Returns: (B, T_new, D), attn_weights, present_kvs
        """
        B, T, D = x.shape

        if past_kvs is None:
            past_kvs = [None] * self.num_layers

        attn_weights = []
        present_kvs = []
        
        for i, layer in enumerate(self.layers):
            x, attn, present_kv = layer(x, pad_mask=pad_mask, past_kv=past_kvs[i])
            attn_weights.append(attn)
            present_kvs.append(present_kv)

        return x, present_kvs

# --------- KV cache helpers ---------

def pack_kvs_padded(kvs_list, idx, num_layers, device):
    """Pack KV caches from multiple environments into batched tensors with padding."""
    idx_list = idx.tolist()
    B = len(idx_list)
    assert all(kvs_list[i] is not None for i in idx_list)

    Tpast_list = [kvs_list[i][0][0].shape[2] for i in idx_list]
    Tpast_max = max(Tpast_list)

    past_mask = torch.zeros((B, Tpast_max), device=device, dtype=torch.bool)
    past_kvs = []

    for l in range(num_layers):
        ks, vs = [], []
        for b, i in enumerate(idx_list):
            k, v = kvs_list[i][l]  # (1,H,T,Dh)
            Ti = k.shape[2]
            past_mask[b, :Ti] = True
            if Ti < Tpast_max:
                pad_len = Tpast_max - Ti
                k = torch.cat([k, torch.zeros((1, k.shape[1], pad_len, k.shape[3]), device=device, dtype=k.dtype)], dim=2)
                v = torch.cat([v, torch.zeros((1, v.shape[1], pad_len, v.shape[3]), device=device, dtype=v.dtype)], dim=2)
            ks.append(k)
            vs.append(v)
        past_kvs.append((torch.cat(ks, dim=0), torch.cat(vs, dim=0)))
    return past_kvs, past_mask


def unpack_kvs(present_kvs, idx, kvs_list):
    """Unpack batched KV caches back to per-environment list."""
    idx = idx.tolist()
    for b, e in enumerate(idx):
        per_env = []
        for (k, v) in present_kvs:
            per_env.append((k[b:b+1].contiguous(), v[b:b+1].contiguous()))
        kvs_list[e] = per_env


def deep_copy_kvs(kvs):
    """Deep copy KV cache for an environment."""
    if kvs is None:
        return None
    return [(k.clone(), v.clone()) for k, v in kvs]


def slice_flat_by_traj(flat_obs, flat_act, flat_ts, splits, traj_ids, device):
    """
    Extract a sub-batch containing only trajectories in traj_ids.
    Returns sub_flat_obs/sub_flat_act/sub_flat_ts/sub_splits.
    """
    splits = [int(x) for x in splits]
    # prefix sums to map traj -> [start, end)
    ends = np.cumsum(splits)
    starts = np.concatenate([[0], ends[:-1]])

    idx_ranges = [(starts[j], ends[j]) for j in traj_ids]
    total = sum(e - s for s, e in idx_ranges)

    # build packed sub-flat by concatenation
    obs_list, act_list, ts_list = [], [], []
    sub_splits = []
    for (s, e) in idx_ranges:
        obs_list.append(flat_obs[s:e])
        act_list.append(flat_act[s:e])
        ts_list.append(flat_ts[s:e])
        sub_splits.append(e - s)

    sub_flat_obs = torch.cat(obs_list, dim=0)
    sub_flat_act = torch.cat(act_list, dim=0)
    sub_flat_ts  = torch.cat(ts_list,  dim=0)

    return sub_flat_obs, sub_flat_act, sub_flat_ts, sub_splits



class TransformerAgent(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        obs_shape = envs.single_observation_space.shape  # (C,H,W) for MinAtar
        n_actions = envs.single_action_space.n
        n_input_channels = obs_shape[0]

        self.dim = args.transformer_dim
        self.pad_action_id = n_actions
        self.max_ep_len = args.max_ep_len

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, args.cnn_feature_dim, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(args.cnn_feature_dim, args.cnn_feature_dim, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1,) + obs_shape)
            flat_dim = self.cnn(dummy).shape[-1]

        self.state_proj = nn.Sequential(nn.Linear(flat_dim, self.dim), nn.ReLU())
        self.embed_action = nn.Embedding(n_actions + 1, self.dim)
        self.embed_timestep = nn.Embedding(self.max_ep_len, self.dim)

        self.transformer = Transformer(
            num_layers=args.transformer_layers,
            dim=self.dim,
            num_heads=args.num_heads,
            max_len=3*args.max_ep_len+1,
            dropout=args.dropout,
        )

        self.pi = layer_init(nn.Linear(self.dim, n_actions), std=0.01)
        self.v  = layer_init(nn.Linear(self.dim, 1), std=1.0)

    def embed_state(self, obs):
        z = self.cnn(obs)
        return self.state_proj(z)

    @torch.no_grad()
    def act_kvcache_per_env(self, obs_t, ts_t, prev_action, is_new_ep, kvs_list):
        device = obs_t.device
        N = obs_t.shape[0]
        num_layers = self.transformer.num_layers

        action  = torch.empty((N,), device=device, dtype=torch.long)
        logprob = torch.empty((N,), device=device, dtype=torch.float32)
        # entropy = torch.empty((N,), device=device, dtype=torch.float32)
        # value   = torch.empty((N,), device=device, dtype=torch.float32)

        idx_all = torch.arange(N, device=device)
        new_idx  = idx_all[is_new_ep]
        cont_idx = idx_all[~is_new_ep]

        def step_tokens(x_new, idx, past_kvs, past_mask=None):
            # x_new: (B, Tnew, D)
            B, Tnew, _ = x_new.shape
            if past_mask is None:
                pad_mask_total = None
            else:
                new_mask = torch.ones((B, Tnew), device=device, dtype=torch.bool)
                pad_mask_total = torch.cat([past_mask, new_mask], dim=1)
            latent, present = self.transformer(x_new, pad_mask=pad_mask_total, past_kvs=past_kvs)
            unpack_kvs(present, idx, kvs_list)
            return latent[:, -1, :]  # last token (S_t)

        # ---------- continuing envs: append [A_{t-1}, S_t] ----------
        if cont_idx.numel() > 0:
            past_kvs, past_mask = pack_kvs_padded(kvs_list, cont_idx, num_layers, device)

            obs_c = obs_t.index_select(0, cont_idx)
            ts_c  = ts_t.index_select(0, cont_idx).long()
            pa_c  = prev_action.index_select(0, cont_idx).long()

            ts_s = ts_c.clamp(max=self.max_ep_len - 1)
            ts_a = (ts_c - 1).clamp(min=0, max=self.max_ep_len - 1)

            a_prev = self.embed_action(pa_c) + self.embed_timestep(ts_a)
            s_tok  = self.embed_state(obs_c) + self.embed_timestep(ts_s)

            x_new = torch.stack([a_prev, s_tok], dim=1)  # (B,2,D)

            h_s = step_tokens(x_new, cont_idx, past_kvs, past_mask=past_mask)

            logits = self.pi(h_s)
            dist = Categorical(logits=logits)
            act = dist.sample()

            action[cont_idx]  = act
            logprob[cont_idx] = dist.log_prob(act)
            # entropy[cont_idx] = dist.entropy()
            # value[cont_idx]   = self.v(h_s).squeeze(-1)

        # ---------- new episodes: start cache empty, append [PAD, S_0] ----------
        if new_idx.numel() > 0:
            obs_n = obs_t.index_select(0, new_idx)
            ts_n  = ts_t.index_select(0, new_idx).long()
            pa_n  = prev_action.index_select(0, new_idx).long()  # should be PAD

            ts_s = ts_n.clamp(max=self.max_ep_len - 1)
            ts_a = (ts_n - 1).clamp(min=0, max=self.max_ep_len - 1)

            a_prev = self.embed_action(pa_n) + self.embed_timestep(ts_a)
            s_tok  = self.embed_state(obs_n) + self.embed_timestep(ts_s)

            x_new = torch.stack([a_prev, s_tok], dim=1)  # (B,2,D)
            past_empty = [None] * num_layers

            latent, present = self.transformer(x_new, pad_mask=None, past_kvs=past_empty)
            h_s = latent[:, -1, :]
            unpack_kvs(present, new_idx, kvs_list)

            logits = self.pi(h_s)
            dist = Categorical(logits=logits)
            act = dist.sample()

            action[new_idx]  = act
            logprob[new_idx] = dist.log_prob(act)
            # entropy[new_idx] = dist.entropy()
            # value[new_idx]   = self.v(h_s).squeeze(-1)

        return action, logprob #, entropy, value


    @torch.no_grad()
    def evaluate_values_only_onebatch(self, flat_obs, flat_actions, flat_ts, splits):

        device = flat_obs.device
        splits = [int(x) for x in splits]
        B = len(splits)
        Lmax = max(splits)
        M = int(sum(splits))

        # ---- 1) CNN on packed observations ONLY ----
        # flat_obs: (M,C,H,W) -> s_flat: (M,D)
        s_flat = self.embed_state(flat_obs)  # (M, dim)

        # ---- 2) Pad embeddings + actions/timesteps (cheap) ----
        s_b = torch.zeros((B, Lmax, self.dim), device=device, dtype=s_flat.dtype)
        act_b = torch.full((B, Lmax), self.pad_action_id, device=device, dtype=torch.long)
        ts_b  = torch.zeros((B, Lmax), device=device, dtype=torch.long)
        mask  = torch.zeros((B, Lmax), device=device, dtype=torch.bool)

        cursor = 0
        for b, L in enumerate(splits):
            L = int(L)
            s_b[b, :L] = s_flat[cursor:cursor+L]
            act_b[b, :L] = flat_actions[cursor:cursor+L].long()
            ts_b[b, :L]  = flat_ts[cursor:cursor+L].long()
            mask[b, :L]  = True
            cursor += L

        # prev actions A_{t-1} (PAD at t=0)
        prev_act_b = torch.full((B, Lmax), self.pad_action_id, device=device, dtype=torch.long)
        prev_act_b[:, 1:] = act_b[:, :-1]

        # timestep embeddings (clip like your other path)
        ts_s = ts_b.clamp(max=self.max_ep_len - 1)
        ts_a = (ts_b - 1).clamp(min=0, max=self.max_ep_len - 1)

        s_b = s_b + self.embed_timestep(ts_s)                  # (B,L,D)
        a_prev = self.embed_action(prev_act_b) + self.embed_timestep(ts_a)  # (B,L,D)

        # interleave [A_{t-1}, S_t] => (B,2L,D)
        x = torch.stack([a_prev, s_b], dim=2).reshape(B, 2 * Lmax, self.dim)
        tok_mask = torch.stack([mask, mask], dim=2).reshape(B, 2 * Lmax)

        # ---- 3) Transformer + value head ----
        latent, _ = self.transformer(x, pad_mask=tok_mask, past_kvs=None)
        h_s = latent[:, 1::2, :]              # state tokens (B,L,D)
        v_b = self.v(h_s).squeeze(-1)         # (B,L)

        # ---- 4) Unpad back to flat order ----
        out = []
        for b, L in enumerate(splits):
            out.append(v_b[b, :L])
        return torch.cat(out, dim=0)          # (M,)


    @torch.no_grad()
    def evaluate_values_only(self, flat_obs, flat_actions, flat_ts, splits,
                            traj_mb_size=64):
        num_traj = len(splits)
        outs = []

        for i in range(0, num_traj, traj_mb_size):
            traj_ids = list(range(i, min(i + traj_mb_size, num_traj)))

            sub_obs, sub_act, sub_ts, sub_splits = slice_flat_by_traj(
                flat_obs, flat_actions, flat_ts, splits, traj_ids, flat_obs.device
            )

            sub_v = self.evaluate_values_only_onebatch(
                sub_obs, sub_act, sub_ts, sub_splits
            )

            outs.append(sub_v)

        return torch.cat(outs, dim=0)




    def evaluate_actions(self, flat_obs, flat_actions, flat_ts, splits):
        device = flat_obs.device
        B = len(splits)
        Lmax = max(map(int, splits))
        C, H, W = flat_obs.shape[1:]

        obs_b = torch.zeros((B, Lmax, C, H, W), device=device, dtype=flat_obs.dtype)
        act_b = torch.full((B, Lmax), self.pad_action_id, device=device, dtype=torch.long)  # A_t
        ts_b  = torch.zeros((B, Lmax), device=device, dtype=torch.long)
        mask  = torch.zeros((B, Lmax), device=device, dtype=torch.bool)

        cursor = 0
        for b, L in enumerate(splits):
            L = int(L)
            obs_b[b, :L] = flat_obs[cursor:cursor+L]
            act_b[b, :L] = flat_actions[cursor:cursor+L].long()
            ts_b[b, :L]  = flat_ts[cursor:cursor+L].long()
            mask[b, :L]  = True
            cursor += L

        # build prev actions: A_{t-1}, with PAD at t=0
        prev_act_b = torch.full((B, Lmax), self.pad_action_id, device=device, dtype=torch.long)
        prev_act_b[:, 1:] = act_b[:, :-1]

        ts_s = ts_b.clamp(max=self.max_ep_len - 1)
        ts_a = (ts_b - 1).clamp(min=0, max=self.max_ep_len - 1)

        s = self.embed_state(obs_b.reshape(B*Lmax, C, H, W)).reshape(B, Lmax, self.dim)
        s = s + self.embed_timestep(ts_s)

        a_prev = self.embed_action(prev_act_b) + self.embed_timestep(ts_a)

        # interleave [A_{t-1}, S_t]
        x = torch.stack([a_prev, s], dim=2).reshape(B, 2*Lmax, self.dim)
        tok_mask = torch.stack([mask, mask], dim=2).reshape(B, 2*Lmax)

        latent, _ = self.transformer(x, pad_mask=tok_mask, past_kvs=None)

        # state tokens are at positions 1,3,5,... => 1::2
        h_s = latent[:, 1::2, :]  # (B,L,D)

        logits = self.pi(h_s)
        dist = Categorical(logits=logits)

        # logprob of actual action A_t (not prev)
        new_logprob = dist.log_prob(act_b.clamp(max=self.pad_action_id - 1))
        ent = dist.entropy()
        val = self.v(h_s).squeeze(-1)

        out_lp, out_ent, out_v = [], [], []
        for b, L in enumerate(splits):
            L = int(L)
            out_lp.append(new_logprob[b, :L])
            out_ent.append(ent[b, :L])
            out_v.append(val[b, :L])
        return torch.cat(out_lp, 0), torch.cat(out_ent, 0), torch.cat(out_v, 0)



def flatten_with_bootstrap_ppo(
    obs, actions, timesteps,
    dones_after, next_obs, next_done, next_timesteps,
    pad_action_id: int,
):
    """
    Build flat arrays of variable-length trajectories split by dones_after.
    For unfinished tail segments, append ONE bootstrap token (next_obs, PAD action, next_timestep).

    Returns:
      flat_obs, flat_act, flat_ts, splits,
      flat_t, flat_env, is_boot, boot_pos
    """
    T, N = dones_after.shape
    device = obs.device

    flat_obs, flat_act, flat_ts = [], [], []
    flat_t, flat_env, is_boot = [], [], []
    splits = []

    boot_pos = torch.full((N,), -1, device=device, dtype=torch.long)
    cursor = 0

    for env in range(N):
        start = 0
        done_idx = torch.nonzero(dones_after[:, env] > 0.5, as_tuple=False).flatten().tolist()

        # complete episodes inside rollout
        for t in done_idx:
            end = t + 1
            if end > start:
                L = end - start
                flat_obs.append(obs[start:end, env])
                flat_act.append(actions[start:end, env].long())
                flat_ts.append(timesteps[start:end, env].long())

                flat_t.append(torch.arange(start, end, device=device, dtype=torch.long))
                flat_env.append(torch.full((L,), env, device=device, dtype=torch.long))
                is_boot.append(torch.zeros((L,), device=device, dtype=torch.bool))

                splits.append(L)
                cursor += L
            start = end

        # unfinished tail => append bootstrap token
        if start < T:
            end = T
            L = end - start

            # rollout tail
            flat_obs.append(obs[start:end, env])
            flat_act.append(actions[start:end, env].long())
            flat_ts.append(timesteps[start:end, env].long())

            flat_t.append(torch.arange(start, end, device=device, dtype=torch.long))
            flat_env.append(torch.full((L,), env, device=device, dtype=torch.long))
            is_boot.append(torch.zeros((L,), device=device, dtype=torch.bool))

            # bootstrap token (next_obs)
            flat_obs.append(next_obs[env].unsqueeze(0))
            flat_act.append(torch.tensor([pad_action_id], device=device, dtype=torch.long))
            flat_ts.append(next_timesteps[env].unsqueeze(0).long())

            flat_t.append(torch.full((1,), -1, device=device, dtype=torch.long))
            flat_env.append(torch.full((1,), env, device=device, dtype=torch.long))
            is_boot.append(torch.ones((1,), device=device, dtype=torch.bool))

            splits.append(L + 1)
            boot_pos[env] = cursor + L
            cursor += (L + 1)

    flat_obs = torch.cat(flat_obs, dim=0)
    flat_act = torch.cat(flat_act, dim=0)
    flat_ts  = torch.cat(flat_ts, dim=0)
    flat_t   = torch.cat(flat_t, dim=0)
    flat_env = torch.cat(flat_env, dim=0)
    is_boot  = torch.cat(is_boot, dim=0)

    return flat_obs, flat_act, flat_ts, splits, flat_t, flat_env, is_boot, boot_pos



class TrajectoryBufferPPO:
    def build(self, obs, actions, logprobs, rewards, dones_after, values, timesteps, next_value, next_done):
        """
        Build trajectory buffer from rollout data.
        
        Splits the rollout into variable-length segments (trajectories) based on episode boundaries.
        Each segment represents one complete episode or a partial episode (if episode didn't finish).
        
        Args:
            obs: (T, N, ...) observations
            actions: (T, N) actions taken
            logprobs: (T, N) log probabilities
            rewards: (T, N) rewards received
            dones_after: (T, N) done flags after action
            values: (T, N) value estimates
            timesteps: (T, N) timestep indices
            next_value: (N,) bootstrap values for incomplete episodes
            next_done: (N,) done flags for next state
        """
        T, N = dones_after.shape
        device = obs.device

        obs_list, act_list, logp_list, rew_list, val_list, ts_list = [], [], [], [], [], []
        lengths, last_values = [], []

        for e in range(N):
            start = 0
            done_idx = torch.nonzero(dones_after[:, e] > 0.5, as_tuple=False).flatten().tolist()
            
            # Process each complete episode
            for t in done_idx:
                end = t + 1
                if end > start:
                    obs_list.append(obs[start:end, e])
                    act_list.append(actions[start:end, e])
                    logp_list.append(logprobs[start:end, e])
                    rew_list.append(rewards[start:end, e])
                    val_list.append(values[start:end, e])
                    ts_list.append(timesteps[start:end, e])
                    lengths.append(end - start)
                    # Terminal state: no bootstrap value
                    last_values.append(torch.tensor(0.0, device=device))
                start = end

            # Handle remaining partial episode (if any)
            if start < T:
                end = T
                obs_list.append(obs[start:end, e])
                act_list.append(actions[start:end, e])
                logp_list.append(logprobs[start:end, e])
                rew_list.append(rewards[start:end, e])
                val_list.append(values[start:end, e])
                ts_list.append(timesteps[start:end, e])
                lengths.append(end - start)
                # Bootstrap if not done at end of rollout
                last_values.append(next_value[e] * (1.0 - next_done[e]))

        self.obs = torch.cat(obs_list, 0)
        self.actions = torch.cat(act_list, 0)
        self.logprobs = torch.cat(logp_list, 0)
        self.rewards = torch.cat(rew_list, 0)
        self.values = torch.cat(val_list, 0)
        self.timesteps = torch.cat(ts_list, 0)

        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.ends = np.cumsum(self.lengths)
        self.starts = np.insert(self.ends, 0, 0)[:-1]
        self.last_values = torch.stack(last_values, 0)

    def iter_traj_minibatches(self, traj_mb_size=64):
        """Iterate over random minibatches of trajectories."""
        idxs = np.random.permutation(len(self.lengths))
        for i in range(0, len(idxs), traj_mb_size):
            pick = idxs[i:i+traj_mb_size]
            obs, act, logp, rew, val, ts, lasts, splits = [], [], [], [], [], [], [], []
            for j in pick:
                s, e = self.starts[j], self.ends[j]
                obs.append(self.obs[s:e])
                act.append(self.actions[s:e])
                logp.append(self.logprobs[s:e])
                rew.append(self.rewards[s:e])
                val.append(self.values[s:e])
                ts.append(self.timesteps[s:e])
                lasts.append(self.last_values[j])
                splits.append(e - s)
            yield (
                torch.cat(obs, 0),
                torch.cat(act, 0),
                torch.cat(logp, 0),
                torch.cat(rew, 0),
                torch.cat(val, 0),
                torch.cat(ts, 0),
                lasts,          # list of 0-d tensors
                splits,         # list[int]
            )



def gae_for_splits(rewards_flat, values_flat, lasts, splits, gamma, lam):
    adv_chunks = []
    ret_chunks = []
    cursor = 0
    for L, last_v in zip(splits, lasts):
        L = int(L)
        r = rewards_flat[cursor:cursor+L]
        v = values_flat[cursor:cursor+L]
        adv = torch.zeros_like(r)
        lastgaelam = 0.0
        for t in reversed(range(L)):
            next_v = last_v if t == L - 1 else v[t + 1]
            delta = r[t] + gamma * next_v - v[t]
            lastgaelam = delta + gamma * lam * lastgaelam
            adv[t] = lastgaelam
        ret = adv + v
        adv_chunks.append(adv)
        ret_chunks.append(ret)
        cursor += L
    return torch.cat(adv_chunks, 0), torch.cat(ret_chunks, 0)



def explained_variance(y_pred, y_true, eps=1e-8):
    """Compute explained variance: 1 - Var(y - y_pred) / Var(y)"""
    var_y = torch.var(y_true)
    if var_y.item() < eps:
        return torch.tensor(float("nan"), device=y_true.device)
    return 1.0 - torch.var(y_true - y_pred) / (var_y + eps)


def lr_decay(progress, args):
    progress = float(np.clip(progress, 0.0, 1.0))
    if args.lr_schedule == "constant":
        decay_ratio = 1.0
    if args.lr_schedule == "linear":
        decay_ratio = 1.0 - progress
    if args.lr_schedule == "cosine":
        decay_ratio = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return decay_ratio


def lr_multiplier(step_idx, total_steps, args):
    warmup_steps = int(total_steps * args.warmup_frac)
    # step_idx = max(0, min(step_idx, total_steps - 1))

    if warmup_steps > 0 and step_idx < warmup_steps:
        w = step_idx / warmup_steps
        return w
    
    rem = float(max(1, total_steps - warmup_steps))
    post = step_idx - warmup_steps
    progress = post / rem

    decay_ratio = lr_decay(progress, args)
    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * decay_ratio



if __name__ == "__main__":
    args = parse_args()
    args.num_iterations = args.total_timesteps // int(args.num_envs * args.num_steps)
    run_name = f"{args.exp_name}_seed_{args.seed}_layers_{args.transformer_layers}_dim_{args.transformer_dim}_heads_{args.num_heads}_dropout_{args.dropout}_cnn_dim_{args.cnn_feature_dim}_schedule_{args.lr_schedule}"
    print(args)
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment setup
    register_envs()
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    pad_action_id = envs.single_action_space.n

    # Agent and optimizer
    agent = TransformerAgent(envs, args).to(device)
    optimizer = torch.optim.Adam(
        agent.parameters(), 
        lr=args.learning_rate, 
        eps=1e-8, 
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2))

    # Storage buffers
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_after = torch.zeros((args.num_steps, args.num_envs)).to(device)
    timesteps = torch.zeros((args.num_steps, args.num_envs), device=device, dtype=torch.long)

    # Initialize environment
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_timesteps = torch.zeros(args.num_envs, device=device, dtype=torch.long)

    running_avg_reward = None
    clip_coef = args.clip_coef

    for iteration in range(1, args.num_iterations + 1):
        # Learning rate annealing
        lr_mul = lr_multiplier(iteration, args.num_iterations + 1, args)
        lrnow = lr_mul * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

        if args.anneal_clip_coef:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            clip_coef = frac * args.clip_coef

        # Initialize KV caches for rollout
        kvs_list = [None] * args.num_envs
        prev_action = torch.full((args.num_envs,), pad_action_id, device=device, dtype=torch.long)
        is_new_ep = torch.ones(args.num_envs, device=device, dtype=torch.bool)

        # === ROLLOUT PHASE ===
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            timesteps[step] = next_timesteps

            # Sample action using KV cache
            with torch.no_grad():
                # action, logprob, entropy, value = agent.act_kvcache_per_env(
                action, logprob = agent.act_kvcache_per_env(
                    obs_t=next_obs,
                    ts_t=next_timesteps,
                    prev_action=prev_action,
                    is_new_ep=is_new_ep,
                    kvs_list=kvs_list,
                )
            
            actions[step]  = action
            logprobs[step] = logprob
            # values[step]   = value
            
            # Execute action in environment
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)

            prev_action = action.clone()
            dones_after[step] = torch.as_tensor(next_done, device=device, dtype=torch.float32)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(next_done).to(device)

            # Update timesteps
            done_mask = next_done.bool()
            next_timesteps += 1
            next_timesteps = torch.clamp(next_timesteps, max=args.max_ep_len - 1)
            next_timesteps[done_mask] = 0

            # Reset KV caches for finished episodes
            for e in torch.nonzero(done_mask, as_tuple=False).flatten().tolist():
                kvs_list[e] = None
            prev_action[done_mask] = pad_action_id
            is_new_ep = done_mask
            
            # Log episode statistics
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        if running_avg_reward is None:
                            running_avg_reward = info["episode"]["r"]
                        else:
                            running_avg_reward = args.running_avg_coef * info["episode"]["r"] + (1.0 - args.running_avg_coef) * running_avg_reward
                        if args.track:
                            wandb.log(
                                {
                                    "global_step": global_step,
                                    "charts/running_avg_episodic_return": running_avg_reward,
                                    "charts/episodic_return": info["episode"]["r"],
                                    "charts/episodic_length": info["episode"]["l"],
                                },
                                step=global_step,
                            )
        

        with torch.no_grad():
            flat_obs, flat_act, flat_ts, splits, flat_t, flat_env, is_boot, boot_pos = flatten_with_bootstrap_ppo(
                obs=obs,
                actions=actions,
                timesteps=timesteps,
                dones_after=dones_after,
                next_obs=next_obs,
                next_done=next_done,
                next_timesteps=next_timesteps,
                pad_action_id=pad_action_id,
            )

            # One batched forward for all values (including bootstrap tokens)
            flat_v = agent.evaluate_values_only(
                flat_obs=flat_obs,
                flat_actions=flat_act,
                flat_ts=flat_ts,
                splits=splits,
                traj_mb_size=args.traj_minibatch_size,
            )

            # Fill rollout values[t, env] from flat_v (excluding bootstrap tokens)
            values.zero_()
            rollout_mask = ~is_boot
            t_idx = flat_t[rollout_mask]
            e_idx = flat_env[rollout_mask]
            values[t_idx, e_idx] = flat_v[rollout_mask]

            # next_value per env from bootstrap positions
            next_value = torch.zeros(args.num_envs, device=device)
            has_boot = boot_pos >= 0
            next_value[has_boot] = flat_v[boot_pos[has_boot]]

            # if env actually ended at the very end, kill bootstrap
            next_value = next_value * (1.0 - next_done.float())


        # === BUILD TRAJECTORY BUFFER ===
        trajbuf = TrajectoryBufferPPO()
        trajbuf.build(
            obs=obs,
            actions=actions,
            logprobs=logprobs,
            rewards=rewards,
            dones_after=dones_after,
            values=values,
            timesteps=timesteps,
            next_value=next_value,
            next_done=next_done,
        )

        # === UPDATE PHASE ===
        clipfracs = []

        for epoch in range(args.update_epochs):
            for data in trajbuf.iter_traj_minibatches(traj_mb_size=args.traj_minibatch_size):
                (data_obs, data_act, old_logp, data_rew, old_val, data_ts, lasts, splits) = data

                # Compute GAE using OLD values
                adv, ret = gae_for_splits(
                    rewards_flat=data_rew,
                    values_flat=old_val,
                    lasts=lasts,
                    splits=splits,
                    gamma=args.gamma,
                    lam=args.gae_lambda,
                )

                # Normalize advantages
                if args.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Re-evaluate with transformer (full sequence)
                new_logp, ent, new_v = agent.evaluate_actions(
                    flat_obs=data_obs,
                    flat_actions=data_act,
                    flat_ts=data_ts,
                    splits=splits,
                )

                # PPO loss computation
                logratio = new_logp - old_logp
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())
                    approx_kl = ((ratio - 1) - logratio).mean()

                # Policy loss (clipped)
                pg_loss1 = -adv * ratio
                pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (optionally clipped)
                if args.clip_vloss:
                    v_unclipped = (new_v - ret) ** 2
                    v_clipped = old_val + torch.clamp(new_v - old_val, -clip_coef, clip_coef)
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - ret) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_v - ret) ** 2).mean()

                entropy_loss = ent.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                # Optimize
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Compute explained variance
        with torch.no_grad():
            y_true = ret
            y_pred = new_v
            var_y = torch.var(y_true)
            explained_var = float("nan") if var_y.item() < 1e-8 else (1 - torch.var(y_true - y_pred) / (var_y + 1e-8)).item()

        if args.track:
            wandb.log(
                {
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "losses/grad_norm": grad_norm.item(),
                    "charts/clip_coef": clip_coef,
                    "charts/learning_rate": lrnow,
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                },
                step=global_step,
            )

    envs.close()
    writer.close()