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
    parser.add_argument("--max_grad_norm_vf", type=float, default=1.0)
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


    # transformer
    parser.add_argument("--pi_transformer_layers", type=int, default=1)
    parser.add_argument("--pi_transformer_dim", type=int, default=128)
    parser.add_argument("--v_dim", type=int, default=128)
    parser.add_argument("--pi_num_heads", type=int, default=2)

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
            entry_point="da2e_separate_minatar_sa:BaseEnv",
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
            entry_point="da2e_separate_minatar_sa:BaseEnv",
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


def pack_kvs_padded(kvs_list, idx, num_layers, device):
    """
    Returns:
      past_kvs: list[num_layers] of (k,v) with shape (B,H,Tpast_max,Dh)
      past_mask: (B, Tpast_max) True for real past positions, False for padded
    """
    idx_list = idx.tolist()
    B = len(idx_list)

    # all must have cache in this group
    assert all(kvs_list[i] is not None for i in idx_list)

    # find max past length (use layer0 as reference)
    Tpast_list = [kvs_list[i][0][0].shape[2] for i in idx_list]  # k shape: (1,H,T,Dh)
    Tpast_max = max(Tpast_list)

    past_mask = torch.zeros((B, Tpast_max), device=device, dtype=torch.bool)
    past_kvs = []

    for l in range(num_layers):
        ks, vs = [], []
        for b, i in enumerate(idx_list):
            k, v = kvs_list[i][l]          # (1,H,T_i,Dh)
            Ti = k.shape[2]
            past_mask[b, :Ti] = True

            if Ti < Tpast_max:
                pad_len = Tpast_max - Ti
                k_pad = torch.zeros((1, k.shape[1], pad_len, k.shape[3]), device=device, dtype=k.dtype)
                v_pad = torch.zeros((1, v.shape[1], pad_len, v.shape[3]), device=device, dtype=v.dtype)
                k = torch.cat([k, k_pad], dim=2)
                v = torch.cat([v, v_pad], dim=2)

            ks.append(k)  # (1,H,Tpast_max,Dh)
            vs.append(v)

        past_kvs.append((torch.cat(ks, dim=0), torch.cat(vs, dim=0)))  # (B,H,Tpast_max,Dh)

    return past_kvs, past_mask, Tpast_max



def unpack_kvs(present_kvs, idx, kvs_list):
    """
    present_kvs: list[num_layers] of (k,v) with batch=len(idx)
    idx: 1D LongTensor of env indices
    Writes back into kvs_list as per-env caches with batch=1.
    """
    idx = idx.tolist()
    B = len(idx)
    for b, e in enumerate(idx):
        per_env = []
        for (k, v) in present_kvs:
            per_env.append((k[b:b+1].contiguous(), v[b:b+1].contiguous()))
        kvs_list[e] = per_env



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

        return x, attn_weights, present_kvs


class DTBackbone(nn.Module):
    def __init__(self, obs_dim, n_actions, transformer_dim, transformer_layers, num_heads, args):
        super().__init__()
        C, H, W = obs_dim
        D = transformer_dim

        cnn = nn.Sequential(
            nn.Conv2d(C, args.cnn_feature_dim, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(args.cnn_feature_dim, args.cnn_feature_dim, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros((1, C, H, W))
            flat_dim = cnn(dummy).shape[-1]

        proj = nn.Sequential(nn.Linear(flat_dim, D), nn.ReLU())

        self.embed_state = nn.Sequential(cnn, proj)
        self.pad_action_id = n_actions
        self.embed_action = nn.Embedding(n_actions + 1, D)
        self.embed_timestep = nn.Embedding(args.max_ep_len, D)

        self.transformer = Transformer(
            num_layers=transformer_layers,
            dim=D,
            num_heads=num_heads,
            max_len=2 * args.max_ep_len + 1,
            dropout=args.dropout,
        )
        self.D = D

    def encode_tokens(self, obs, act, ts):
        """
        obs: (B,L,C,H,W)
        act: (B,L) long
        ts : (B,L) long
        Returns x_full: (B, 2L, D) = [S0,A0,S1,A1,...]
        """
        B, L, C, H, W = obs.shape
        D = self.D

        obs_flat = obs.reshape(B * L, C, H, W)
        s = self.embed_state(obs_flat).reshape(B, L, D)
        a = self.embed_action(act)
        t = self.embed_timestep(ts.clamp(max=self.embed_timestep.num_embeddings - 1))

        s = s + t
        a = a + t
        return torch.stack([s, a], dim=2).reshape(B, 2 * L, D)


class PolicyNet(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        n_actions = envs.single_action_space.n
        obs_dim = envs.single_observation_space.shape

        self.backbone = DTBackbone(obs_dim, n_actions, args.pi_transformer_dim, args.pi_transformer_layers, args.pi_num_heads, args)
        self.action_net = layer_init(nn.Linear(self.backbone.D, n_actions), std=0.01)

    @property
    def pad_action_id(self):
        return self.backbone.pad_action_id

    def act_kvcache_per_env(self, obs_t, ts_t, prev_action, is_new_ep, kvs_list):
        """
        Same logic as your current Agent.act_kvcache_per_env,
        but use self.backbone.embed_* and self.backbone.transformer,
        and head is self.action_net.
        """
        device = obs_t.device
        N = obs_t.shape[0]
        D = self.backbone.D
        A = self.action_net.out_features
        num_layers = self.backbone.transformer.num_layers

        action  = torch.empty((N,), device=device, dtype=torch.long)
        logprob = torch.empty((N,), device=device, dtype=torch.float32)
        probs   = torch.empty((N, A), device=device, dtype=torch.float32)

        new_idx  = torch.nonzero(is_new_ep, as_tuple=False).squeeze(-1)
        cont_idx = torch.nonzero(~is_new_ep, as_tuple=False).squeeze(-1)

        if cont_idx.numel() > 0:
            past_kvs, past_mask, Tpast_max = pack_kvs_padded(
                kvs_list, cont_idx, num_layers, device=device
            )

            obs_c = obs_t.index_select(0, cont_idx)
            ts_c  = ts_t.index_select(0, cont_idx)
            pa_c  = prev_action.index_select(0, cont_idx)

            ts_a = (ts_c - 1).clamp(min=0)

            s = self.backbone.embed_state(obs_c)
            a = self.backbone.embed_action(pa_c)
            t = self.backbone.embed_timestep(ts_c.clamp(max=self.backbone.embed_timestep.num_embeddings-1))
            t_a = self.backbone.embed_timestep(ts_a)

            a = a + t_a
            s = s + t

            x_new = torch.stack([a, s], dim=1)  # (B,2,D)

            B = cont_idx.numel()
            new_mask = torch.ones((B, x_new.shape[1]), device=device, dtype=torch.bool)
            pad_mask_total = torch.cat([past_mask, new_mask], dim=1)

            latent, _, present_kvs = self.backbone.transformer(
                x_new, pad_mask=pad_mask_total, past_kvs=past_kvs
            )
            h = latent[:, -1, :]

            logits=self.action_net(h)
            dist = Categorical(logits=logits)
            act = dist.sample()

            action[cont_idx]  = act
            logprob[cont_idx] = dist.log_prob(act)
            probs[cont_idx]   = dist.probs

            unpack_kvs(present_kvs, cont_idx, kvs_list)

        if new_idx.numel() > 0:
            obs_n = obs_t.index_select(0, new_idx)
            ts_n  = ts_t.index_select(0, new_idx)

            s = self.backbone.embed_state(obs_n)
            t = self.backbone.embed_timestep(ts_n.clamp(max=self.backbone.embed_timestep.num_embeddings-1))
            s = s + t

            x_new = s.unsqueeze(1)  # (B,1,D)
            latent, _, present_kvs = self.backbone.transformer(
                x_new, pad_mask=None, past_kvs=[None]*num_layers
            )
            h = latent[:, -1, :]

            logits=self.action_net(h)

            dist = Categorical(logits=logits)
            act = dist.sample()

            action[new_idx]  = act
            logprob[new_idx] = dist.log_prob(act)
            probs[new_idx]   = dist.probs

            unpack_kvs(present_kvs, new_idx, kvs_list)

        return action, logprob, probs

    def evaluate_state_policy(self, data_obs, data_ts, data_actions, splits):
        device = data_obs.device
        B = len(splits)
        Lmax = max(splits)
        C, H, W = data_obs.shape[1:]
        n_actions = self.action_net.out_features
        D = self.backbone.D

        obs_batch = torch.zeros(B, Lmax, C, H, W, device=device, dtype=data_obs.dtype)
        act_batch = torch.zeros(B, Lmax, device=device, dtype=torch.long)
        ts_batch  = torch.zeros(B, Lmax, device=device, dtype=torch.long)
        pad_mask  = torch.zeros(B, Lmax, device=device, dtype=torch.bool)

        cursor = 0
        for b, L in enumerate(splits):
            obs_batch[b, :L] = data_obs[cursor:cursor+L]
            act_batch[b, :L] = data_actions[cursor:cursor+L].long()
            ts_batch[b, :L]  = data_ts[cursor:cursor+L].long()
            pad_mask[b, :L]  = True
            cursor += L

        # tokens: (B, 2*Lmax, D) = [S0,A0,S1,A1,...]
        x_full = self.backbone.encode_tokens(obs_batch, act_batch, ts_batch)
        x = x_full[:, :-1, :]  # drop last action token (current action unknown)

        token_mask_full = torch.stack([pad_mask, pad_mask], dim=2).reshape(B, 2 * Lmax)
        token_mask = token_mask_full[:, :-1]

        latent, _, _ = self.backbone.transformer(x, pad_mask=token_mask)
        h_s = latent[:, 0::2, :]                     # (B, Lmax, D)
        h_s_flat = h_s.reshape(B * Lmax, D)          # (B*Lmax, D)

        logits = self.action_net(h_s_flat)

        dist = Categorical(logits=logits)
        probs_flat = dist.probs                      # (B*Lmax, A)
        entropy_flat = dist.entropy()                # (B*Lmax,)

        probs = probs_flat.reshape(B, Lmax, n_actions)
        entropy = entropy_flat.reshape(B, Lmax)

        probs_list, entropy_list = [], []
        for b, L in enumerate(splits):
            probs_list.append(probs[b, :L])
            entropy_list.append(entropy[b, :L])

        new_dist = torch.cat(probs_list, dim=0)                   # (M, A)
        entropy_out = torch.cat(entropy_list, dim=0)              # (M,)
        new_logprobs = torch.log(new_dist.clamp(min=1e-8))        # (M, A)

        return new_dist, new_logprobs, entropy_out


    
class ValueAdvNet(nn.Module):
    def __init__(self, envs, args):
        super().__init__()
        n_actions = envs.single_action_space.n
        obs_dim = envs.single_observation_space.shape
        C, H, W = obs_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(C, args.cnn_feature_dim, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(args.cnn_feature_dim, args.cnn_feature_dim, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            flat_dim = self.encoder(dummy).shape[-1]

        hidden_dim = args.v_dim
        self.trunk = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_net = layer_init(nn.Linear(hidden_dim, 1), std=1.0)
        self.advantage_net = layer_init(nn.Linear(hidden_dim, n_actions), std=0.1)

        self.n_actions = n_actions
        self.pad_action_id = n_actions  # kept only for compatibility

    def encode_state(self, obs):
        x = self.encoder(obs)
        x = self.trunk(x)
        return x

    def forward_values_tokens_dt(self, flat_obs, flat_act, flat_ts, splits):
        h = self.encode_state(flat_obs)
        v = self.value_net(h).squeeze(-1)
        return v

    def evaluate_state(self, data_obs, data_ts, data_actions, old_dist, splits):

        h = self.encode_state(data_obs)                   # (M, D)
        values = self.value_net(h).squeeze(-1)            # (M,)
        advantages_raw = self.advantage_net(h)            # (M, A)

        advantages = advantages_raw - torch.sum(
            old_dist * advantages_raw, dim=-1, keepdim=True
        )
        return values, advantages


class TrajectoryBuffer:
    def build_dae_traj_buffer(
        self, 
        obs, 
        actions, 
        old_probs, 
        logprobs, 
        rewards, 
        dones, 
        values, 
        next_done, 
        next_value,
        timesteps):

        num_steps, num_envs = dones.shape
        device = values.device
        self.device = device

        flat_obs_list = []
        flat_act_list = []
        flat_logp_list = []
        flat_rew_list = []
        flat_done_list = []
        flat_val_list = []
        flat_old_probs_list = []

        flat_ts_list = []

        lengths = []
        last_values = []

        for env in range(num_envs):
            start = 0
            done_idx = torch.nonzero(dones[:, env] > 0.5, as_tuple=False).flatten().tolist()
            # print(f"env: {env}, {dones.shape}, done_idx: {done_idx}")

            for t in done_idx:
                end = t + 1
                if end > start:
                    # if end - start == 1:   # len-1 segment about to be created
                    #     print(f"[LEN1] env={env} start={start} t={t} dones[t]={dones[t, env].item()}")

                    # append segment slice
                    flat_obs_list.append(obs[start:end, env])
                    flat_act_list.append(actions[start:end, env])
                    flat_logp_list.append(logprobs[start:end, env])
                    flat_rew_list.append(rewards[start:end, env])
                    flat_done_list.append(dones[start:end, env])
                    flat_val_list.append(values[start:end, env])
                    flat_old_probs_list.append(old_probs[start:end, env])

                    lengths.append(end - start)
                    last_values.append(torch.tensor(0.0, device=device))

                    flat_ts_list.append(timesteps[start:end, env])

                start = end
            
            if start < num_steps:
                end = num_steps
                flat_obs_list.append(obs[start:end, env])
                flat_act_list.append(actions[start:end, env])
                flat_logp_list.append(logprobs[start:end, env])
                flat_rew_list.append(rewards[start:end, env])
                flat_done_list.append(dones[start:end, env])
                flat_val_list.append(values[start:end, env])
                flat_old_probs_list.append(old_probs[start:end, env])

                lengths.append(end - start)
                # bootstrap if not done at end
                last_values.append(next_value[env] * (1.0 - next_done[env]))

                flat_ts_list.append(timesteps[start:end, env])

            
        self.observations = torch.cat(flat_obs_list, dim=0)
        self.actions = torch.cat(flat_act_list, dim=0)
        self.logprobs = torch.cat(flat_logp_list, dim=0)
        self.rewards = torch.cat(flat_rew_list, dim=0)
        self.dones = torch.cat(flat_done_list, dim=0)
        self.values = torch.cat(flat_val_list, dim=0)
        self.old_probs = torch.cat(flat_old_probs_list, dim=0)
        self.last_values = torch.stack(last_values, dim=0)


        self.lengths_np = np.asarray(lengths, dtype=np.int64)
        self.end_indices = np.cumsum(self.lengths_np)
        self.start_indices = np.insert(self.end_indices, 0, 0)[:-1]
        # print(len(self.observations), len(self.start_indices))
        # print(f"lengths_np: {self.lengths_np}")
        # print(f"start_indices: {self.start_indices}")
        # print(f"end_indices: {self.end_indices}")

        self.timesteps = torch.cat(flat_ts_list, dim=0)


        self.advantages = torch.empty(
            self.old_probs.shape, dtype=torch.float32, device=device
        )


    def get_trajs(self, batch_size):
        if batch_size is None:
            batch_size = len(self.observations)
        batch_size = min(batch_size, len(self.observations))

        indices = np.random.permutation(len(self.start_indices))

        start_idx = 0
        while start_idx < len(indices):
            traj_indices = []
            total_frames = 0
            while total_frames < batch_size and start_idx < len(indices):
                t_idx = indices[start_idx]
                traj_indices.append(t_idx)
                total_frames += self.end_indices[t_idx] - self.start_indices[t_idx]
                start_idx += 1
            if total_frames < batch_size:
                break
            yield self._get_traj_samples(traj_indices)


    def _get_traj_samples(self, indices):
        _obs, _act, _rew, _prob, _lpol, _val, _last, _timesteps, splits = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for idx in indices:
            start, end = self.start_indices[idx], self.end_indices[idx]
            _obs.append(self.observations[start:end])
            _act.append(self.actions[start:end])
            _rew.append(self.rewards[start:end])
            _prob.append(self.old_probs[start:end])
            _lpol.append(self.logprobs[start:end])
            _val.append(self.values[start:end])
            _last.append(self.last_values[idx])
            _timesteps.append(self.timesteps[start:end])
            splits.append(end - start)

        return (
            torch.cat(_obs),
            torch.cat(_act),
            torch.cat(_rew),
            torch.cat(_prob),
            torch.cat(_lpol),
            torch.cat(_val),
            torch.cat(_timesteps),
            _last,
            splits,
        )



def compute_value_loss(deltas, values, lasts, discount_matrix, discount_vector):
    # print(f"discount_matrix: {discount_matrix.shape}, {discount_matrix}")
    # print(f"discount_vector: {discount_vector.shape}, {discount_vector}")
    loss = torch.cat(
        [
            (
                discount_matrix[: len(d), : len(d)].matmul(d)
                + l * discount_vector[-len(d) :]
                - v
            ).square()
            for d, v, l in zip(deltas, values, lasts)
        ]
    ).mean()
    return loss



def normalize_advantage(advantages, policies, eps=1e-5):
    std = (policies * advantages.pow(2)).sum(dim=1).mean().sqrt()
    return advantages / (std + eps)



def compute_policy_loss(args, advantages, log_policy, old_log_policy, actions, clip_range=None):
        if args.full_action:
            ratio = torch.exp(log_policy - old_log_policy)
            loss = -(advantages * torch.exp(log_policy)).sum(dim=1).mean()
        else:
            adv = advantages.gather(-1, actions).flatten()
            logp = log_policy.gather(-1, actions).flatten()
            old_logp = old_log_policy.gather(-1, actions).flatten()
            ratio = torch.exp(logp - old_logp)

            policy_loss_1 = adv * ratio
            policy_loss_2 = adv * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
            loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        return loss, ratio


def dae_targets_and_preds(deltas, values, lasts, discount_matrix, discount_vector):

    targets = []
    preds = []

    # lasts in your code is usually a Python list of 0-dim tensors
    for d, v, l in zip(deltas, values, lasts):
        T = len(d)
        y = discount_matrix[:T, :T].matmul(d) + l * discount_vector[-T:]
        targets.append(y)
        preds.append(v)

    return torch.cat(targets, dim=0), torch.cat(preds, dim=0)


def explained_variance(y_pred, y_true, eps=1e-8):
    # EV = 1 - Var(y - yhat) / Var(y)
    var_y = torch.var(y_true)
    if var_y.item() < eps:
        return torch.tensor(float("nan"), device=y_true.device)
    return 1.0 - torch.var(y_true - y_pred) / (var_y + eps)



def flatten_with_bootstrap_dt(obs, actions, timesteps,
                             dones_after, next_obs, next_done,
                             next_timesteps,
                             pad_action_id: int):
    """
    Returns flat_* arrays aligned by index so that for every flat_obs[i],
    you also have flat_action[i], flat_ts[i].

    For unfinished tail segments, we append ONE bootstrap state with:
      action = PAD
      ts     = next_timesteps[env]
    """
    T, N = dones_after.shape
    device = obs.device

    flat_obs, flat_act, flat_ts = [], [], []
    flat_t, flat_env, is_boot, splits = [], [], [], []
    boot_pos = torch.full((N,), -1, device=device, dtype=torch.long)

    cursor = 0
    for env in range(N):
        start = 0
        done_idx = torch.nonzero(dones_after[:, env] > 0.5, as_tuple=False).flatten().tolist()

        # finished segments inside rollout
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

        # tail segment (unfinished within rollout)
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
            flat_ts.append(next_timesteps[env].long().unsqueeze(0))

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



def build_traj_batch(step, obs, actions, timesteps, ep_start, pad_action_id):
    """
    Build per-env episode-prefix trajectories up to current step (inclusive),
    padded to max length across envs.

    Returns:
      states_b:   (N, Lmax, C, H, W)
      actions_b:  (N, Lmax)   # previous actions, with PAD in last position
      t_b:        (N, Lmax)
      pad_mask:   (N, Lmax)   # True=valid, False=pad
      lengths:    (N,)
    """

    # print(f"step: {step}, ep_start: {ep_start}")

    device = obs.device
    N = obs.shape[1]
    C, H, W = obs.shape[2:]
 
    lengths = (step + 1 - ep_start).clamp(min=1)
    Lmax = int(lengths.max().item())

    # print(f"lengths: {lengths}, Lmax: {Lmax}")

    states_b = torch.zeros((N, Lmax, C, H, W), device=device, dtype=obs.dtype)
    actions_b = torch.full((N, Lmax), pad_action_id, device=device, dtype=torch.long)
    ts_b = torch.zeros((N, Lmax), device=device, dtype=torch.long)
    pad_mask = torch.zeros((N, Lmax), device=device, dtype=torch.bool)

    for e in range(N):
        s0 = int(ep_start[e].item())
        L = int(lengths[e].item())

        states_b[e, :L] = obs[s0:step+1, e]
        ts_b[e, :L] = timesteps[s0:step+1, e].long()
        pad_mask[e, :L] = True

        # previous actions only (up to step-1), PAD at the last slot (current action unknown)
        if L > 1:
            actions_b[e, :L-1] = actions[s0:step, e].long()
        actions_b[e, L-1] = pad_action_id

    return states_b, actions_b, ts_b, pad_mask, lengths



def forward_values_tokens_dt_traj_minibatches(
    agent,
    flat_obs, flat_act, flat_ts,
    splits,
    traj_minibatch_size=256):
    return agent.forward_values_tokens_dt(flat_obs, flat_act, flat_ts, splits)



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
    # args = tyro.cli(Args)
    args = parse_args()
    args.num_iterations = args.total_timesteps // int(args.num_envs * args.num_steps)
    run_name = f"{args.exp_name}_seed_{args.seed}_pi_layers_{args.pi_transformer_layers}_pi_dim_{args.pi_transformer_dim}_v_dim_{args.v_dim}_pi_heads_{args.pi_num_heads}_cnn_dim_{args.cnn_feature_dim}_schedule_{args.lr_schedule}_entropy_{args.ent_coef}_lr_{args.learning_rate}"
    print(args)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            # sync_tensorboard=True,
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    register_envs()
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    pad_action_id = envs.single_action_space.n

    policy = PolicyNet(envs, args).to(device)
    vadv = ValueAdvNet(envs, args).to(device)

    optimizer_pi = torch.optim.Adam(
        policy.parameters(), 
        lr=args.learning_rate, 
        eps=1e-8,
        weight_decay=args.weight_decay, 
        betas=(args.beta1, args.beta2)
    )
    optimizer_v = torch.optim.Adam(
        vadv.parameters(), 
        lr=args.learning_rate_vf, 
        eps=1e-8,
        weight_decay=args.weight_decay, 
        betas=(args.beta1, args.beta2)
    )

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    dones_after = torch.zeros((args.num_steps, args.num_envs)).to(device)
    next_timesteps = torch.zeros(args.num_envs).to(device)
    timesteps = torch.zeros((args.num_steps, args.num_envs)).to(device)

    traj_buffer = TrajectoryBuffer()

    # store behavior policy for centering (mu): logits_old -> probs_old
    old_probs = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.n)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    running_avg_reward = None


    discount_matrix = torch.tensor(
        [
            [0 if j < i else args.gamma ** (j - i) for j in range(args.num_steps)]
            for i in range(args.num_steps)
        ],
        dtype=torch.float32,
        device=device,
    )
    discount_vector = args.gamma ** torch.arange(
        args.num_steps, 0, -1, dtype=torch.float32, device=device
    )


    clip_coef = args.clip_coef

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        lr_mul = lr_multiplier(iteration, args.num_iterations + 1, args)
        lrnow = lr_mul * args.learning_rate
        lrnow_vf = lr_mul * args.learning_rate_vf
        optimizer_pi.param_groups[0]["lr"] = lrnow
        optimizer_v.param_groups[0]["lr"] = lrnow_vf

        if args.anneal_clip_coef:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            clip_coef = frac * args.clip_coef


        kvs_list = [None] * args.num_envs
        prev_action = torch.full((args.num_envs,), pad_action_id, device=device, dtype=torch.long)
        is_new_ep = torch.ones(args.num_envs, device=device, dtype=torch.bool)  # first step

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            timesteps[step] = next_timesteps

            # print(f"step: {step}, is_new_ep: {is_new_ep}, ts_t: {next_timesteps}")

            with torch.no_grad():
                action, logprob, probs = policy.act_kvcache_per_env(
                    obs_t=next_obs,
                    ts_t=next_timesteps.long(),
                    prev_action=prev_action,
                    is_new_ep=is_new_ep,
                    kvs_list=kvs_list,
                )
        
            actions[step] = action
            logprobs[step] = logprob
            old_probs[step] = probs
            
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)

            prev_action = action.clone()


            dones_after[step] = torch.as_tensor(next_done, device=device, dtype=torch.float32)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            done_mask = next_done.bool()
            # print(f"done_mask, {done_mask}")
            # print(f"reward, {reward}")
            next_timesteps += 1       
            next_timesteps = torch.clamp(next_timesteps, max=args.max_ep_len-1)
            next_timesteps[done_mask] = 0


            
            for e in torch.nonzero(done_mask, as_tuple=False).flatten().tolist():
                kvs_list[e] = None
            prev_action[done_mask] = pad_action_id
            is_new_ep = done_mask
            
        
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
            flat_obs, flat_act, flat_ts, splits, flat_t, flat_env, is_boot, boot_pos = flatten_with_bootstrap_dt(
                obs=obs,
                actions=actions,
                timesteps=timesteps,
                dones_after=dones_after,
                next_obs=next_obs,
                next_done=next_done,
                next_timesteps=next_timesteps,
                pad_action_id=policy.pad_action_id,
            )


            flat_v = forward_values_tokens_dt_traj_minibatches(
                vadv, flat_obs, flat_act, flat_ts, splits,
                traj_minibatch_size=64,
            )

            values.zero_()
            rollout_mask = ~is_boot
            t_idx = flat_t[rollout_mask]
            e_idx = flat_env[rollout_mask]
            values[t_idx, e_idx] = flat_v[rollout_mask]

            next_value = torch.zeros(args.num_envs, device=device)
            has_boot = boot_pos >= 0
            next_value[has_boot] = flat_v[boot_pos[has_boot]]
            next_value = next_value * (1.0 - next_done.float())


        # traj_buffer.build_dae_traj_buffer(obs, actions, old_probs, logprobs, rewards, dones, values, next_done, next_value)
        traj_buffer.build_dae_traj_buffer(
            obs, actions, old_probs, logprobs, rewards, dones_after, values, next_done, next_value, timesteps)

        # break

        all_targets = []
        all_preds = []
        clipfracs = []
        
        for epoch in range(args.update_epochs):
            for data in traj_buffer.get_trajs(batch_size=args.batch_size):
                (
                    data_obs, 
                    data_actions, 
                    data_rewards, 
                    old_dist, 
                    old_logprobs_action, 
                    old_values, 
                    data_ts, 
                    data_last_values, 
                    data_splits
                ) = data

                new_dist, new_logprobs, entropy = policy.evaluate_state_policy(
                    data_obs, data_ts, data_actions, splits=data_splits
                )

                new_values, new_advantages = vadv.evaluate_state(
                    data_obs, data_ts, data_actions, old_dist=old_dist, splits=data_splits
                )
        
                # print(f"new_values:{new_values.shape}, new_advantages:{new_advantages.shape}, new_dist:{new_dist.shape}, new_logprobs:{new_logprobs.shape}, entropy:{entropy.shape}")

                new_values = new_values.flatten().split(data_splits)
                # print(f"new_adv: {new_advantages}")
                # print(f"action: {data_actions}")
                # print(f"adv_action, {new_advantages.gather(dim=1, index=data_actions.long().view(-1, 1))}")
                deltas = (
                    data_rewards - new_advantages.gather(dim=1, index=data_actions.long().view(-1, 1)).flatten()
                ).split(data_splits)
                # print(f"data_splits: {data_splits}")
                # print(f"deltas: {type(deltas)}, {deltas}")

                v_loss = compute_value_loss(deltas, new_values, data_last_values, discount_matrix, discount_vector)

                with torch.no_grad():
                    y_true, y_pred = dae_targets_and_preds(
                        deltas=deltas,
                        values=new_values,
                        lasts=data_last_values,
                        discount_matrix=discount_matrix,
                        discount_vector=discount_vector,
                    )
                    all_targets.append(y_true)
                    all_preds.append(y_pred)


                eps = 1e-8
                old_logprobs = torch.log(old_dist.clamp(min=eps))
                kl_loss = (
                    (old_dist * (old_logprobs - new_logprobs)).sum(dim=1).mean()
                )

                new_advantages = new_advantages.detach().clone()
                if args.norm_adv:
                    new_advantages = normalize_advantage(new_advantages, old_dist)

                # print(new_advantages.shape, new_logprobs.shape, old_logprobs.shape, data_actions.shape, data_actions.long().view(-1,1).shape)
                pg_loss, ratio = compute_policy_loss(args, new_advantages, new_logprobs, old_logprobs, data_actions.long().view(-1,1), clip_range=clip_coef)
                clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                entropy_loss = torch.mean(entropy)

                # loss = (
                #     pg_loss
                #     - args.ent_coef * entropy_loss
                #     + args.kl_coef * kl_loss
                #     + args.vf_coef * v_loss
                # )

                policy_loss_total = pg_loss - args.ent_coef * entropy_loss + args.kl_coef * kl_loss
                v_loss_total = args.vf_coef * v_loss

                optimizer_pi.zero_grad()
                (policy_loss_total).backward()
                policy_grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer_pi.step()

                optimizer_v.zero_grad()
                (v_loss_total).backward()
                value_grad_norm = nn.utils.clip_grad_norm_(vadv.parameters(), args.max_grad_norm_vf)
                optimizer_v.step()


        if len(all_targets) > 0:
            targets = torch.cat(all_targets, dim=0)
            preds = torch.cat(all_preds, dim=0)
            explained_var = explained_variance(y_pred=preds, y_true=targets).item()
        else:
            explained_var = float("nan")


        if args.track:
            wandb.log(
                {
                    "losses/value_loss": v_loss_total.item(),
                    "losses/policy_loss": policy_loss_total.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "losses/grad_norm": policy_grad_norm.item(),
                    "losses/grad_norm_v": value_grad_norm.item(),
                    "charts/clip_coef": clip_coef,
                    "charts/learning_rate": lrnow,
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                },
                step=global_step,
            )
        
    

    envs.close()
    writer.close()
