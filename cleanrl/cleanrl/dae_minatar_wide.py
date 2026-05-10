# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import argparse
import random
import time
from dataclasses import dataclass

import gymnasium as gym
# from gym.wrappers import TimeLimit
import numpy as np
import torch
import torch.nn as nn
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
    parser.add_argument("--batch_size_vf", type=int, default=64)
    parser.add_argument("--full_action", action="store_true")
    parser.add_argument("--kl_coef", type=float, default=0.0)
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--anneal_clip_coef", action="store_true")
    parser.add_argument("--max_grad_norm_vf", type=float, default=100.0)

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
            # max_episode_steps=500000,
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
            # max_episode_steps=500000,
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





# def make_env(env_id, idx, capture_video, run_name):
#     def thunk():
#         if "MinAtar" in env_id or "minatar" in env_id.lower():
#             # game = env_id.replace("-v0", "").replace("-MinAtar", "").lower()
#             game = env_id
#             env = MinAtarGymnasiumEnv(
#                 game=game,
#                 use_minimal_action_set=True,
#                 sticky_action_prob=0.0,
#                 difficulty_ramping=False,
#                 max_steps=1000000,
#             )
#         else:
#             env = gym.make(env_id)

#         if capture_video and idx == 0:
#             env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         return env
#     return thunk



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SeperateAgent(nn.Module):
    def __init__(self, envs, features_dim=1024):
        super().__init__()
        n_actions = envs.single_action_space.n
        obs_dim = envs.single_observation_space.shape
        print(f"obs_space: {obs_dim}")

        n_input_channels = obs_dim[0]

        cnn_1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_2 = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1,) + obs_dim)
            # dummy = dummy.permute(0, 3, 1, 2)
            flat_dim = cnn_1(dummy).shape[-1]
            # print(f"n_flatten: {flat_dim}")

        linear_1 = nn.Sequential(nn.Linear(flat_dim, features_dim), nn.ReLU(),)
        linear_2 = nn.Sequential(nn.Linear(flat_dim, features_dim), nn.ReLU(),)
        
        self.features_extractor = nn.Sequential(cnn_1, linear_1)
        self.features_extractor_vf = nn.Sequential(cnn_2, linear_2)

        self.action_net = layer_init(nn.Linear(features_dim, n_actions), std=0.01)
        self.value_net = layer_init(nn.Linear(features_dim, 1), std=1.0)
        self.advantage_net = layer_init(nn.Linear(features_dim, n_actions), std=0.1)


    def get_action_and_value(self, x, action=None):
        latent = self.features_extractor(x)
        latent_vf = self.features_extractor_vf(x)

        action_logits = self.action_net(latent)
        action_probs = Categorical(logits=action_logits)
        probs = action_probs.probs
        if action is None:
            action = action_probs.sample()

        entropy = action_probs.entropy()
        log_probs = action_probs.log_prob(action)

        values = self.value_net(latent_vf)

        return action, log_probs, entropy, values, probs

    
    def predict_policy(self, obs):
        latent = self.features_extractor(obs)
        action_logits = self.action_net(latent)
        action_probs = Categorical(logits=action_logits)
        probs = action_probs.probs
        log_probs_all = torch.log(probs + 1e-8)
        entropy = action_probs.entropy()

        return probs, log_probs_all, entropy

    
    def get_value(self, x):
        latent_vf = self.features_extractor_vf(x)
        values = self.value_net(latent_vf)
        return values


    def predict_values(self, obs, policies):
        latent = self.features_extractor_vf(obs)
        advantages_raw = self.advantage_net(latent)
        if policies is None:
            advantages = advantages_raw
        else:
            advantages = advantages_raw - torch.sum(
                policies * advantages_raw, dim=1, keepdim=True
            )
        values = self.value_net(latent)
        return values, advantages



class SharedAgent(nn.Module):
    def __init__(self, envs, features_dim=1024):
        super().__init__()
        n_actions = envs.single_action_space.n
        obs_dim = envs.single_observation_space.shape
        print(f"obs_space: {obs_dim}")

        n_input_channels = obs_dim[0]

        cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 128, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1,) + obs_dim)
            # dummy = dummy.permute(0, 3, 1, 2)
            flat_dim = cnn(dummy).shape[-1]
            # print(f"n_flatten: {flat_dim}")
        linear = nn.Sequential(nn.Linear(flat_dim, features_dim), nn.ReLU(),)
        
        self.features_extractor = nn.Sequential(cnn, linear)

        self.action_net = layer_init(nn.Linear(features_dim, n_actions), std=0.01)
        self.value_net = layer_init(nn.Linear(features_dim, 1), std=1.0)
        self.advantage_net = layer_init(nn.Linear(features_dim, n_actions), std=0.1)


    def get_action_and_value(self, x, action=None):
        # x = x.permute(0, 3, 1, 2)
        latent = self.features_extractor(x)

        action_logits = self.action_net(latent)
        action_probs = Categorical(logits=action_logits)
        probs = action_probs.probs
        if action is None:
            action = action_probs.sample()

        entropy = action_probs.entropy()
        log_probs = action_probs.log_prob(action)

        values = self.value_net(latent)

        return action, log_probs, entropy, values, probs

    
    def get_value(self, x):
        # x = x.permute(0, 3, 1, 2)
        latent_vf = self.features_extractor(x)
        values = self.value_net(latent_vf)
        return values


    def predict_values(self, obs, policies):
        # obs = obs.permute(0, 3, 1, 2)
        latent = self.features_extractor(obs)
        advantages_raw = self.advantage_net(latent)
        if policies is None:
            advantages = advantages_raw
        else:
            advantages = advantages_raw - torch.sum(
                policies * advantages_raw, dim=1, keepdim=True
            )
        values = self.value_net(latent)
        return values, advantages


    def evaluate_state(self, obs, policies):
        # obs = obs.permute(0, 3, 1, 2)
        latent = self.features_extractor(obs)
        action_logits = self.action_net(latent)
        action_probs = Categorical(logits=action_logits)
        probs = action_probs.probs
        log_probs_all = torch.log(probs + 1e-8)

        advantages_raw = self.advantage_net(latent)
        # print(f"predict_values: adv: {advantages_raw.shape}, policies: {policies.shape}, mul: {torch.sum(policies * advantages_raw, dim=1, keepdim=True).shape}")

        advantages = advantages_raw - torch.sum(
            policies * advantages_raw, dim=1, keepdim=True
        )
        values = self.value_net(latent)
        return values, advantages, probs, log_probs_all, action_probs.entropy()




class TrajectoryBuffer:
    def build_dae_traj_buffer(self, 
        obs, actions, old_probs, logprobs, rewards, dones, values, next_done, next_value):

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


        self.advantages = torch.empty(
            self.old_probs.shape, dtype=torch.float32, device=device
        )

    
    def get(self, batch_size):
        indices = torch.randperm(len(self.observations), device=self.device)

        if batch_size is None:
            batch_size = len(indices)
        batch_size = min(batch_size, len(self.observations))

        start_idx = 0
        while start_idx < len(indices):
            yield self._get_samples(
                indices[start_idx : min(start_idx + batch_size, len(indices))]
            )
            start_idx += batch_size
        

    def _get_samples(self, indices):
        data = (
            self.observations.index_select(dim=0, index=indices),
            self.actions.index_select(dim=0, index=indices),
            self.rewards.index_select(dim=0, index=indices),
            self.old_probs.index_select(dim=0, index=indices),
            self.logprobs.index_select(dim=0, index=indices),
            self.advantages.index_select(dim=0, index=indices),
        )
        return data
    

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
        _obs, _act, _rew, _prob, _lpol, _val, _last, splits = (
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
            splits.append(end - start)

        return (
            torch.cat(_obs),
            torch.cat(_act),
            torch.cat(_rew),
            torch.cat(_prob),
            torch.cat(_lpol),
            torch.cat(_val),
            _last,
            splits,
        )

    
    def update_advantage(self, agent, batch_size=1024):
        start = 0
        size = len(self.observations)
        while start < size:
            end = min(start + batch_size, size)
            _obs = self.observations[start:end]
            _pol = self.old_probs[start:end]
            with torch.no_grad():
                _, adv = agent.predict_values(_obs, _pol)
            self.advantages[start:end] = adv
            start = end




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



if __name__ == "__main__":
    # args = tyro.cli(Args)
    args = parse_args()
    args.num_iterations = args.total_timesteps // int(args.num_envs * args.num_steps)
    run_name = f"{args.exp_name}_base_seed_{args.seed}_shared_{args.shared}"
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


    if args.shared:
        agent = SharedAgent(envs).to(device)
        optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    else:
        agent = SeperateAgent(envs).to(device)
        modules_pi = nn.ModuleList([agent.features_extractor, agent.action_net])
        modules_vf = nn.ModuleList([agent.features_extractor_vf, agent.value_net, agent.advantage_net])

        optimizer_pi = torch.optim.Adam(modules_pi.parameters(), lr=args.learning_rate, eps=1e-5)
        optimizer_vf = torch.optim.Adam(modules_vf.parameters(), lr=args.learning_rate_vf, eps=1e-5)



    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    dones_after = torch.zeros((args.num_steps, args.num_envs)).to(device)

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

    # print(discount_matrix)
    # print(discount_vector)


    clip_coef = args.clip_coef

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.shared:
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow
        else:
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer_pi.param_groups[0]["lr"] = lrnow

            if args.anneal_lr_vf:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate_vf
                optimizer_vf.param_groups[0]["lr"] = lrnow

        if args.anneal_clip_coef:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            clip_coef = frac * args.clip_coef

        
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, probs = agent.get_action_and_value(next_obs)
                # print(probs)
                # print(f"action: {action}")
                # print(f"logprob: {logprob}")
                # print(f"next_obs: {next_obs.shape}, {next_obs}")
                # print(f"value: {value.shape}, {value}")
                values[step] = value.flatten()
                
            actions[step] = action
            logprobs[step] = logprob
            old_probs[step] = probs

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # print(f"next_obs: {next_obs.min()}, {next_obs.max()}, {next_obs.dtype}")

            next_done = np.logical_or(terminations, truncations)

            dones_after[step] = torch.as_tensor(next_done, device=device, dtype=torch.float32)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)


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
            next_value = agent.get_value(next_obs).flatten()

        # print(f"next_obs: {next_obs.shape}, {next_obs}")
        # print(f"next_done: {next_done.shape}, {next_done}")
        # print(f"next_value: {next_value.shape}, {next_value}")

        # print(f"done: {dones.shape}, {dones}")
        # print(f"done_after: {dones_after.shape}, {dones_after}")


        # traj_buffer.build_dae_traj_buffer(obs, actions, old_probs, logprobs, rewards, dones, values, next_done, next_value)
        traj_buffer.build_dae_traj_buffer(obs, actions, old_probs, logprobs, rewards, dones_after, values, next_done, next_value)

        # break

        all_targets = []
        all_preds = []
        clipfracs = []
        vf_grad_norm = None
        grad_norm = None

        if not args.shared:
            for epoch in range(args.update_epochs_vf):
                for data in traj_buffer.get_trajs(batch_size=args.batch_size_vf):
                    data_obs, data_actions, data_rewards, data_old_probs, data_old_logprobs, data_old_values, data_last_old_values, data_splits = data
                    # print(data_old_probs)
                    new_values, new_advantages = agent.predict_values(data_obs, data_old_probs)

                    new_values = new_values.flatten().split(data_splits)
                    # print(new_advantages.gather(dim=1, index=data_actions.long().view(-1, 1)).shape)
                    deltas = (
                        data_rewards - new_advantages.gather(dim=1, index=data_actions.long().view(-1, 1)).flatten()
                    ).split(data_splits)
                    # print(len(new_values), len(deltas))

                    v_loss = compute_value_loss(deltas, new_values, data_last_old_values, discount_matrix, discount_vector)

                    with torch.no_grad():
                        y_true, y_pred = dae_targets_and_preds(
                            deltas=deltas,
                            values=new_values,
                            lasts=data_last_old_values,
                            discount_matrix=discount_matrix,
                            discount_vector=discount_vector,
                        )
                        all_targets.append(y_true)
                        all_preds.append(y_pred)

                    optimizer_vf.zero_grad()
                    v_loss.backward()
                    vf_grad_norm = nn.utils.clip_grad_norm_(modules_vf.parameters(), args.max_grad_norm_vf)
                    optimizer_vf.step()

            traj_buffer.update_advantage(agent)
    
            for epoch in range(args.update_epochs):
                for data in traj_buffer.get(batch_size=args.batch_size): 
                    data_obs, data_actions, data_reward, old_dist, old_logprob, data_advantages = data
                    data_actions = data_actions.long().view(-1)

                    # _, new_logprob, entropy, new_value, new_dist = agent.get_action_and_value(
                    #     data_obs, data_actions.long())

                    new_dist, new_logprobs, entropy = agent.predict_policy(data_obs)

                    eps = 1e-8
                    old_logprobs = torch.log(old_dist.clamp(min=eps))
                    kl_div = (
                        (old_dist * (old_logprobs - new_logprobs)).sum(dim=1).mean()
                    )

                    if args.norm_adv:
                        data_advantages = normalize_advantage(data_advantages, old_dist)

                    pg_loss, ratio = compute_policy_loss(args, data_advantages, new_logprobs, old_logprobs, data_actions.long().view(-1,1), clip_range=clip_coef)


                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + args.kl_coef * kl_div

                    optimizer_pi.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(modules_pi.parameters(), args.max_grad_norm)
                    optimizer_pi.step()


        else:
            for epoch in range(args.update_epochs):
                for data in traj_buffer.get_trajs(batch_size=args.batch_size):
                    data_obs, data_actions, data_rewards, old_dist, old_logprobs_action, old_values, data_last_values, data_splits = data
                    new_values, new_advantages, new_dist, new_logprobs, entropy = agent.evaluate_state(data_obs, old_dist)
                    
                    # print(f"data_actions, {data_actions.shape}")
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

                    loss = (
                        pg_loss
                        - args.ent_coef * entropy_loss
                        + args.kl_coef * kl_loss
                        + args.vf_coef * v_loss
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    # nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    
                    optimizer.step()


        if len(all_targets) > 0:
            targets = torch.cat(all_targets, dim=0)
            preds = torch.cat(all_preds, dim=0)
            explained_var = explained_variance(y_pred=preds, y_true=targets).item()
        else:
            explained_var = float("nan")


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
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                },
                step=global_step,
            )
            if vf_grad_norm:
                wandb.log(
                    {
                        "losses/vf_grad_norm": vf_grad_norm.item(),
                    },
                    step=global_step,
                )
        

    envs.close()
    writer.close()
