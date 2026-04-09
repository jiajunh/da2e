# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import os
import random
import time
import argparse
from collections import deque
from dataclasses import dataclass

import envpool

# import gymnasium as gym
import gym
import numpy as np
import tensordict
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import from_module
from tensordict.nn import CudaGraphModule
from torch.distributions.categorical import Categorical, Distribution

Distribution.set_default_validate_args(False)

# This is a quick fix while waiting for https://github.com/pytorch/pytorch/pull/138080 to land
# Categorical.logits = property(Categorical.__dict__["logits"].wrapped)
# Categorical.probs = property(Categorical.__dict__["probs"].wrapped)

torch.set_float32_matmul_precision("high")




def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment args
    parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__)[:-len(".py")])
    parser.add_argument("--project_name", type=str, default="dae_atari")
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
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument("--anneal_lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--norm_adv", action="store_true")
    parser.add_argument("--clip_coef", type=float, default=0.1)
    parser.add_argument("--clip_vloss", action="store_true")
    parser.add_argument("--ent_coef", type=float, default=0.1)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=None)

    # Runtime filled
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_iterations", type=int, default=0)

    # New added
    parser.add_argument("--model_type", type=str, default="natural")
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--avg_returns_last_k", type=int, default=20)

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


def explained_variance(y_pred, y_true, eps=1e-8):
    # EV = 1 - Var(y - yhat) / Var(y)
    var_y = torch.var(y_true)
    if var_y.item() < eps:
        return torch.tensor(float("nan"), device=y_true.device)
    return 1.0 - torch.var(y_true - y_pred) / (var_y + eps)



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, device=None, model_type=None):
        super().__init__()

        feature_dim = 512

        if model_type == "natural2x":
            feature_dim = 1024
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(4, 64, 8, stride=4, device=device)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 128, 4, stride=2, device=device)),
                nn.ReLU(),
                layer_init(nn.Conv2d(128, 128, 3, stride=1, device=device)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(128 * 7 * 7, feature_dim, device=device)),
                nn.ReLU(),
            )

        else:
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(4, 32, 8, stride=4, device=device)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2, device=device)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1, device=device)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, feature_dim, device=device)),
                nn.ReLU(),
            )

        self.action_net = layer_init(nn.Linear(feature_dim, envs.single_action_space.n, device=device), std=0.01)
        self.value_net = layer_init(nn.Linear(feature_dim, 1, device=device), std=1.0)
        self.advantage_net = layer_init(nn.Linear(feature_dim, envs.single_action_space.n, device=device), std=0.1)


    def get_action_and_value(self, x, action=None):
        latent = self.network(x / 255.0)
        action_logits = self.action_net(latent)
        action_probs = Categorical(logits=action_logits)
        if action is None:
            action = action_probs.sample()
        probs = action_probs.probs
        entropy = action_probs.entropy()
        log_probs = action_probs.log_prob(action)
        values = self.value_net(latent)
        return action, log_probs, entropy, values, probs
    

    def get_value(self, x):
        latent_vf = self.network(x / 255.0)
        values = self.value_net(latent_vf)
        return values


    def evaluate_state(self, x, policies):
        latent = self.network(x / 255.0)
        action_logits = self.action_net(latent)
        action_probs = Categorical(logits=action_logits)

        probs = action_probs.probs
        log_probs_all = torch.log(probs.clamp(min=1e-8))

        advantages_raw = self.advantage_net(latent)
        advantages = advantages_raw - torch.sum(
            policies * advantages_raw, dim=1, keepdim=True
        )
        values = self.value_net(latent)
        return values, advantages, probs, log_probs_all, action_probs.entropy()


def rollout(obs, done, avg_returns=[]):
    ts = []
    for step in range(args.num_steps):
        torch.compiler.cudagraph_mark_step_begin()
        action, logprob, _, value, probs = policy(obs)
        next_obs_np, reward, next_done_np, info = envs.step(action.cpu().numpy())
        
        next_obs = torch.as_tensor(next_obs_np)
        reward = torch.as_tensor(reward, device=device, dtype=torch.float32)
        next_done = torch.as_tensor(next_done_np)

        idx = next_done
        if idx.any():
            idx = idx & torch.as_tensor(info["lives"] == 0, device=next_done.device, dtype=torch.bool)
            if idx.any():
                r = torch.as_tensor(info["r"])
                avg_returns.extend(r[idx])
        
        ts.append(
            tensordict.TensorDict._new_unsafe(
                obs=obs,
                dones=done,
                vals=value.flatten(),
                actions=action,
                logprobs=logprob,
                old_probs=probs,
                rewards=reward,
                dones_after=next_done,
                batch_size=(args.num_envs,),
            )
        )

        obs = next_obs = next_obs.to(device, non_blocking=True)
        done = next_done.to(device, non_blocking=True)

    container = torch.stack(ts, 0)
    return obs, done, container



def build_dae_dataset(container, next_done, next_value):
    obs = container["obs"]
    actions = container["actions"]
    logprobs = container["logprobs"]
    rewards = container["rewards"]
    values = container["vals"]
    old_probs = container["old_probs"]
    dones_after = container["dones_after"].float()

    num_steps, num_envs = dones_after.shape
    device = obs.device

    flat_obs_list = []
    flat_act_list = []
    flat_logp_list = []
    flat_rew_list = []
    flat_val_list = []
    flat_old_probs_list = []
    lengths = []
    last_values = []

    for env in range(num_envs):
        start = 0
        done_idx = torch.nonzero(dones_after[:, env] > 0.5, as_tuple=False).flatten().tolist()

        for t in done_idx:
            end = t + 1
            if end > start:
                flat_obs_list.append(obs[start:end, env])
                flat_act_list.append(actions[start:end, env])
                flat_logp_list.append(logprobs[start:end, env])
                flat_rew_list.append(rewards[start:end, env])
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
            flat_val_list.append(values[start:end, env])
            flat_old_probs_list.append(old_probs[start:end, env])

            lengths.append(end - start)
            last_values.append(next_value[env] * (1.0 - next_done[env].float()))

    dataset = {
        "observations": torch.cat(flat_obs_list, dim=0),
        "actions": torch.cat(flat_act_list, dim=0),
        "logprobs": torch.cat(flat_logp_list, dim=0),
        "rewards": torch.cat(flat_rew_list, dim=0),
        "values": torch.cat(flat_val_list, dim=0),
        "old_probs": torch.cat(flat_old_probs_list, dim=0),
        "last_values": torch.stack(last_values, dim=0),
        "lengths": np.asarray(lengths, dtype=np.int64),
    }

    dataset["end_indices"] = np.cumsum(dataset["lengths"])
    dataset["start_indices"] = np.insert(dataset["end_indices"], 0, 0)[:-1]
    return dataset


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

        obs_list, act_list, rew_list, prob_list, logp_list, val_list, last_list, splits = [], [], [], [], [], [], [], []

        for i in chosen:
            s, e = start_indices[i], end_indices[i]
            obs_list.append(dataset["observations"][s:e])
            act_list.append(dataset["actions"][s:e])
            rew_list.append(dataset["rewards"][s:e])
            prob_list.append(dataset["old_probs"][s:e])
            logp_list.append(dataset["logprobs"][s:e])
            val_list.append(dataset["values"][s:e])
            last_list.append(dataset["last_values"][i])
            splits.append(e - s)

        last_tensor = torch.stack(last_list, dim=0)

        yield (
            torch.cat(obs_list),
            torch.cat(act_list),
            torch.cat(rew_list),
            torch.cat(prob_list),
            torch.cat(logp_list),
            torch.cat(val_list),
            last_tensor,
            splits,
        )


def compute_value_loss(deltas, values, lasts, discount_matrix, discount_vector):
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
    for d, v, l in zip(deltas, values, lasts):
        T = len(d)
        y = discount_matrix[:T, :T].matmul(d) + l * discount_vector[-T:]
        targets.append(y)
        preds.append(v)
    return torch.cat(targets, dim=0), torch.cat(preds, dim=0)


def dae_update(
    data_obs,
    data_actions,
    data_rewards,
    old_dist,
    data_last_values,
    data_splits,
    clip_coef,
    ):

    new_values, new_advantages, new_dist, new_logprobs, entropy = agent.evaluate_state(
        data_obs, old_dist
    )
    new_values = new_values.flatten().split(data_splits)
    # print(f"data_rewards: {type(data_rewards)}, new_values: {type(new_values)}, data_last_values: {type(data_last_values)}, new_advantages: {type(new_advantages)}")
    # print(f"data_rewards: {data_rewards.shape}, new_advantages: {new_advantages.gather(dim=1, index=data_actions.long().view(-1, 1)).flatten().shape}")

    deltas = (
        data_rewards - new_advantages.gather(dim=1, index=data_actions.long().view(-1, 1)).flatten()
    ).split(data_splits)

    v_loss = compute_value_loss(
        deltas, new_values, data_last_values, discount_matrix, discount_vector
    )

    with torch.no_grad():
        y_true, y_pred = dae_targets_and_preds(
            deltas=deltas,
            values=new_values,
            lasts=data_last_values,
            discount_matrix=discount_matrix,
            discount_vector=discount_vector,
        )
    
    old_logprobs = torch.log(old_dist.clamp(min=1e-8))
    # kl_loss = (old_dist * (old_logprobs - new_logprobs)).sum(dim=1).mean()

    adv_for_policy = new_advantages.detach().clone()
    if args.norm_adv:
        adv_for_policy = normalize_advantage(adv_for_policy, old_dist)

    pg_loss, ratio = compute_policy_loss(
        args,
        adv_for_policy,
        new_logprobs,
        old_logprobs,
        data_actions.long().view(-1, 1),
        clip_range=clip_coef,
    )

    clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean()
    entropy_loss = entropy.mean()

    loss = (
        pg_loss
        - args.ent_coef * entropy_loss
        + args.vf_coef * v_loss
    )

    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    return (
        v_loss.detach(),
        pg_loss.detach(),
        entropy_loss.detach(),
        clipfrac.detach(),
        grad_norm.detach(),
        y_true.detach(),
        y_pred.detach(),
    )


if __name__ == "__main__":
    args = parse_args()
    print(args)

    args.num_iterations = args.total_timesteps // int(args.num_envs * args.num_steps)
    run_name = (
        f"{args.exp_name},seed={args.seed},lr={args.learning_rate},num_env={args.num_envs},"
        f"batch={args.batch_size},update_epochs={args.update_epochs},"
        f"entropy={args.ent_coef},model_type={args.model_type}"
    )

    wandb.init(
        project=args.project_name,
        name=f"{run_name}",
        config=vars(args),
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
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
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"


    ####### Agent #######
    agent = Agent(envs, device=device, model_type=args.model_type)
    # Make a version of agent with detached params
    agent_inference = Agent(envs, device=device, model_type=args.model_type)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)


    ####### Optimizer #######
    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    ####### Executables #######
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value


    # Compile policy
    if args.compile:
        mode = "reduce-overhead" if not args.cudagraphs else None
        policy = torch.compile(policy, mode=mode)
        # dae_update = torch.compile(dae_update, mode=mode)

    if args.cudagraphs:
        policy = CudaGraphModule(policy, warmup=20)
        # dae_update = CudaGraphModule(update, warmup=20)


    avg_returns = deque(maxlen=args.avg_returns_last_k)
    global_step = 0
    container_local = None
    next_obs = torch.tensor(envs.reset(), device=device, dtype=torch.uint8)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    max_ep_ret = -float("inf")

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


    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))
    desc = ""
    global_step_burnin = None
    global_start_time = time.time()


    for iteration in pbar:
        global_step_burnin = global_step
        start_time = time.time()

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        torch.compiler.cudagraph_mark_step_begin()
        next_obs, next_done, container = rollout(next_obs, next_done, avg_returns=avg_returns)
        global_step += container.numel()

        with torch.no_grad():
            next_value = get_value(next_obs).flatten()

        rollout_dataset = build_dae_dataset(container, next_done, next_value)


        all_targets = []
        all_preds = []

        mb_stats = {
            "clipfrac": [],
            "v_loss": [],
            "pg_loss": [],
            "entropy_loss": [],
            "grad_norm": [],
        }

        for epoch in range(args.update_epochs):
            for data in iter_traj_batches(rollout_dataset, batch_size=args.batch_size):
                (
                    data_obs,
                    data_actions,
                    data_rewards,
                    old_dist,
                    old_logprobs_action,
                    old_values,
                    data_last_values,
                    data_splits,
                ) = data

                (
                    v_loss,
                    pg_loss,
                    entropy_loss,
                    clipfrac,
                    grad_norm,
                    y_true,
                    y_pred,
                ) = dae_update(
                    data_obs,
                    data_actions,
                    data_rewards,
                    old_dist,
                    data_last_values,
                    data_splits,
                    args.clip_coef,
                )

                all_targets.append(y_true)
                all_preds.append(y_pred)
                mb_stats["clipfrac"].append(clipfrac)
                mb_stats["v_loss"].append(v_loss)
                mb_stats["pg_loss"].append(pg_loss)
                mb_stats["entropy_loss"].append(entropy_loss)
                mb_stats["grad_norm"].append(grad_norm)


        if len(all_targets) > 0:
            targets = torch.cat(all_targets, dim=0)
            preds = torch.cat(all_preds, dim=0)
            explained_var = explained_variance(y_pred=preds, y_true=targets).item()
        else:
            explained_var = float("nan")


        if iteration % args.log_freq == 0:
            cur_time = time.time()
            speed = (global_step - global_step_burnin) / (cur_time - start_time)
            global_step_burnin = global_step
            start_time = cur_time

            r = container["rewards"].mean().item()
            r_max = container["rewards"].max().item()
            avg_returns_t = float(np.mean(avg_returns)) if len(avg_returns) > 0 else float("nan")

            update_stats = {
                k: torch.stack(v).mean().item() for k, v in mb_stats.items()
            }

            with torch.no_grad():
                logs = {
                    "episode_return": avg_returns_t,
                    "logprobs": container["logprobs"].mean().item(),
                    # "vals": container["vals"].mean().item(),
                    "explained_variance": explained_var,
                    "clipfrac": update_stats["clipfrac"],
                    "policy_loss": update_stats["pg_loss"],
                    "value_loss": update_stats["v_loss"],
                    "entropy": update_stats["entropy_loss"],
                    "grad_norm": update_stats["grad_norm"],
                    "time": time.time() - global_start_time,
                }

            lr = optimizer.param_groups[0]["lr"]
            lr = lr.item() if torch.is_tensor(lr) else lr

            pbar.set_description(
                f"speed: {speed: 4.1f} sps, "
                f"reward avg: {r :4.2f}, "
                f"reward max: {r_max:4.2f}, "
                f"returns: {avg_returns_t: 4.2f},"
                f"lr: {lr: 4.2f}"
            )
            wandb.log(
                {"speed": speed, "lr": lr, **logs}, step=global_step
            )


    envs.close()
