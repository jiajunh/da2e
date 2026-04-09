# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy
import os

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import os
import random
import time
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
Categorical.logits = property(Categorical.__dict__["logits"].wrapped)
Categorical.probs = property(Categorical.__dict__["probs"].wrapped)

torch.set_float32_matmul_precision("high")




@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    project_name: str = "ppo_atari"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    total_timesteps: int = 10000000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    batch_size: int = 0
    minibatch_size: int = 256
    num_iterations: int = 0

    compile: bool = False
    cudagraphs: bool = False

    model_type: str = "natural"
    log_freq: int = 10
    avg_returns_last_k: int = 20


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

        self.actor = layer_init(nn.Linear(feature_dim, envs.single_action_space.n, device=device), std=0.01)
        self.critic = layer_init(nn.Linear(feature_dim, 1, device=device), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, obs, action=None):
        hidden = self.network(obs / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)



def gae(next_obs, next_done, container):
    # bootstrap value if not done
    next_value = get_value(next_obs).reshape(-1)
    lastgaelam = 0
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value
    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - cur_val
        advantages.append(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
        lastgaelam = advantages[-1]

        nextnonterminal = nextnonterminals[t]
        nextvalues = cur_val

    advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
    container["returns"] = advantages + vals
    return container


def rollout(obs, done, avg_returns=[]):
    ts = []
    for step in range(args.num_steps):
        torch.compiler.cudagraph_mark_step_begin()
        action, logprob, _, value = policy(obs=obs)
        next_obs_np, reward, next_done, info = envs.step(action.cpu().numpy())
        next_obs = torch.as_tensor(next_obs_np)
        reward = torch.as_tensor(reward)
        next_done = torch.as_tensor(next_done)

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
                rewards=reward,
                batch_size=(args.num_envs,),
            )
        )

        obs = next_obs = next_obs.to(device, non_blocking=True)
        done = next_done.to(device, non_blocking=True)

    container = torch.stack(ts, 0).to(device)
    return next_obs, done, container


def update(obs, actions, logprobs, advantages, returns, vals):
    optimizer.zero_grad()
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
    logratio = newlogprob - logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = vals + torch.clamp(
            newvalue - vals,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac, gn


update = tensordict.nn.TensorDictModule(
    update,
    in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
    out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
)



if __name__ == "__main__":
    args = tyro.cli(Args)
    print(args)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = (
        f"{args.exp_name},seed={args.seed},lr={args.learning_rate},num_env={args.num_envs},"
        f"batch={args.minibatch_size},update_epochs={args.update_epochs},"
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
    # Define networks: wrapping the policy in a TensorDictModule allows us to use CudaGraphModule
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    # Compile policy
    if args.compile:
        mode = "reduce-overhead" if not args.cudagraphs else None
        policy = torch.compile(policy, mode=mode)
        gae = torch.compile(gae, fullgraph=True, mode=mode)
        update = torch.compile(update, mode=mode)

    if args.cudagraphs:
        policy = CudaGraphModule(policy, warmup=20)
        #gae = CudaGraphModule(gae, warmup=20)
        update = CudaGraphModule(update, warmup=20)


    avg_returns = deque(maxlen=args.avg_returns_last_k)
    global_step = 0
    container_local = None
    next_obs = torch.tensor(envs.reset(), device=device, dtype=torch.uint8)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    max_ep_ret = -float("inf")
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

        torch.compiler.cudagraph_mark_step_begin()
        container = gae(next_obs, next_done, container)
        container_flat = container.view(-1)

        # explained variance
        with torch.no_grad():
            y_pred = container["vals"].reshape(-1)
            y_true = container["returns"].reshape(-1)
            var_y = torch.var(y_true)
            explained_var = torch.tensor(float("nan"), device=y_true.device)
            if var_y > 0:
                explained_var = 1 - torch.var(y_true - y_pred) / var_y


        # Optimizing the policy and value network
        mb_stats = {
            "clipfrac": [],
            "v_loss": [],
            "pg_loss": [],
            "entropy_loss": [],
            "gn": [],
        }


        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            for b in b_inds:
                container_local = container_flat[b]

                torch.compiler.cudagraph_mark_step_begin()
                out = update(container_local, tensordict_out=tensordict.TensorDict())
                
                for k in mb_stats:
                    mb_stats[k].append(out[k].detach())


        if iteration % args.log_freq == 0:
            cur_time = time.time()
            speed = (global_step - global_step_burnin) / (cur_time - start_time)
            global_step_burnin = global_step
            start_time = cur_time

            r = container["rewards"].mean().item()
            r_max = container["rewards"].max().item()
            avg_returns_t = float(np.mean(avg_returns)) if len(avg_returns) > 0 else float("nan")

            update_stats = {k: torch.stack(v).mean() for k, v in mb_stats.items()}

            with torch.no_grad():
                logs = {
                    "episode_return": avg_returns_t,
                    "logprobs": container["logprobs"].mean().item(),
                    # "advantages": container["advantages"].mean().item(),
                    # "returns": container["returns"].mean().item(),
                    # "vals": container["vals"].mean().item(),
                    "explained_variance": explained_var,
                    "clipfrac": update_stats["clipfrac"],
                    "policy_loss": update_stats["pg_loss"],
                    "value_loss": update_stats["v_loss"],
                    "entropy": update_stats["entropy_loss"],
                    "grad_norm": update_stats["gn"],
                    "time": time.time() - global_start_time,
                }

            lr = optimizer.param_groups[0]["lr"]
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
