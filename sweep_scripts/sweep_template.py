import os
import argparse
import uuid
import random
import itertools
import shortuuid

# -----------------------------------------------------------------------------
# Sweep generator for cleanrl/rda_atari.py
# -----------------------------------------------------------------------------
# Usage: tweak the lists below to define your grid. The script writes one command
# per configuration, splits them into SLURM array batches (≤999 tasks), and
# submits each array unless --dryrun is given.
# -----------------------------------------------------------------------------

su = shortuuid.ShortUUID()

# ───────────────────────────── CLI ──────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--nhrs", type=int, default=24, help="Job runtime in hours")
parser.add_argument("--gpu", type=int, default=1, help="GPUs per task")
parser.add_argument("--memory", type=int, default=240, help="Memory in GiB")
parser.add_argument("--cluster", choices=["kempner_h100", "shared", "seas_compute", "seas_gpu"], default="sapphire")
parser.add_argument("--base_save_dir", default=os.getcwd(), help="Project directory to cd into before running jobs")
parser.add_argument(
    "--save_root",
    default="/n/netscratch/kdbrantley_lab/Lab/npeng",
    help="Scratch root where outputs, runs, and wandb data will be stored",
)
parser.add_argument("--output-dirname", default="output")
parser.add_argument("--wandb_mode", choices=["offline", "online"], default="online")
parser.add_argument("--dryrun", action="store_true")
args = parser.parse_args()

# ─────────────── GRID OF PARAMETERS ───────────────
# Use NoFrameskip-v4 equivalents for 14 Atari games
# env_ids = [
#     "PongNoFrameskip-v4",
#     "BreakoutNoFrameskip-v4",
#     "SpaceInvadersNoFrameskip-v4",
#     "SeaquestNoFrameskip-v4",
#     "QbertNoFrameskip-v4",
#     "BeamRiderNoFrameskip-v4",
#     "EnduroNoFrameskip-v4",
#     "MsPacmanNoFrameskip-v4",
#     "FreewayNoFrameskip-v4",
#     "AsterixNoFrameskip-v4",
#     "MontezumaRevengeNoFrameskip-v4",
#     "PitfallNoFrameskip-v4",
#     "PrivateEyeNoFrameskip-v4",
#     "GravitarNoFrameskip-v4",
# ]

# env_ids = [
#     "ALE/Pong-v5",
#     "ALE/Breakout-v5",
#     "ALE/SpaceInvaders-v5",
#     "ALE/Seaquest-v5",
#     "ALE/Qbert-v5",
#     "ALE/BeamRider-v5",
#     "ALE/Enduro-v5",
#     "ALE/MsPacman-v5",
#     "ALE/Freeway-v5",
#     "ALE/Asterix-v5",
#     "ALE/MontezumaRevenge-v5",
#     "ALE/Pitfall-v5",
#     "ALE/PrivateEye-v5",
#     "ALE/Gravitar-v5",
# ]

# env_ids = [
#     "MontezumaRevenge-v5",
# ]

env_ids = [
    # "Pong-v5",
    # "BeamRider-v5",
    # "Breakout-v5",
    # "SpaceInvaders-v5",
    # "Seaquest-v5",
    # "Qbert-v5",
    # "Enduro-v5",
    # "MsPacman-v5",
    # "Freeway-v5",
    # "Asterix-v5",
    "MontezumaRevenge-v5",
    # "Pitfall-v5",
    # "PrivateEye-v5",
    # "Gravitar-v5",
]

# Seeds
seeds = [4]

# Optional overrides for faster sweeps can be added here if desired
total_timesteps = 10_000_000  # match rda_atari_envpool.py default
num_envs = 4  # default in rda_atari_envpool.py

# ───────────── YAML-ALIGNED HYPERPARAMETERS ─────────────
# These mirror cleanrl/scripts/rda_atari_sweep.yaml
num_steps_list = [256]
learning_rates = [2.5e-4]
vf_coefs = [0.5]
max_grad_norms = [2]
clip_coefs = [0.1]
clip_vlosses = [False]
gae_lambdas = [0.85]
gae_lambda_frac_floors = [1.0]
alphas = [0.5]
taus = [0.005]
buffer_sizes = [300_000]
update_epochs_list = [2]
adv_minibatch_sizes = [None]
anneal_lrs = [True]
anneal_lambdas = [True]
separate_trunks_list = [True]
conv1_channels_list = [32]
conv2_channels_list = [64]
conv3_channels_list = [64]

# No fixed toggles here; rely on defaults in rda_atari_envpool.py

# Cartesian product of parameters
param_grid = list(
    itertools.product(
        env_ids,
        seeds,
        num_steps_list,
        learning_rates,
        vf_coefs,
        max_grad_norms,
        clip_coefs,
        clip_vlosses,
        gae_lambdas,
        gae_lambda_frac_floors,
        alphas,
        taus,
        buffer_sizes,
        update_epochs_list,
        adv_minibatch_sizes,
        anneal_lrs,
        anneal_lambdas,
        separate_trunks_list,
        conv1_channels_list,
        conv2_channels_list,
        conv3_channels_list,
    )
)
random.shuffle(param_grid)

# ───────── COMMAND TEMPLATE ─────────
CMD_TEMPLATE = (
    "poetry run python cleanrl/rda_atari_envpool.py "
    "--env_id {env_id} "
    "--seed {seed} "
    "--total_timesteps {total_timesteps} "
    "--num_envs {num_envs} "
    "--num_steps {num_steps} "
    "--learning_rate {learning_rate} "
    "--vf_coef {vf_coef} "
    "--max_grad_norm {max_grad_norm} "
    "--clip_coef {clip_coef} "
    "--gae_lambda {gae_lambda} "
    "--gae_lambda_frac_floor {gae_lambda_frac_floor} "
    "--alpha {alpha} "
    "--tau {tau} "
    "--buffer_size {buffer_size} "
    "--update_epochs {update_epochs} "
    "--adv_minibatch_size {adv_minibatch_size} "
    "--conv1_channels {conv1_channels} "
    "--conv2_channels {conv2_channels} "
    "--conv3_channels {conv3_channels} "
    "--wandb_project_name rda_atari_ablation_buffer_size "
)

# ──────────────── BUILD JOB LIST ─────────────────────
jobs = []
for (
    env_id,
    seed,
    num_steps,
    learning_rate,
    vf_coef,
    max_grad_norm,
    clip_coef,
    clip_vloss,
    gae_lambda,
    gae_lambda_frac_floor,
    alpha,
    tau,
    buffer_size,
    update_epochs,
    adv_minibatch_size,
    anneal_lr,
    anneal_lambda,
    separate_trunks,
    conv1_channels,
    conv2_channels,
    conv3_channels,
) in param_grid:
    exp_name = (
        f"{env_id}_seed{seed}"
    )
    unique = su.random(length=8)
    cmd = CMD_TEMPLATE.format(
        env_id=env_id,
        seed=seed,
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        num_steps=num_steps,
        learning_rate=learning_rate,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        clip_coef=clip_coef,
        gae_lambda=gae_lambda,
        gae_lambda_frac_floor=gae_lambda_frac_floor,
        alpha=alpha,
        tau=tau,
        buffer_size=buffer_size,
        update_epochs=update_epochs,
        adv_minibatch_size=adv_minibatch_size,
        conv1_channels=conv1_channels,
        conv2_channels=conv2_channels,
        conv3_channels=conv3_channels,
    )
    # Append Tyro-style boolean toggles
    def toggle(flag: str, value: bool) -> str:
        return f" --{flag}" if value else f" --no-{flag}"

    cmd += toggle("clip_vloss", clip_vloss)
    cmd += toggle("anneal_lr", anneal_lr)
    cmd += toggle("anneal_lambda", anneal_lambda)
    cmd += toggle("separate_trunks", separate_trunks)
    jobs.append((cmd, f"{exp_name}_{unique}"))

print(f"Total runs: {len(jobs)}")

# ────────── OUTPUT & SLURM ARRAY GENERATION ───────────
# Direct all outputs to scratch save_root by default
out_dir = os.path.join(args.save_root, args.output_dirname)
runs_dir = os.path.join(args.save_root, "runs")
wandb_dir = os.path.join(args.save_root, "wandb")
wandb_cache_dir = os.path.join(args.save_root, "wandb_cache")

for d in [out_dir, runs_dir, wandb_dir, wandb_cache_dir]:
    os.makedirs(d, exist_ok=True)

print("Output directory:", out_dir)


def _tmp(prefix: str) -> str:
    return os.path.join(out_dir, f"{prefix}_{uuid.uuid4()}.txt")

now_file, log_file, err_file = _tmp("now"), _tmp("log"), _tmp("err")
threshold, written, batch_idx = 999, 0, 0  # write up to 999 jobs per array


def flush(count: int):
    """Emit a SLURM script for the last <count> commands."""
    global written, batch_idx, now_file, log_file, err_file
    if count == 0:
        return
    batch_idx += 1
    script = os.path.join(out_dir, f"sweep_rda_atari_{batch_idx}.slurm")
    with open(script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=rda_atari_sweep\n")
        f.write(f"#SBATCH -o {runs_dir}/out_%A_%a.out\n")
        f.write(f"#SBATCH -e {runs_dir}/err_%A_%a.err\n")
        f.write(f"#SBATCH --array=1-{count}\n")
        f.write(f"#SBATCH --time={args.nhrs}:00:00\n")
        if args.cluster == "kempner_h100":
            f.write("#SBATCH --partition=kempner_h100\n")
            f.write("#SBATCH --account=kempner_kdbrantley_lab\n")
            f.write(f"#SBATCH --gres=gpu:{args.gpu}\n")
        elif args.cluster == "seas_compute":
            f.write("#SBATCH --partition=seas_compute\n")
            f.write("#SBATCH --account=kdbrantley_lab\n")
        elif args.cluster == "seas_gpu":
            f.write("#SBATCH --partition=seas_gpu\n")
            f.write("#SBATCH --account=kdbrantley_lab\n")
            f.write("#SBATCH --constraint=h200\n")
            f.write(f"#SBATCH --gres=gpu:{args.gpu}\n")
        else:
            f.write("#SBATCH --partition=shared\n")
            f.write("#SBATCH --account=kdbrantley_lab\n")
        if args.memory > 0:
            f.write(f"#SBATCH --mem={args.memory}G\n")
        f.write("#SBATCH --cpus-per-task=8\n")
        f.write("#SBATCH --mail-user=nianli_peng@g.harvard.edu\n")
        f.write("#SBATCH --mail-type=BEGIN,END,FAIL\n\n")
        f.write("source /n/sw/Mambaforge-23.11.0-0/etc/profile.d/conda.sh\n")
        f.write("conda activate cleanrl\n")
        f.write("module load cuda\n\n")
        # Respect requested W&B mode and store data on scratch
        if args.wandb_mode == "offline":
            f.write("export WANDB_MODE=offline\n")
        else:
            f.write("export WANDB_MODE=online\n")
        f.write(f"export WANDB_DIR={wandb_dir}\n")
        f.write(f"export WANDB_DATA_DIR={wandb_dir}\n")
        f.write(f"export WANDB_CACHE_DIR={wandb_cache_dir}\n\n")
        # Ensure training code writes TensorBoard logs to scratch
        f.write(f"export SCRATCH_ROOT={args.save_root}\n\n")
        f.write("unset WANDB_X_REQUIRE_LEGACY_SERVICE || true\n\n")
        f.write(f"cd {args.base_save_dir}\n")
        f.write(
            "srun --output=$(head -n $SLURM_ARRAY_TASK_ID "
            f"{log_file} | tail -n 1) "
            "--error=$(head -n $SLURM_ARRAY_TASK_ID "
            f"{err_file} | tail -n 1) "
            "$(head -n $SLURM_ARRAY_TASK_ID "
            f"{now_file} | tail -n 1)\n"
        )
    if not args.dryrun:
        os.system(f"sbatch {script} &")
    # reset collectors
    now_file, log_file, err_file = _tmp("now"), _tmp("log"), _tmp("err")
    written = 0

# write commands and split into batches
for cmd, name in jobs:
    with open(now_file, "a") as nf, open(log_file, "a") as lf, open(err_file, "a") as ef:
        nf.write(cmd + "\n")
        lf.write(os.path.join(out_dir, name) + ".log\n")
        ef.write(os.path.join(out_dir, name) + ".err\n")
    written += 1
    if written == threshold:
        flush(written)

# flush any remaining jobs
flush(written)
