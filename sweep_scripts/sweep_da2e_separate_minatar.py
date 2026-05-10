import os
import argparse
import uuid
import random
import itertools
import shortuuid

# -----------------------------------------------------------------------------
# Sweep generator for cleanrl/da2e_separate_minatar_sa.py
#       - dae
#       - pi with transformer, v / adv independent transformer
#       - no ppg like alignment

# -----------------------------------------------------------------------------
# Usage: tweak the lists below to define your grid. The script writes one command
# per configuration, splits them into SLURM array batches (≤999 tasks), and
# submits each array unless --dryrun is given.
# -----------------------------------------------------------------------------

su = shortuuid.ShortUUID()

# ───────────────────────────── CLI ──────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--nhrs", type=int, default=1, help="Job runtime in hours")
parser.add_argument("--mins", type=int, default=10, help="Job runtime in minutes")
parser.add_argument("--cpus", type=int, default=2, help="number of cpus")
parser.add_argument("--gpu", type=int, default=1, help="GPUs per task")
parser.add_argument("--memory", type=int, default=16, help="Memory in GiB")
parser.add_argument("--parallel", type=int, default=2, help="parallel_per_task")
parser.add_argument("--cluster", default="seas_gpu", choices=["sapphire", "gpu_test", "kempner_h100", "shared", "seas_compute", "seas_gpu", "gpu_requeue"])
parser.add_argument(
    "--base_save_dir", 
    default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/da2e/cleanrl/", 
    help="Project directory to cd into before running jobs")
parser.add_argument(
    "--save_root",
    default="/n/netscratch/kdbrantley_lab/Lab/jiajunh/da2e/sweep_files/",
    help="Scratch root where outputs, runs, and wandb data will be stored",
)
parser.add_argument("--output_dirname", default="output")
parser.add_argument("--wandb_mode", choices=["offline", "online"], default="online")
parser.add_argument("--dryrun", action="store_true")
args = parser.parse_args()

# ─────────────── GRID OF PARAMETERS ───────────────

env_ids = [
    # "MinAtar/Asterix-v0", 
    "MinAtar/Breakout-v0", 
    "MinAtar/Seaquest-v0", 
    # "MinAtar/SpaceInvaders-v0",
]

# Seeds
seeds = [4]

# Optional overrides for faster sweeps can be added here if desired
total_timesteps = 10000000
num_envs = 512 
num_steps = 128
gamma = 0.99
running_avg_coef = 0.05
kl_coef = 0.0
dropout = 0.0
weight_decay = 0.0
lr_schedule = "cosine" 


# ───────────── YAML-ALIGNED HYPERPARAMETERS ─────────────

learning_rate = [5.0e-4]
learning_rate_vf = [5.0e-4]
batch_size = [512, 2048]
update_epochs = [5]
clip_coef = [0.1]
ent_coef = [0.1, 0.2]
max_grad_norm = [1.0]
max_grad_norm_vf = [1.0]
vf_coef = [1.0]
pi_transformer_layers = [1, 1, 2] 
v_transformer_layers = [1, 2, 2]
pi_transformer_dim = [64, 64, 128, 128]
v_transformer_dim = [128, 256, 128, 256]
pi_num_heads = [1, 2, 2, 4]
v_num_heads = [4, 2, 4, 4]
cnn_feature_dim = [128]
warmup_frac = [0.10]
min_lr_ratio = [0.1]
beta1 = [0.9]
beta2 = [0.95]


# learning_rate = [5.0e-4]
# learning_rate_vf = [5.0e-4]
# batch_size = [1024]
# update_epochs = [5]
# clip_coef = [0.1]
# ent_coef = [0.1, 0.2]
# max_grad_norm = [1.0]
# max_grad_norm_vf = [1.0]
# vf_coef = [1.0]
# pi_transformer_layers = [2]
# v_transformer_layers = [2]
# pi_transformer_dim = [64, 128]
# v_transformer_dim = [32, 64]
# pi_num_heads = [4]
# v_num_heads = [4]
# cnn_feature_dim = [128]
# warmup_frac = [0.10]
# min_lr_ratio = [0.1]
# beta1 = [0.9]
# beta2 = [0.95]


# paired parameters
lr_pairs = list(zip(learning_rate, learning_rate_vf))
layers_pairs = list(zip(pi_transformer_layers, v_transformer_layers))
dim_pairs = list(zip(pi_transformer_dim, v_transformer_dim))
head_pairs = list(zip(pi_num_heads, v_num_heads))


# Cartesian product of parameters
param_grid = list(
    itertools.product(
        env_ids,
        seeds,
        lr_pairs,
        batch_size,
        update_epochs,
        clip_coef,
        ent_coef,
        max_grad_norm,
        max_grad_norm_vf,
        vf_coef,
        layers_pairs,
        dim_pairs,
        head_pairs,
        cnn_feature_dim,
        warmup_frac,
        min_lr_ratio,
        beta1,
        beta2,
    )
)
random.shuffle(param_grid)

# ───────── COMMAND TEMPLATE ─────────
CMD_TEMPLATE = (
    "python cleanrl/da2e_separate_minatar_sa.py "
    "--seed {seed} "
    "--env_id {env_id} "
    "--total_timesteps {total_timesteps} "
    "--wandb_project_name sweep_da2e_separate_{proj_name} "
    "--learning_rate {learning_rate} "
    "--learning_rate_vf {learning_rate_vf} "
    "--num_envs {num_envs} "
    "--num_steps {num_steps} "
    "--batch_size {batch_size} "
    "--update_epochs {update_epochs} "
    "--gamma {gamma} "
    "--clip_coef {clip_coef} "
    "--ent_coef {ent_coef} "
    "--max_grad_norm {max_grad_norm} "
    "--max_grad_norm_vf {max_grad_norm_vf} "
    "--running_avg_coef {running_avg_coef} "
    "--torch_deterministic "
    "--norm_adv "
    "--vf_coef {vf_coef} "
    "--kl_coef {kl_coef} "
    "--pi_transformer_layers {pi_transformer_layers} "
    "--v_transformer_layers {v_transformer_layers} "
    "--pi_transformer_dim {pi_transformer_dim} "
    "--v_transformer_dim {v_transformer_dim} "
    "--pi_num_heads {pi_num_heads} "
    "--v_num_heads {v_num_heads} "
    "--cuda "
    "--dropout {dropout} "
    "--cnn_feature_dim {cnn_feature_dim} "
    "--weight_decay {weight_decay} "
    "--lr_schedule {lr_schedule} "
    "--warmup_frac {warmup_frac} "
    "--min_lr_ratio {min_lr_ratio} "
    "--beta1 {beta1} "
    "--beta2 {beta2} "
    "--track "
)


# ──────────────── BUILD JOB LIST ─────────────────────
jobs = []

for (
    env_ids,
    seeds,
    lr_pairs,
    batch_size,
    update_epochs,
    clip_coef,
    ent_coef,
    max_grad_norm,
    max_grad_norm_vf,
    vf_coef,
    layers_pairs,
    dim_pairs,
    head_pairs,
    cnn_feature_dim,
    warmup_frac,
    min_lr_ratio,
    beta1,
    beta2,
) in param_grid:
    # unzip paired parameters
    learning_rate_, learning_rate_vf_ = lr_pairs
    pi_transformer_layers_, v_transformer_layers_ = layers_pairs
    pi_transformer_dim_, v_transformer_dim_ = dim_pairs
    pi_num_heads_, v_num_heads_ = head_pairs

    exp_name = (
        f"{env_ids}_seed{seeds}"
    )
    unique = su.random(length=8)
    cmd = CMD_TEMPLATE.format(
        seed=seeds,
        env_id=env_ids,
        total_timesteps=total_timesteps,
        proj_name=env_ids.replace("/", "_"),
        learning_rate=learning_rate_,
        learning_rate_vf=learning_rate_vf_,
        num_envs=num_envs,
        num_steps=num_steps,
        batch_size=batch_size,
        update_epochs=update_epochs,
        gamma=gamma,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        max_grad_norm=max_grad_norm,
        max_grad_norm_vf=max_grad_norm_vf,
        running_avg_coef=running_avg_coef,
        vf_coef=vf_coef,
        kl_coef=kl_coef,
        pi_transformer_layers=pi_transformer_layers_,
        v_transformer_layers=v_transformer_layers_,
        pi_transformer_dim=pi_transformer_dim_,
        v_transformer_dim=v_transformer_dim_,
        pi_num_heads=pi_num_heads_,
        v_num_heads=v_num_heads_,
        dropout=dropout,
        cnn_feature_dim=cnn_feature_dim,
        weight_decay=weight_decay,
        lr_schedule=lr_schedule,
        warmup_frac=warmup_frac,
        min_lr_ratio=min_lr_ratio,
        beta1=beta1,
        beta2=beta2,
    )

    jobs.append((cmd, f"{exp_name}_{unique}"))

print(f"Total runs: {len(jobs)}")
# print(jobs)

# ────────── OUTPUT & SLURM ARRAY GENERATION ───────────
# Direct all outputs to scratch save_root by default
out_dir = os.path.join(args.save_root, args.output_dirname)
runs_dir = os.path.join(args.save_root, "runs")
wandb_dir = os.path.join(args.save_root, "wandb")
wandb_cache_dir = os.path.join(args.save_root, "wandb_cache")
minatar_dir = os.path.join(out_dir, "MinAtar")

for d in [out_dir, runs_dir, wandb_dir, wandb_cache_dir, minatar_dir]:
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
    num_tasks = (count + args.parallel - 1) // args.parallel
    script = os.path.join(out_dir, f"sweep_da2e_separate_{batch_idx}.slurm")
    with open(script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=sweep_da2e_separate_minatari\n")
        f.write(f"#SBATCH -o {runs_dir}/out_%A_%a.out\n")
        f.write(f"#SBATCH -e {runs_dir}/err_%A_%a.err\n")
        f.write(f"#SBATCH --array=1-{num_tasks}\n")
        f.write(f"#SBATCH --time={args.nhrs}:{args.mins}:00\n")
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
            f.write(f"#SBATCH --gres=gpu:{args.gpu}\n")
            # f.write("#SBATCH --constraint=a100\n")
        elif args.cluster == "gpu_test":
            f.write("#SBATCH --partition=gpu_test\n")
            f.write("#SBATCH --account=kdbrantley_lab\n")
            f.write(f"#SBATCH --gres=gpu:{args.gpu}\n")
        elif args.cluster == "sapphire":
            f.write("#SBATCH --partition=sapphire\n")
            f.write("#SBATCH --account=kdbrantley_lab\n")
        elif args.cluster == "gpu_requeue":
            f.write("#SBATCH --partition=gpu_requeue\n")
            f.write("#SBATCH --account=kdbrantley_lab\n")
            # f.write("#SBATCH --constraint=nvidia_a100_1g.10gb|gpu:nvidia_a100_3g.20gb|\n")
            f.write(f"#SBATCH --gres=gpu:{args.gpu}\n")
            f.write(f"#SBATCH --open-mode=append\n")
        else:
            f.write("#SBATCH --partition=shared\n")
            f.write("#SBATCH --account=kdbrantley_lab\n")
        if args.memory > 0:
            f.write(f"#SBATCH --mem={args.memory}G\n")
        f.write(f"#SBATCH --cpus-per-task={args.cpus}\n")

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

        f.write(f"PARALLEL_JOBS={args.parallel}\n")
        f.write('START=$(( (SLURM_ARRAY_TASK_ID - 1) * PARALLEL_JOBS + 1 ))\n')
        f.write('END=$(( SLURM_ARRAY_TASK_ID * PARALLEL_JOBS ))\n')
        f.write(f'TOTAL=$(wc -l < "{now_file}")\n')
        f.write('if [ "$END" -gt "$TOTAL" ]; then END=$TOTAL; fi\n')
        f.write('export START END\n')
        f.write(f'export NOW_FILE="{now_file}"\n')
        f.write(f'export LOG_FILE="{log_file}"\n')
        f.write(f'export ERR_FILE="{err_file}"\n\n')

        f.write("srun bash -c '\n")
        f.write('for line_no in $(seq "$START" "$END"); do\n')
        f.write('    cmd=$(sed -n "${line_no}p" "$NOW_FILE")\n')
        f.write('    log=$(sed -n "${line_no}p" "$LOG_FILE")\n')
        f.write('    err=$(sed -n "${line_no}p" "$ERR_FILE")\n')
        f.write('    echo "Launching line $line_no"\n')
        f.write('    bash -c "$cmd" >"$log" 2>"$err" &\n')
        f.write('done\n')
        f.write('wait\n')
        f.write("'\n")


    print(script)

    if not args.dryrun:
        print(f"sbatch {script} &")
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
