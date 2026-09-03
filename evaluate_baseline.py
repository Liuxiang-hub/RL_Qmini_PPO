"""Quantitative evaluation for a frozen Qmini policy.

The script never updates policy weights.  It fixes the command seen by both the
policy and simulator, runs parallel rollouts, and writes raw trajectories plus
per-rollout and aggregate metrics for thesis experiments.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import isaacgym
import numpy as np
import pandas as pd
import torch
from isaacgym import gymtorch, gymutil

from env import GymEnvWrapper, LeggedRobotEnv
from env.tasks import load_task_cls
from env.utils.helpers import class_to_dict, parse_sim_params, set_seed
from model import load_actor
from utils.yaml import ParamsProcess
import importlib


SCENARIOS = {
    "stand": {"vx": 0.0, "yaw": 0.0},
    "forward_02": {"vx": 0.2, "yaw": 0.0},
    "forward_05": {"vx": 0.5, "yaw": 0.0},
    "speed_transition": {"vx": 0.0, "yaw": 0.0},
    "turn": {"vx": 0.0, "yaw": 0.5},
    "walk_turn": {"vx": 0.5, "yaw": 0.5},
    "push_forward": {
        "vx": 0.5, "yaw": 0.0, "push": "forward", "push_velocity": -1.0,
    },
    "push_lateral": {
        "vx": 0.3, "yaw": 0.0, "push": "lateral", "push_velocity": 1.0,
    },
}


def get_args():
    custom = [
        {"name": "--name", "type": str, "default": "q2"},
        {"name": "--scenario", "type": str, "default": "stand",
         "choices": list(SCENARIOS)},
        {"name": "--num_envs", "type": int, "default": 30},
        {"name": "--duration", "type": float, "default": 10.0},
        {"name": "--seed", "type": int, "default": 202601},
        {"name": "--push_time", "type": float, "default": 4.0},
        {"name": "--push_velocity", "type": float, "default": None},
        {"name": "--command_vx", "type": float, "default": None},
        {"name": "--render", "action": "store_true", "default": False},
        {"name": "--fix_cam", "action": "store_true", "default": False},
        {"name": "--output", "type": str, "default": "evaluation"},
    ]
    args = gymutil.parse_arguments(
        description="Evaluate a frozen Qmini policy", custom_parameters=custom
    )
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += ":{}".format(args.sim_device_id)
    args.rl_device = args.sim_device
    if args.render and args.num_envs != 1:
        raise ValueError("Visual evaluation requires --num_envs 1")
    return args


def command_at(scenario: str, t: float, command_vx=None):
    if scenario == "speed_transition":
        if t < 2.0:
            return 0.0, 0.0
        if t < 5.0:
            return 0.2, 0.0
        if t < 8.0:
            return 0.5, 0.0
        return 0.0, 0.0
    cfg = SCENARIOS[scenario]
    vx = cfg["vx"] if command_vx is None else command_vx
    return vx, cfg["yaw"]


def set_command(task, vx: float, yaw: float):
    task.commands.zero_()
    task.commands[:, 0] = vx
    task.commands[:, 2] = yaw
    moving = float(abs(vx) >= 0.15 or abs(yaw) >= 0.15)
    task.static_flag.fill_(moving)


def repair_command_observation(task):
    """Replace the newest history frame after the task's random resampling."""
    task.obs_history[-1] = task.pure_observation()
    return torch.cat(list(task.obs_history), dim=-1).float()


def apply_push(env, direction: str, magnitude: float):
    axis = 0 if direction == "forward" else 1
    env.root_states[:, 7 + axis] += magnitude
    env.gym.set_actor_root_state_tensor(
        env.sim, gymtorch.unwrap_tensor(env.root_states)
    )


def build_environment(args):
    exp_dir = Path("experiments") / args.name
    model_dir = exp_dir / "model"
    params_process = ParamsProcess()
    params = params_process.read_param(str(model_dir / "cfg.yaml"))
    config_name = params["task"]["cfg"]
    cfg = getattr(importlib.import_module("config." + config_name), config_name)
    cfg = params_process.dict2class(cfg, params)

    cfg.runner.num_envs = args.num_envs
    # Keep the environment timeout outside the measurement window. Otherwise
    # its final timeout/reset can be misclassified as a fall on the last step.
    cfg.runner.episode_length_s = args.duration + 1.0
    cfg.terrain.num_rows = 5
    cfg.terrain.num_cols = 5
    cfg.noise_values.randomize_noise = False
    cfg.domain_rand.delay_observation = False
    cfg.domain_rand.push_robots = False
    cfg.domain_rand.randomize_damping = False
    cfg.domain_rand.randomize_mass = False
    cfg.domain_rand.randomize_friction = False
    cfg.domain_rand.randomize_gains = False
    cfg.domain_rand.randomize_torque = False
    cfg.init_state.random_rot = False

    set_seed(args.seed)
    sim_params = parse_sim_params(args, class_to_dict(cfg.sim))
    env = LeggedRobotEnv(
        cfg=cfg, sim_params=sim_params, physics_engine=args.physics_engine,
        sim_device=args.sim_device, render=args.render, debug=False,
        fix_cam=args.fix_cam, epochs=1,
    )
    task = load_task_cls(cfg.task.cfg)(env)
    gym_env = GymEnvWrapper(env, task, debug=False)
    task.num_observations = len(task.pure_observation()[0]) * task.obs_history.maxlen
    task.num_actions = len(task.action_low)

    actor = load_actor(class_to_dict(cfg.policy), args.rl_device).eval()
    checkpoint = torch.load(str(model_dir / "policy.pt"), map_location=args.rl_device)
    actor.load_state_dict(checkpoint["actor"])
    return cfg, env, task, gym_env, actor


def evaluate(args):
    cfg, env, task, gym_env, actor = build_environment(args)
    dt = cfg.sim.dt * cfg.pd_gains.decimation
    scenario_cfg = SCENARIOS[args.scenario]
    push_velocity = (
        scenario_cfg.get("push_velocity", 0.5)
        if args.push_velocity is None else args.push_velocity
    )
    # Match the environment's integer episode horizon. A timeout is successful
    # completion, not a fall; the reported duration remains the requested value.
    max_steps = int(args.duration / dt)
    ids = torch.arange(args.num_envs, device=args.rl_device)
    obs, _ = gym_env.reset(ids)
    vx_cmd, yaw_cmd = command_at(args.scenario, 0.0, args.command_vx)
    set_command(task, vx_cmd, yaw_cmd)
    for _ in range(task.obs_history.maxlen):
        task.obs_history.append(task.pure_observation())
    obs = torch.cat(list(task.obs_history), dim=-1).float()

    alive = torch.ones(args.num_envs, dtype=torch.bool, device=args.rl_device)
    first_failure_step = torch.full(
        (args.num_envs,), max_steps, dtype=torch.long, device=args.rl_device
    )
    energy = torch.zeros(args.num_envs, device=args.rl_device)
    recovery_step = torch.full_like(first_failure_step, -1)
    recovery_streak = torch.zeros_like(first_failure_step)
    pushed = False
    records = []

    for step in range(max_steps):
        t = step * dt
        vx_cmd, yaw_cmd = command_at(args.scenario, t, args.command_vx)
        set_command(task, vx_cmd, yaw_cmd)
        obs = repair_command_observation(task)

        if "push" in SCENARIOS[args.scenario] and not pushed and t >= args.push_time:
            apply_push(env, scenario_cfg["push"], push_velocity)
            pushed = True

        with torch.inference_mode():
            action = actor(obs)["act"].detach()
        obs, _, reward, done, info, _ = gym_env.step(action)
        set_command(task, vx_cmd, yaw_cmd)

        roll = task.base_euler[:, 0]
        pitch = task.base_euler[:, 1]
        roll_rate = task.base_ang_vel[:, 0]
        pitch_rate = task.base_ang_vel[:, 1]
        vx = task.base_lin_vel[:, 0]
        yaw_rate = task.base_ang_vel[:, 2]
        height = env.base_pos_hd[:, 2]
        # `react_tau` is zero for this URDF unless force sensors are configured
        # before tensor acquisition. The task's PD torque estimate is available
        # consistently in both baseline and proposed-method evaluations.
        power = torch.sum(torch.abs(task.joint_tau * task.joint_vel), dim=1)
        energy += power * dt * alive.float()

        done_now = done.reshape(-1).bool()
        timeout_now = info.get("timeouts", env.time_out_buf).reshape(-1).bool()
        failure_now = done_now & ~timeout_now & alive
        first_failure_step[failure_now] = step + 1
        alive &= ~failure_now

        if pushed:
            recovered = (
                (torch.abs(roll) < np.deg2rad(5.0))
                & (torch.abs(pitch) < np.deg2rad(5.0))
                & (torch.abs(vx - vx_cmd) < 0.1)
                & alive
            )
            recovery_streak = torch.where(recovered, recovery_streak + 1, torch.zeros_like(recovery_streak))
            newly_recovered = (recovery_step < 0) & (recovery_streak >= int(0.5 / dt))
            recovery_step[newly_recovered] = step + 1

        arrays = [x.detach().cpu().numpy() for x in
                  (roll, pitch, roll_rate, pitch_rate, vx, yaw_rate, height, reward, alive)]
        for env_id in range(args.num_envs):
            records.append({
                "env_id": env_id, "step": step, "time_s": t + dt,
                "cmd_vx": vx_cmd, "cmd_yaw_rate": yaw_cmd,
                "roll_rad": arrays[0][env_id], "pitch_rad": arrays[1][env_id],
                "roll_rate": arrays[2][env_id], "pitch_rate": arrays[3][env_id],
                "vx": arrays[4][env_id], "yaw_rate": arrays[5][env_id],
                "height_m": arrays[6][env_id], "reward": arrays[7][env_id],
                "alive": bool(arrays[8][env_id]),
            })

    raw = pd.DataFrame.from_records(records)
    runs = []
    failure_steps = first_failure_step.detach().cpu().numpy()
    recovery_steps = recovery_step.detach().cpu().numpy()
    energy_np = energy.detach().cpu().numpy()
    for env_id, data in raw.groupby("env_id", sort=True):
        valid = data.iloc[:failure_steps[env_id]]
        survived = failure_steps[env_id] >= max_steps
        angle_sq = valid.roll_rad.pow(2) + valid.pitch_rad.pow(2)
        rate_sq = valid.roll_rate.pow(2) + valid.pitch_rate.pow(2)
        recovery_time = np.nan
        if recovery_steps[env_id] >= 0:
            recovery_time = max(0.0, recovery_steps[env_id] * dt - args.push_time - 0.5)
        runs.append({
            "scenario": args.scenario, "seed": args.seed, "env_id": env_id,
            "completed_duration": bool(survived),
            "survival_time_s": args.duration if survived else failure_steps[env_id] * dt,
            "angle_rmse_rad": float(np.sqrt(angle_sq.mean())),
            "roll_rmse_rad": float(np.sqrt(valid.roll_rad.pow(2).mean())),
            "pitch_rmse_rad": float(np.sqrt(valid.pitch_rad.pow(2).mean())),
            "max_abs_angle_rad": float(np.maximum(valid.roll_rad.abs(), valid.pitch_rad.abs()).max()),
            "angular_rate_rmse": float(np.sqrt(rate_sq.mean())),
            "vx_rmse": float(np.sqrt((valid.vx - valid.cmd_vx).pow(2).mean())),
            "yaw_rate_rmse": float(np.sqrt((valid.yaw_rate - valid.cmd_yaw_rate).pow(2).mean())),
            "min_height_m": float(valid.height_m.min()),
            "energy_abs_tau_qdot_j": float(energy_np[env_id]),
            "recovered": bool(recovery_steps[env_id] >= 0) if pushed else np.nan,
            "recovery_time_s": recovery_time,
        })
    per_run = pd.DataFrame(runs)
    numeric = per_run.select_dtypes(include=[np.number, bool]).drop(columns=["seed", "env_id"])
    summary = pd.DataFrame({
        "mean": numeric.mean(), "std": numeric.std(ddof=0),
        "median": numeric.median(), "min": numeric.min(), "max": numeric.max(),
    })

    out = Path(args.output) / ("baseline_" + args.name) / args.scenario / str(args.seed)
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "trajectories.csv", index=False)
    per_run.to_csv(out / "metrics_by_run.csv", index=False)
    summary.to_csv(out / "metrics_summary.csv")
    metadata = vars(args).copy()
    metadata.update({"policy": str(Path("experiments") / args.name / "model/policy.pt"),
                     "dt": dt, "steps": max_steps,
                     "effective_command_vx": command_at(args.scenario, 0.0, args.command_vx)[0],
                     "effective_push_velocity": push_velocity})
    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    print("\nEvaluation complete:", out)
    print(summary.loc[["completed_duration", "survival_time_s", "angle_rmse_rad",
                       "angular_rate_rmse", "vx_rmse", "yaw_rate_rmse"]])


if __name__ == "__main__":
    evaluate(get_args())
