import isaacgym

import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
import json
from common.util import (
    omegaconf_to_dict,
    print_dict,
    fix_wandb,
    update_dict,
    AverageMeter,
)

from run import get_agent

import torch
import numpy as np

import wandb

from tkinter import *

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-p", "--path", help="model save path", type=str, required=True)
parser.add_argument(
    "-c",
    "--checkpoint",
    help="specific saved model e.g. model10",
    type=str,
    required=True,
)
parser.add_argument("-n", "--num_envs", type=int, default=1)

# Read arguments from command line
args = parser.parse_args()


class PlayUI:
    def __init__(self, cfg_dict, model_path) -> None:
        self.root = Tk()
        self.root.title("test")
        self.root.geometry("300x500")

        # init and load agent
        self.agent = get_agent(cfg_dict)
        self.agent.load_torch_model(model_path)

        # check if single task agent
        self.isSingleTask = cfg_dict["agent"]["name"] == "ppo"

        if not self.isSingleTask:
            self.weights = self.agent.task.Eval.W.clone()
            self.weightLabels = cfg_dict["env"]["task"]["taskLabels"]
            self.generate_scales()

        self.rew = None
        self.print_step_reward()

    def weight_update_function(self, dimension):
        def update_val(val):
            self.weights[..., dimension] = float(val)
            self.agent.task.Eval.W[:] = self.weights[:]
            self.agent.task.Eval.W = (
                self.agent.task.Eval.W / self.agent.task.Eval.W.norm(1, 1, keepdim=True)
            )

        return update_val

    def target_update_function(self, dimension):
        def update_val(val):
            self.agent.env.goal_pos[..., dimension] = float(val)

        return update_val

    def add_scale(self, dimension, gen_func, label, range=(0, 1), type="weight"):
        scale = Scale(
            self.root,
            from_=range[0],
            to=range[1],
            digits=3,
            resolution=0.01,
            label=label,
            orient=HORIZONTAL,
            command=gen_func(dimension),
        )
        if type == "weight":
            scale.set(self.agent.task.Eval.W[0, dimension].item())
        scale.pack()

    def generate_scales(self):
        for i, label in enumerate(self.weightLabels):
            self.add_scale(
                dimension=i, gen_func=self.weight_update_function, label=label
            )

        self.add_scale(
            dimension=0,
            gen_func=self.target_update_function,
            label="target pos",
            range=(-2, 2),
            type="target",
        )

    def print_step_reward(self):
        self.rew = DoubleVar(name="reward")  # instantiate the IntVar variable class
        self.rew.set(0.0)  # set it to 0 as the initial value

        # the label's textvariable is set to the variable class instance
        Label(self.root, text="step reward").pack()
        Label(self.root, textvariable=self.rew).pack()

    def _debug_ui(self):
        # only runs UI loop without inference
        while True:
            self.root.update_idletasks()
            self.root.update()

            print(self.agent.task.Eval.W[0])

    def play(self):
        if self.isSingleTask:
            return self.play_ppo()

        avgStepRew = AverageMeter(1, 20).to(self.agent.device)
        while True:
            s = self.agent.reset_env()
            for _ in range(5000):
                self.root.update_idletasks()
                self.root.update()

                a = self.agent.act(s, self.agent.task.Eval, "exploit")
                print(self.agent.task.Eval.W[0])

                self.agent.env.step(a)
                s_next = self.agent.env.obs_buf.clone()
                self.agent.env.reset()

                r = self.agent.calc_reward(s_next, self.agent.task.Eval.W)
                s = s_next
                avgStepRew.update(r)
                if self.rew:
                    self.rew.set(avgStepRew.get_mean())

    # play for single task
    def play_ppo(self):
        avgStepRew = AverageMeter(1, 20).to(self.agent.device)
        self.agent.agent.eval()
        while True:
            next_obs = self.agent.env.obs_buf
            for _ in range(5000):
                with torch.no_grad():
                    action, _, _, _ = self.agent.agent.get_action_and_value(next_obs)

                self.agent.env.step(action)
                next_obs, reward = (
                    self.agent.env.obs_buf,
                    self.agent.env.reward_buf,
                )
                self.agent.env.reset()

                avgStepRew.update(reward)
                if self.rew:
                    self.rew.set(avgStepRew.get_mean())


def modify_cfg(cfg_dict):
    cfg_dict["agent"]["norm_task_by_sf"] = False
    cfg_dict["agent"]["phase"] = 1

    cfg_dict["buffer"]["n_env"] = cfg_dict["env"]["num_envs"]
    cfg_dict["buffer"]["min_n_experience"] = 0

    # cfg_dict["env"]["episode_max_step"] = int(50 * (512 / cfg_dict["env"]["num_envs"]))
    cfg_dict["env"]["task"]["rand_task"] = False
    cfg_dict["env"]["task"]["rand_vel_targets"] = False
    cfg_dict["env"]["mode"] = "play"
    cfg_dict["env"]["sim"]["headless"] = False
    cfg_dict["env"]["num_envs"] = args.num_envs

    if "aero" in cfg_dict["env"]:
        cfg_dict["env"]["aero"]["wind_mag"] = 0
    if "domain_rand" in cfg_dict["env"]["task"]:
        cfg_dict["env"]["task"]["domain_rand"] = False
    print_dict(cfg_dict)

    return cfg_dict


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)

    wandb.init(mode="disabled")
    wandb_dict = fix_wandb(wandb.config)

    # print_dict(wandb_dict)
    update_dict(cfg_dict, wandb_dict)

    cfg_dict = modify_cfg(cfg_dict)

    model_path = "/home/yutang/rl/sf_mutitask_rl/logs/rmacompblimp/BlimpRand/2023-12-23-07-15-08/model90"

    playob = PlayUI(cfg_dict, model_path)
    playob.play()

    wandb.finish()


def launch_play():
    model_folder = args.path
    model_checkpoint = args.checkpoint

    cfg_path = model_folder + "/cfg"
    model_path = model_folder + "/" + model_checkpoint

    cfg_dict = None
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    cfg_dict = modify_cfg(cfg_dict)
    print(cfg_dict)

    playob = PlayUI(cfg_dict, model_path)
    playob.play()


if __name__ == "__main__":
    torch.manual_seed(456)
    np.random.seed(456)

    # launch_rlg_hydra()
    launch_play()
