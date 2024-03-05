import isaacgym

import hydra
from omegaconf import DictConfig, OmegaConf
from common.util import omegaconf_to_dict, print_dict, fix_wandb, update_dict


import torch
import numpy as np

import wandb


def get_agent(cfg_dict):
    if "sac" in cfg_dict["agent"]["name"].lower():
        from agents.sac import SACAgent

        agent = SACAgent(cfg=cfg_dict)

    elif "ppo" in cfg_dict["agent"]["name"].lower():
        from agents.ppo import PPO_agent

        agent = PPO_agent(cfg=cfg_dict)

    elif "comp" in cfg_dict["agent"]["name"].lower():
        from agents.compose import CompAgent

        agent = CompAgent(cfg=cfg_dict)

    else:
        raise NotImplementedError(
            f'agent {cfg_dict["agent"]["name"].lower()} not implemented'
        )

    return agent


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)

    if cfg_dict["wandb_log"]:
        wandb.init()
    else:
        wandb.init(mode="disabled")

    # print(wandb.config, "\n\n")
    wandb_dict = fix_wandb(wandb.config)

    # print_dict(wandb_dict)
    update_dict(cfg_dict, wandb_dict)

    cfg_dict["buffer"]["n_env"] = cfg_dict["env"]["num_envs"]
    print_dict(cfg_dict)

    torch.manual_seed(cfg_dict["seed"])
    np.random.seed(cfg_dict["seed"])

    agent = get_agent(cfg_dict)

    agent.run()
    wandb.finish()


if __name__ == "__main__":
    launch_rlg_hydra()
