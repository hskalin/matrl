# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy
import time
from distutils.util import strtobool
from env.wrapper.multiTask import multitaskenv_constructor
from common.util import AverageMeter, dump_cfg

import wandb

from env import env_map

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import datetime

import pathlib
import omegaconf

EXP_DATETIME = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(env.num_obs, 250)),
            nn.Tanh(),
            layer_init(nn.Linear(250, 250)),
            nn.Tanh(),
            layer_init(nn.Linear(250, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(env.num_obs, 250)),
            nn.Tanh(),
            layer_init(nn.Linear(250, 250)),
            nn.Tanh(),
            layer_init(nn.Linear(250, env.num_act), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, env.num_act))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


class PPO_agent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env_cfg = cfg["env"]
        self.agent_cfg = cfg["agent"]
        self.buffer_cfg = cfg["buffer"]

        self.batch_size = self.env_cfg["num_envs"] * self.agent_cfg["num_steps"]
        self.minibatch_size = self.batch_size // self.agent_cfg["num_minibatches"]

        self.device = cfg["rl_device"]

        self.env, self.feature, self.task = multitaskenv_constructor(
            env_cfg=self.env_cfg, device=self.device
        )
        assert self.feature.dim == self.task.dim, "feature and task dimension mismatch"

        # logging
        self.loggingEnabled = self.env_cfg.get("log_results", False)
        self.save_model = self.env_cfg["save_model"]
        if self.loggingEnabled:
            log_dir = (
                self.agent_cfg["name"]
                + "/"
                + self.env_cfg["env_name"]
                + "/"
                + EXP_DATETIME
                + "/"
            )
            self.log_path = self.env_cfg["log_path"] + log_dir
            pathlib.Path(self.log_path).mkdir(parents=True, exist_ok=True)
            dcfg = omegaconf.DictConfig(cfg)
            dcfg = omegaconf.OmegaConf.to_object(dcfg)
            dump_cfg(self.log_path + "cfg", dcfg)

            self.writer = SummaryWriter(self.log_path + "/tensorlogs/")

        self.agent = Agent(self.env).to(self.device)
        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.agent_cfg["learning_rate"], eps=1e-5
        )

        self.game_rewards = AverageMeter(len(self.env.log_rewards), max_size=100).to(
            "cuda:0"
        )
        self.game_lengths = AverageMeter(1, max_size=100).to("cuda:0")

        self._init_buffers()

    def _init_buffers(self):
        # ALGO Logic: Storage setup
        self.obs = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"], self.env.num_obs),
            dtype=torch.float,
        ).to(self.device)
        self.actions = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"], self.env.num_act),
            dtype=torch.float,
        ).to(self.device)
        self.logprobs = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"]), dtype=torch.float
        ).to(self.device)
        self.rewards = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"]), dtype=torch.float
        ).to(self.device)
        self.dones = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"]), dtype=torch.float
        ).to(self.device)
        self.values = torch.zeros(
            (self.agent_cfg["num_steps"], self.env_cfg["num_envs"]), dtype=torch.float
        ).to(self.device)
        self.advantages = torch.zeros_like(self.rewards, dtype=torch.float).to(
            self.device
        )

    def update(self):
        raise NotImplemented

    def calc_reward(self, s, w):
        f = self.feature.extract(s)
        r = torch.sum(w * f, 1)
        return r

    def run(self):
        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()

        # next_obs = envs.reset()
        next_obs = self.env.obs_buf

        next_done = torch.zeros(self.env_cfg["num_envs"], dtype=torch.float).to(
            self.device
        )
        num_updates = self.agent_cfg["total_timesteps"] // self.batch_size

        for update in range(1, num_updates + 1):
            wandb_metrics = {}
            # Annealing the rate if instructed to do so.
            if self.agent_cfg["anneal_lr"]:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.agent_cfg["learning_rate"]
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.agent_cfg["num_steps"]):
                global_step += 1 * self.env_cfg["num_envs"]
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_action_and_value(
                        next_obs
                    )
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.

                # next_obs, rewards[step], next_done, info = envs.step(action)

                self.env.step(action)
                next_obs, next_done, episodeLen, episodeRet = (
                    self.env.obs_buf,
                    self.env.reset_buf.clone(),
                    self.env.progress_buf.clone(),
                    self.env.return_buf.clone(),
                )
                self.rewards[step] = self.calc_reward(next_obs, self.task.Train.W)
                self.env.reset()

                # if 0 <= step <= 2:
                done_ids = next_done.nonzero(as_tuple=False).squeeze(-1)
                if done_ids.size()[0]:
                    # taking mean over all envs that are done at the
                    # current timestep
                    # episodic_return = torch.mean(episodeRet[done_ids].float()).item()
                    # episodic_length = torch.mean(episodeLen[done_ids].float()).item()

                    self.game_rewards.update(episodeRet[done_ids])
                    self.game_lengths.update(episodeLen[done_ids])

                    episodic_length = self.game_lengths.get_mean()
                    episodic_returns = self.game_rewards.get_mean()

                    # print(
                    #     f"global_step={global_step}, episodic_return={episodic_return}"
                    # )

                    if self.loggingEnabled:
                        wandb_metrics.update({"episode_lengths/step": episodic_length})
                        for i, name in enumerate(self.env.log_rewards):
                            wandb_metrics.update(
                                {f"episodic_{name}": episodic_returns[i]}
                            )
                            self.writer.add_scalar(
                                f"step/{name}", episodic_returns[i], global_step
                            )

                        self.writer.add_scalar(
                            "time/return",
                            episodic_returns[0],
                            time.time() - start_time,
                        )
                        self.writer.add_scalar(
                            "step/episode_lengths", episodic_length, global_step
                        )

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.agent_cfg["num_steps"])):
                    if t == self.agent_cfg["num_steps"] - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = (
                        self.rewards[t]
                        + self.agent_cfg["gamma"] * nextvalues * nextnonterminal
                        - self.values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.agent_cfg["gamma"]
                        * self.agent_cfg["gae_lambda"]
                        * nextnonterminal
                        * lastgaelam
                    )
                returns = advantages + self.values

            # flatten the batch
            b_obs = self.obs.reshape((-1, self.env.num_obs))
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1, self.env.num_act))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimizing the policy and value network
            clipfracs = []
            for epoch in range(self.agent_cfg["update_epochs"]):
                b_inds = torch.randperm(self.batch_size, device=self.device)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > self.agent_cfg["clip_coef"])
                            .float()
                            .mean()
                            .item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if self.agent_cfg["norm_adv"]:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                            mb_advantages.std() + 1e-8
                        )

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio,
                        1 - self.agent_cfg["clip_coef"],
                        1 + self.agent_cfg["clip_coef"],
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.agent_cfg["clip_vloss"]:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.agent_cfg["clip_coef"],
                            self.agent_cfg["clip_coef"],
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = (
                        pg_loss
                        - self.agent_cfg["ent_coef"] * entropy_loss
                        + v_loss * self.agent_cfg["vf_coef"]
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.agent.parameters(), self.agent_cfg["max_grad_norm"]
                    )
                    self.optimizer.step()

                if self.agent_cfg["target_kl"] is not None:
                    if approx_kl > self.agent_cfg["target_kl"]:
                        break

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if self.loggingEnabled:
                self.writer.add_scalar(
                    "charts/learning_rate",
                    self.optimizer.param_groups[0]["lr"],
                    global_step,
                )
                self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                self.writer.add_scalar(
                    "losses/policy_loss", pg_loss.item(), global_step
                )
                self.writer.add_scalar(
                    "losses/entropy", entropy_loss.item(), global_step
                )
                self.writer.add_scalar(
                    "losses/old_approx_kl", old_approx_kl.item(), global_step
                )
                self.writer.add_scalar(
                    "losses/approx_kl", approx_kl.item(), global_step
                )
                self.writer.add_scalar(
                    "losses/clipfrac", np.mean(clipfracs), global_step
                )
                # print("SPS:", int(global_step / (time.time() - start_time)))
                self.writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                wandb_metrics.update(
                    {
                        "charts/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "losses/value_loss": v_loss.item(),
                        "losses/entropy": entropy_loss.item(),
                        "losses/old_approx_kl": old_approx_kl.item(),
                        "losses/approx_kl": approx_kl.item(),
                        "losses/clipfrac": np.mean(clipfracs),
                    }
                )

                wandb.log(wandb_metrics)

            print(
                f"Update: {update}\tSPS: {int(global_step / (time.time() - start_time))}\tReturn: {self.game_rewards.get_mean()[0]}\tLenght: {self.game_lengths.get_mean()}"
            )
            if self.save_model and update % 100 == 0:
                print(f"Saving model at step : {update}")
                self.save_torch_model(update // 100)

        if self.save_model:
            print(f"Saving final model")
            self.save_torch_model(num_updates // 100)

        # envs.close()
        if self.loggingEnabled:
            self.writer.close()

    def test(self):
        raise NotImplemented

    def save_torch_model(self, update_step):
        from pathlib import Path

        path = self.log_path + f"/model{update_step}.pt"
        torch.save(self.agent.state_dict(), path)
        # self.agent.save(path)

    def load_torch_model(self, path):
        self.agent.load_state_dict(torch.load(path))
        # self.policy.load(path + "policy")
        # self.critic.load(path + "critic")
        # hard_update(self.critic_target, self.critic)
        # grad_false(self.critic_target)
