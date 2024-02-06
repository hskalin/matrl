from abc import ABC, abstractmethod

import torch
from common.torch_jit_utils import *


class FeatureAbstract(ABC):
    @abstractmethod
    def extract(self, s):
        """extract state and return hand-crafted features"""
        pass

    def compute_gaussDist(self, mu, sigma, scale):
        mu = torch.norm(mu, dim=1, keepdim=True) ** 2
        return torch.exp(scale * -mu / sigma**2)


class PointMassFeature(FeatureAbstract):
    """features
    pos_norm: position norm
    vel_err: velocity error
    vel_norm: velocity norm
    prox: proximity to the goal
    """

    def __init__(
        self,
        env_cfg,
        device,
    ) -> None:
        super().__init__()

        self.env_cfg = env_cfg
        self.feature_cfg = self.env_cfg["feature"]
        self.device = device

        self.use_feature = self.feature_cfg["use_feature"]
        self.verbose = self.feature_cfg.get("verbose", False)

        self.envdim = int(self.env_cfg["feature"]["dim"])
        self.Kp = torch.tensor(
            self.env_cfg["goal_lim"], dtype=torch.float64, device=self.device
        )
        self.Kv = torch.tensor(
            self.env_cfg["vel_lim"], dtype=torch.float64, device=self.device
        )
        self.ProxThresh = torch.tensor(
            self.env_cfg["task"]["proximity_threshold"],
            dtype=torch.float64,
            device=self.device,
        )
        self.proxRange = 1 / self.compute_gaussDist(
            mu=self.ProxThresh**2, sigma=self.Kp, scale=-12.8
        )

        (
            self.use_pos_norm,
            self.use_vel_err,
            self.use_vel_norm,
            self.use_prox,
        ) = self.use_feature

        self.feature_dim = [
            self.use_pos_norm,
            self.envdim * self.use_vel_err,
            self.use_vel_norm,
            self.use_prox,
        ]
        self.dim = int(sum(self.feature_dim))

        self.slice_pos = slice(0, self.envdim)
        self.slice_vel = slice(self.envdim, 2 * self.envdim)
        self.slice_velAbsNorm = slice(2 * self.envdim, 2 * self.envdim + 1)

    def extract(self, s):
        features = []

        pos = s[:, self.slice_pos]
        vel = s[:, self.slice_vel]
        velAbsNorm = s[:, self.slice_velAbsNorm]

        if self.use_pos_norm:
            posSquaredNorm = self.compute_posSquareNorm(pos)
            featurePosNorm = self.compute_featurePosNorm(posSquaredNorm)
            features.append(featurePosNorm)

        if self.use_vel_err:
            featureVel = self.compute_featureVel(vel)
            features.append(featureVel)

        if self.use_vel_norm:
            featureVelNorm = self.compute_featureVelNorm(velAbsNorm)
            features.append(featureVelNorm)

        if self.use_prox:
            posSquaredNorm = self.compute_posSquareNorm(pos)
            proxFeature = self.compute_featureProx(posSquaredNorm)
            features.append(proxFeature)

        return torch.cat(features, 1)

    def compute_posSquareNorm(self, pos):
        return torch.norm(pos, dim=1, keepdim=True) ** 2

    def compute_featurePosNorm(self, posSquaredNorm, scale=[-7.2, -360]):
        featurePosNorm = 0.5 * (
            self.compute_gaussDist(posSquaredNorm, self.Kp, scale[0])
            + self.compute_gaussDist(posSquaredNorm, self.Kp, scale[1])
        )
        return featurePosNorm

    def compute_featureVel(self, vel, scale=-16):
        return self.compute_gaussDist(vel**2, self.Kv, scale)

    def compute_featureVelNorm(self, velAbsNorm, scale=-16):
        return self.compute_gaussDist(velAbsNorm**2, self.Kv, scale)

    def compute_featureProx(self, posSquaredNorm):
        return torch.where(posSquaredNorm > self.ProxThresh**2, self.proxRange, 1)

    def compute_gaussDist(self, mu, sigma, scale):
        return torch.exp(scale * mu / sigma**2)


class PointerFeature(FeatureAbstract):
    def __init__(
        self,
        env_cfg,
        device,
    ) -> None:
        super().__init__()

        self.env_cfg = env_cfg
        self.feature_cfg = self.env_cfg["feature"]
        self.device = device

        self.use_feature = self.feature_cfg["use_feature"]
        self.verbose = self.feature_cfg.get("verbose", False)

        self.Kp = torch.tensor(
            self.env_cfg["goal_lim"], dtype=torch.float64, device=device
        )
        self.Kv = torch.tensor(
            self.env_cfg["vel_lim"], dtype=torch.float64, device=device
        )
        self.ProxThresh = torch.tensor(
            self.env_cfg["task"]["proximity_threshold"],
            dtype=torch.float64,
            device=device,
        )
        self.Ka = torch.pi

        (
            self.use_posX,
            self.use_posY,
            self.use_vel_norm,
            self.use_ang_norm,
            self.use_angvelNorm,
        ) = self.use_feature

        self.feature_dim = [
            self.use_posX,
            self.use_posY,
            self.use_vel_norm,
            self.use_ang_norm,
            self.use_angvelNorm,
        ]
        self.dim = sum(self.feature_dim)

        self.proxScale = 1 / self.compute_gaussDist(
            mu=self.ProxThresh[None, None], sigma=self.Kp, scale=25
        )

        self.slice_yaw = slice(0, 1)
        self.slice_posX = slice(3, 4)
        self.slice_posY = slice(4, 5)
        self.slice_vel = slice(5, 7)
        self.slice_angvel = slice(7, 8)

    def extract(self, s):
        features = []

        errorYaw = s[:, self.slice_yaw]
        errorPosX = s[:, self.slice_posX]
        errorPosY = s[:, self.slice_posY]
        errorVel = s[:, self.slice_vel]
        errorAngVel = s[:, self.slice_angvel]

        if self.use_posX:
            featureProxX = self.compute_featureProx(errorPosX)
            features.append(featureProxX)

        if self.use_posY:
            featureProxY = self.compute_featureProx(errorPosY)
            features.append(featureProxY)

        if self.use_vel_norm:
            featureVelNorm = self.compute_featureVelNorm(errorVel)
            features.append(featureVelNorm)

        if self.use_ang_norm:
            featureAngNorm = self.compute_featureAngNorm(errorYaw)
            features.append(featureAngNorm)

        if self.use_angvelNorm:
            featureAngVelNorm = self.compute_featureAngVelNorm(errorAngVel)
            features.append(featureAngVelNorm)

        return torch.concatenate(features, 1)

    def compute_featureProx(self, errorPos, scale=25):
        d = torch.norm(errorPos, dim=1, keepdim=True) ** 2
        prox = self.proxScale * torch.exp(scale * -d / self.Kp**2)
        return torch.where(d > self.ProxThresh**2, prox, 1)

    def compute_featureVelNorm(self, errorVel, scale=30):
        return self.compute_gaussDist(errorVel, self.Kv, scale)

    def compute_featureAngNorm(self, errorYaw, scale=50):
        return self.compute_gaussDist(errorYaw, self.Ka, scale)

    def compute_featureAngVelNorm(self, errorAngVel, scale=50):
        return self.compute_gaussDist(errorAngVel, self.Kv, scale)

    def compute_gaussDist(self, mu, sigma, scale):
        mu = torch.norm(mu, dim=1, keepdim=True) ** 2
        return torch.exp(scale * -mu / sigma**2)


class BlimpFeature(FeatureAbstract):
    def __init__(
        self,
        env_cfg,
        device,
    ) -> None:
        super().__init__()

        self.env_cfg = env_cfg
        self.feature_cfg = self.env_cfg["feature"]
        self.device = device

        self.verbose = self.feature_cfg.get("verbose", False)

        self.Kp = torch.tensor(
            self.env_cfg["goal"]["pos_lim"], dtype=torch.float64, device=device
        )
        self.Kv = torch.tensor(
            self.env_cfg["goal"]["vel_lim"], dtype=torch.float64, device=device
        )
        self.ProxThresh = torch.tensor(
            self.env_cfg["task"]["proximity_threshold"],
            dtype=torch.float64,
            device=device,
        )
        self.trigger_dist = torch.tensor(
            self.env_cfg["goal"].get("trigger_dist", 2)
        ).to(self.device)
        self.Ka = torch.pi

        self.dim = 12
        if self.verbose:
            print("[Feature] dim", self.dim)

        self.proxScale = 1 / self.compute_gaussDist(
            mu=self.ProxThresh[None, None], sigma=self.Kp, scale=25
        )

        # robot angle
        self.slice_rb_angle = slice(0, 0 + 3)

        # goal angle
        self.slice_goal_angle = slice(3, 3 + 3)

        # robot z
        self.slice_rb_z = slice(6, 6 + 1)

        # goal trigger
        self.slice_goal_trigger = slice(7, 7 + 1)

        # relative position to nav goal
        self.slice_err_posNav = slice(8, 8 + 3)

        # relative position to hov goal
        self.slice_err_posHov = slice(11, 11 + 3)

        # robot vel
        self.slice_rb_v = slice(14, 14 + 3)

        # goal vel
        self.slice_goal_v = slice(17, 17 + 3)

        # robot ang vel
        self.slice_rb_angvel = slice(20, 20 + 3)

        # goal ang vel
        self.slice_goal_angvel = slice(23, 23 + 3)

        # robot thrust
        self.slice_actuators = slice(26, 26 + 1)

        # robot actions
        self.slice_prev_act = slice(28, 28 + 3)

    def extract(self, s):
        features = []

        # raw obs
        robot_angle = s[:, self.slice_rb_angle]
        # robot_angVel = s[:, self.slice_rb_angvel]
        robot_v = s[:, self.slice_rb_v]
        robot_thrust = s[:, self.slice_actuators]
        # robot_prev_act = s[:, self.slice_prev_act]
        robot_headingV, _, _ = globalToLocalRot(
            robot_angle[:, 0],
            robot_angle[:, 1],
            robot_angle[:, 2],
            robot_v[:, 0],
            robot_v[:, 1],
            robot_v[:, 2],
        )

        goal_ang = s[:, self.slice_goal_angle]
        goal_v = s[:, self.slice_goal_v]
        # goal_angVel = s[:, self.slice_goal_angvel]
        # goal_trigger = s[:, self.slice_goal_trigger]

        # prepare feature
        error_ang = check_angle(robot_angle - goal_ang)  # rel angle
        error_posNav = s[:, self.slice_err_posNav]
        # error_posHov = s[:, self.slice_err_posHov]
        error_v = s[:, self.slice_rb_v] - s[:, self.slice_goal_v]
        # error_angVel = robot_angVel - goal_angVel

        error_navHeading = check_angle(
            compute_heading(yaw=robot_angle[:, 2:3], rel_pos=error_posNav)
        )
        error_vnorm = robot_headingV - torch.norm(goal_v[:, 0:2], dim=1, keepdim=True)

        # Nav planar:
        x = self.compute_featurePosNorm(error_posNav[:, 0:2])
        features.append(x)

        # Nav z:
        x = self.compute_featurePosNorm(error_posNav[:, 2:3])
        features.append(x)

        # Nav trigger:
        x = 10000.0 * s[:, self.slice_goal_trigger]
        features.append(x)

        # Nav yaw_to_goal:
        x = self.compute_featureAngNorm(error_navHeading)
        features.append(x)

        # Hov proxDist:
        x = self.compute_featureProx(s[:, self.slice_err_posHov])
        features.append(x)

        # Hov yaw:
        x = self.compute_featureAngNorm(error_ang[:, 2:3])
        features.append(x)

        # Nav/Hov vnorm:
        x = self.compute_featureVelNorm(error_vnorm)
        features.append(x)

        # vx:
        x = self.compute_featureVelNorm(error_v[:, 0:1])
        features.append(x)

        # vy:
        x = self.compute_featureVelNorm(error_v[:, 1:2])
        features.append(x)

        # vz:
        x = self.compute_featureVelNorm(error_v[:, 2:3])
        features.append(x)

        # regulate row and pitch:
        x = self.compute_featureAngNorm(robot_angle[:, 0:2])
        features.append(x)

        # regulate robot thrust: rescale to [0, 2], similar to angle scale
        x = self.compute_featureAngNorm(robot_thrust + 1)
        features.append(x)

        f = torch.concatenate(features, 1)
        if self.verbose:
            print(
                "[Feature] features [planar, Z, trigger, yaw2goal, proximity, yaw, vnorm, vx, vy, vz,  regRP, regThrust]"
            )
            print(f)
        return f

    def compute_featurePosNorm(self, x, scale=25):
        return self.compute_gaussDist(x, self.Kp, scale)

    def compute_featureProx(self, x, scale=25):
        d = torch.norm(x, dim=1, keepdim=True) ** 2
        prox = self.proxScale * torch.exp(scale * -d / self.Kp**2)
        return torch.where(d > self.ProxThresh, prox, 1)

    def compute_featureVelNorm(self, x, scale=30):
        return self.compute_gaussDist(x, self.Kv, scale)

    def compute_featureAngNorm(self, x, scale=50):
        return self.compute_gaussDist(x, self.Ka, scale)

    def compute_featureAngVelNorm(self, x, scale=50):
        return self.compute_gaussDist(x, self.Kv, scale)


class AntFeature(FeatureAbstract):
    """
    features : tbd
    """

    def __init__(self, env_cfg, device) -> None:
        super().__init__()

        self.env_cfg = env_cfg
        self.feature_cfg = self.env_cfg["feature"]
        self.device = device

        self.verbose = self.feature_cfg.get("verbose", False)

        self.dim = 5
        if self.verbose:
            print("[Feature] dim", self.dim)

        # constants
        self.direction_threshold = 0.93
        self.dof_pos_threshold = 0.99
        self.alive_reward_scale = 0.5
        self.jump_threshold = 0.8
        self.up_weight = self.env_cfg["upWeight"]
        self.actions_cost_scale = self.env_cfg["actionsCost"]
        self.energy_cost_scale = self.env_cfg["energyCost"]
        self.joints_at_limit_cost_scale = self.env_cfg["jointsAtLimitCost"]
        self.death_cost = self.env_cfg["deathCost"]
        self.termination_height = self.env_cfg["terminationHeight"]

        # indices
        self.slice_pos_actions = slice(52, 52 + 8)
        self.slice_pos_dof_velocity = slice(20, 20 + 8)
        self.slice_pos_dof_pos_scaled = slice(12, 12 + 8)
        self.slice_pos_deviation = slice(60, 60 + 2)
        self.pos_up_proj = 10
        self.pos_z = 0

    def extract(self, s):
        features = compute_ant_jump_reward(
            s,
            self.up_weight,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.termination_height,
            self.death_cost,
            self.dim
        )
        
        return features

@torch.jit.script
def compute_ant_jump_reward(
    obs_buf,
    up_weight,
    actions_cost_scale,
    energy_cost_scale,
    joints_at_limit_cost_scale,
    termination_height,
    death_cost,
    scale,
):
    # type: (Tensor, float, float, float, float, float, float, float) -> Tensor

    # aligning up axis of ant and environment
    up_reward = torch.zeros_like(obs_buf[:, 11])
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(obs_buf[:,52:60]**2, dim=-1)
    electricity_cost = torch.sum(torch.abs(obs_buf[:,52:60] * obs_buf[:, 20:28]), dim=-1)
    dof_at_limit_cost = torch.sum(obs_buf[:, 12:20] > 0.99, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(up_reward) * 0.5
    progress_reward = torch.where(
        obs_buf[:, 0] > 0.8,
        torch.ones_like(up_reward) * 2,
        -torch.zeros_like(up_reward) * 0.5,
    )

    jump_deviation = (
        torch.norm(obs_buf[:,60:62], p=2, dim=-1) * 0.5
    )

    features = torch.cat(
        (
            progress_reward.unsqueeze(-1),
            alive_reward.unsqueeze(-1),
            up_reward.unsqueeze(-1),
            -jump_deviation.unsqueeze(-1),
            (
                - actions_cost_scale * actions_cost
                - energy_cost_scale * electricity_cost
                - dof_at_limit_cost * joints_at_limit_cost_scale
            ).unsqueeze(-1)
        ),
        1,
    )

    condition = obs_buf[:, 0] < termination_height
    # Create a mask for the rows that satisfy the condition
    mask = condition.unsqueeze(1).expand_as(features)
    # Replace the values for the selected rows
    features[mask] = death_cost / scale

    return features



def feature_constructor(env_cfg, device):
    if "pointer" in env_cfg["env_name"].lower():
        return PointerFeature(env_cfg, device)
    elif "pointmass" in env_cfg["env_name"].lower():
        return PointMassFeature(env_cfg, device)
    elif "ant" in env_cfg["env_name"].lower():
        return AntFeature(env_cfg, device)
    elif "blimp" in env_cfg["env_name"].lower():
        return BlimpFeature(env_cfg, device)
    else:
        print(f'feature not implemented: {env_cfg["env_name"]}')
        return None
