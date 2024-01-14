#!/usr/bin/env python
import os
import copy
import math
from tkinter import *
import json
import hydra
# import isaacgym
import numpy as np
import rospy
import torch
import wandb
from common.torch_jit_utils import *
from common.util import (
    AverageMeter,
    fix_wandb,
    omegaconf_to_dict,
    print_dict,
    update_dict,
)
from common.pid import BlimpHoverControl, BlimpVelocityControl
from common.goal import FixWayPoints
from librepilot.msg import LibrepilotActuators
from omegaconf import DictConfig, OmegaConf
from run_gaz import get_agent
from sensor_msgs.msg import Imu
from std_srvs.srv import Empty
from uav_msgs.msg import uav_pose
from geometry_msgs.msg import Pose, TwistStamped, Point
from visualization_msgs.msg import Marker, MarkerArray

real_exp=True
testing=True # off-line testing

model_folder = os.getcwd()+"/logs/rmacompblimp/BlimpRand/2024-01-10-13-34-18"
model_checkpoint = "model60"

name_space="machine_1"
goal_style="circle" # square, hourglass, circle
max_thrust=0.5
trigger_dist=5
with_backup_ctrl=True
reset_dist=30 # [m] activate backup ctrl

flightmode=3

dbg_ros=False
dbg_obs=False

if real_exp:
    if testing:
        master_ip = "129.69.124.164"
        slave_ip = "192.168.7.217"
        os.environ["ROS_IP"] = slave_ip
        os.environ["ROS_MASTER_URI"] = "http://" + master_ip + ":11311/"
        os.environ["GAZEBO_MASTER_URI"] = "http://" + master_ip + ":11351/"
    else:
        pass
else:
    master_ip = "192.168.7.56"
    os.environ["ROS_IP"] = master_ip
    os.environ["ROS_MASTER_URI"] = "http://" + master_ip + ":11311/"
    os.environ["GAZEBO_MASTER_URI"] = "http://" + master_ip + ":11351/"

GRAVITY = 9.81

def obj2tensor(
    rosobj,
    attr_list=["w", "x", "y", "z"],
):
    val_list = []
    for attr in attr_list:
        try:
            val_list.append(getattr(rosobj, attr))
        except:
            pass

    return torch.tensor(val_list, device="cuda", dtype=torch.float).unsqueeze(0)


def lmap(v, x, y) -> float:
    """Linear map of value v with range x to desired range y."""
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0])


class PlayUI:
    def __init__(
        self, cfg_dict, model_path, device="cuda", 
    ) -> None:
        if not real_exp:
            self.root = Tk()
            self.root.title("test")
            self.root.geometry("300x500")

        # init exp params
        self.real_exp = real_exp
        self.dbg_ros = dbg_ros
        self.device = device
        self.dt = 0.1
        self.max_thrust = max_thrust

        # init and load agent
        self.agent = get_agent(cfg_dict)
        self.agent.load_torch_model(model_path)

        self.weights = self.agent.task.Eval.W.clone()
        self.weightLabels = cfg_dict["env"]["task"]["taskLabels"]

        if not real_exp:
            self.rew = None
            self.generate_scales()
            self.print_step_reward()

        # init waypints
        self.wp = FixWayPoints(device=self.device, num_envs=1, trigger_dist=trigger_dist)

        # init buffer
        self.obs_buf = torch.zeros(1, 34, device=self.device)

        self.rb_pos = torch.zeros(1, 3, device=self.device)
        self.rb_lvels = torch.zeros(1, 3, device=self.device)
        self.rb_lacc = torch.zeros(1, 3, device=self.device)
        self.ori_data = torch.zeros(1, 4, device=self.device)
        self.rb_rot = torch.zeros(1, 3, device=self.device)
        self.rb_avels = torch.zeros(1, 3, device=self.device)

        self.prev_actions = torch.zeros((1, 4), device=self.device)
        self.prev_actuator = torch.zeros((1, 3), device=self.device)
        self.ema_smooth = torch.tensor([[2 * self.dt, 3 * self.dt]], device=self.device)

        # init ros node
        self._pub_and_sub = False

        rospy.init_node("rl_node")
        self.rate = rospy.Rate(1 / self.dt)

        if real_exp:
            self.action_publisher = rospy.Publisher(
                name_space + "/actuatorcommand", LibrepilotActuators, queue_size=1
            )
            self.flightmode_publisher = rospy.Publisher(
                name_space + "/command", uav_pose, queue_size=1
            )
            rospy.Subscriber(name_space+"/imu", Imu, self._imu_callback)
            rospy.Subscriber(name_space+"/pose", uav_pose, self._pose_callback)

        else:
            self.action_publisher = rospy.Publisher(
                name_space + "/GCSACTUATORS", LibrepilotActuators, queue_size=1
            )
            rospy.Subscriber(name_space+"/tail/imu", Imu, self._imu_callback)
            rospy.Subscriber(name_space+"/ground_speed", TwistStamped, self._vel_callback)
            rospy.Subscriber(name_space+"/tail/pose", Pose, self._pose_callback)

        # init monitor
        self.wp_hov_publisher = rospy.Publisher(
            name_space + "/rviz_wp_hov", Marker, queue_size=1
        )
        self.wp_nav_publisher = rospy.Publisher(
            name_space + "/rviz_wp_nav", Marker, queue_size=1
        )
        self.wplist_publisher = rospy.Publisher(
            name_space + "/rviz_wp_list", MarkerArray, queue_size=10
        )
        self._pub_and_sub = True

        if not real_exp:
            self.reset_world_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
            self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
            self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)

        self.ros_cnt = 0

    def create_rviz_marker(
        self, position, scale=(2, 2, 2), color=(1, 1, 1, 0)
    ):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.action = marker.ADD
        marker.type = marker.SPHERE
        marker.id = 0
        marker.scale.x, marker.scale.y, marker.scale.z = scale
        marker.color.a, marker.color.r, marker.color.g, marker.color.b = color
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
            position[0],
            position[1],
            position[2],
        )  # rviz(ENU)
        marker.pose.orientation.w = 1
        return marker
    
    def publish_wp_toRviz(
        self, position, publisher, scale = (4, 4, 4), color = (1, 0, 0, 1)
    ):
        marker = self.create_rviz_marker(position, scale=scale, color=color)
        publisher.publish(marker)

    def publish_wplist_toRviz(self, wp_list):
        markerArray = MarkerArray()

        for wp in wp_list:
            marker = self.create_rviz_marker(wp)
            markerArray.markers.append(marker)

            id = 0
            for m in markerArray.markers:
                m.id = id
                id += 1

            self.wplist_publisher.publish(markerArray)

    def NED_to_ENU(self, data):
        data_copy = copy.copy(data)
        data[:, 0] = data_copy[:, 1]
        data[:, 1] = data_copy[:, 0]
        data[:, 2] = -data_copy[:, 2]
        return data

    def _imu_callback(self, msg):
        """imu msg callback

        Args:
            msg ([Imu]): imu sensor raw data
        """
        acc = obj2tensor(msg.linear_acceleration)
        if self.real_exp:
            acc[:, 2] += GRAVITY
        else:
            acc[:, 2] -= GRAVITY

        self.rb_lacc = acc

        if self.dbg_ros:
            self.ros_cnt += 1
            if self.ros_cnt % 100 == 0:
                print(
                    "[ KinematicObservation ] imu_callback: linear_acceleration",
                    self.rb_lacc,
                )

    def _vel_callback(self, msg):
        """imu msg callback

        Args:
            msg ([Imu]): imu sensor raw data
        """
        self.rb_lvels = obj2tensor(msg.twist.linear)
          
        if self.real_exp: 
            self.rb_lvels = self.NED_to_ENU(self.rb_lvels)

        if self.dbg_ros:
            self.ros_cnt += 1
            if self.ros_cnt % 100 == 0:
                print(
                    "[ KinematicObservation ] vel_callback: velocity",
                    self.rb_lvels,
                )


    def _pose_callback(self, msg):
        """pose msg callback

        Args:
            msg ([uav_pose]): gcs processed sensor data
        """
        self.rb_pos = obj2tensor(msg.position)
        self.ori_data = obj2tensor(msg.orientation)
        self.rb_rot = torch.concat(get_euler_wxyz(self.ori_data)).unsqueeze(0)

        if self.real_exp:  
            self.rb_pos = self.NED_to_ENU(self.rb_pos)

            self.rb_lvels = obj2tensor(msg.velocity)
            self.rb_lvels = self.NED_to_ENU(self.rb_lvels)

            self.rb_avels = obj2tensor(msg.angVelocity)

        if self.dbg_ros:
            print(
                "[ KinematicObservation ] pose_callback: position",
                self.rb_pos,
            )
            print(
                "[ KinematicObservation ] pose_callback: orientation",
                self.ori_data,
            )
            print(
                "[ KinematicObservation ] pose_callback: angle",
                self.rb_rot,
            )
            if self.real_exp:
                print(
                    "[ KinematicObservation ] pose_callback: ang_vel",
                    self.rb_avels,
                )
                print(
                    "[ KinematicObservation ] pose_callback: velocity",
                    self.rb_lvels,
                )


    def reset_world(self):
        """reset gazebo world"""
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_world_proxy()
        except rospy.ServiceException as err:
            print("/gazebo/reset_world service call failed", err)

    def pause_sim(self):
        """pause simulation with ros service call"""
        rospy.logdebug("PAUSING START")
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as err:
            print("/gazebo/pause_physics service call failed", err)

        rospy.logdebug("PAUSING FINISH")

    def unpause_sim(self):
        """unpause simulation with ros service call"""
        rospy.logdebug("UNPAUSING START")
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as err:
            print("/gazebo/unpause_physics service call failed", err)

        rospy.logdebug("UNPAUSING FiNISH")

    def play(self):
        def shutdownhook():
            print("shutdown time!")

        ctrler = BlimpHoverControl(device=self.device)
        avgStepRew = AverageMeter(1, 20).to(self.agent.device)
        while not rospy.is_shutdown():
            s = self.reset()

            for _ in range(5000):
                self.publish_wp_to_rviz()

                if not real_exp:
                    self.root.update_idletasks()
                    self.root.update()

                dist_to_hov_goal = torch.norm(s[0, 11:14])
                if with_backup_ctrl and dist_to_hov_goal > reset_dist:
                    print(f"backup control activated with distance to center {dist_to_hov_goal}")
                    a = ctrler.act(s)
                else:
                    a = self.agent.act(s, self.agent.task.Eval, "exploit")


                print("task:", self.agent.task.Eval.W[0])
                print("action:", a)

                # a: [thrust, yaw, stick, pitch]
                a = self.prepare_action(a)

                # actuator:
                # [m2, lfin, rfin, tfin, bfin, stick, m1, unused, m0, unused, unused, unused]
                actuator = self.map_actuator(a)

                act_msg = LibrepilotActuators()
                act_msg.header.stamp = rospy.Time.now()
                act_msg.data.data = actuator
                self.action_publisher.publish(act_msg)

                mode = uav_pose()
                mode.flightmode = flightmode
                self.flightmode_publisher.publish(mode)

                self.rate.sleep()

                s_next = self.get_obs()

                r = self.agent.calc_reward(s_next, self.agent.task.Eval.W)
                s = s_next
                avgStepRew.update(r)
                if not real_exp:
                    if self.rew:
                        self.rew.set(avgStepRew.get_mean())

                rospy.on_shutdown(shutdownhook)

    def publish_wp_to_rviz(self):
        pos_hov = self.wp.pos_hov[0].squeeze().detach().cpu().numpy()
        pos_nav = self.wp.get_pos_nav()[0].squeeze().detach().cpu().numpy()
        wplist = self.wp.pos_nav[0].tolist()

        self.publish_wp_toRviz(pos_hov, self.wp_hov_publisher, color=(1, 0, 1, 0))
        self.publish_wp_toRviz(pos_nav, self.wp_nav_publisher)
        self.publish_wplist_toRviz(wplist)

    def map_actuator(self, a):
        actuator = 1500 * np.ones([12, 1])
        actuator[[6, 8]] = a[0]
        actuator[[0, 3, 4]] = a[1]
        actuator[5] = a[2]
        actuator[[1, 2]] = a[3]
        return actuator

    def prepare_action(self, a):
        a = torch.clip(a, -1, 1)
        self.prev_actions = copy.deepcopy(a)

        a[:, 0] = a[:, 0] * self.ema_smooth[:, 0] + self.prev_actuator[:, 0] * (
            1 - self.ema_smooth[:, 0]
        )
        a[:, 2] = a[:, 2] * self.ema_smooth[:, 1] + self.prev_actuator[:, 1] * (
            1 - self.ema_smooth[:, 1]
        )
        bot_thrust = a[:, 1] * self.ema_smooth[:, 0] + self.prev_actuator[:, 2] * (
            1 - self.ema_smooth[:, 0]
        )

        self.prev_actuator[:, 0] = a[:, 0]
        self.prev_actuator[:, 1] = a[:, 2]
        self.prev_actuator[:, 2] = bot_thrust

        a[:, 0] = torch.clip((a[:, 0] + 1) / 2, 0, self.max_thrust)
        a[:, 1] = -a[:, 1]
        a[:, 3] = -a[:, 3]

        a = lmap(a, [-1, 1], [1000, 2000])
        a = a.squeeze().detach().cpu().numpy()
        return a

    def get_obs(self):
        env_ids = 0
        roll, pitch, yaw = self.rb_rot[env_ids]

        # robot angle
        self.obs_buf[env_ids, 0] = check_angle(roll)
        self.obs_buf[env_ids, 1] = check_angle(pitch)
        self.obs_buf[env_ids, 2] = check_angle(yaw)

        if dbg_obs:
            print("roll", check_angle(roll))
            print("pitch", check_angle(pitch))
            print("yaw", check_angle(yaw))

        # goal angles
        self.obs_buf[env_ids, 3] = check_angle(self.wp.ang[env_ids, 0])
        self.obs_buf[env_ids, 4] = check_angle(self.wp.ang[env_ids, 1])
        self.obs_buf[env_ids, 5] = check_angle(self.wp.ang[env_ids, 2])

        # robot z
        self.obs_buf[env_ids, 6] = self.rb_pos[env_ids, 2]

        if dbg_obs:
            print("z", self.rb_pos[env_ids, 2])

        # trigger navigation goal
        trigger = self.wp.update_state(self.rb_pos)
        self.obs_buf[env_ids, 7] = trigger[env_ids, 0]

        if dbg_obs:
            print("wp trigger", trigger[env_ids, 0])
            print("wp idx", self.wp.idx)

        # relative pos to navigation goal
        rel_pos = self.rb_pos - self.wp.get_pos_nav()
        self.obs_buf[env_ids, 8] = rel_pos[env_ids, 0]
        self.obs_buf[env_ids, 9] = rel_pos[env_ids, 1]
        self.obs_buf[env_ids, 10] = rel_pos[env_ids, 2]

        if dbg_obs:
            print("rel_pos nav", rel_pos)   

        # relative pos to hover goal
        rel_pos = self.rb_pos - self.wp.pos_hov
        self.obs_buf[env_ids, 11] = rel_pos[env_ids, 0]
        self.obs_buf[env_ids, 12] = rel_pos[env_ids, 1]
        self.obs_buf[env_ids, 13] = rel_pos[env_ids, 2]

        if dbg_obs:
            print("rel_pos hov", rel_pos)

        # robot vel
        self.obs_buf[env_ids, 14] = self.rb_lvels[env_ids, 0]
        self.obs_buf[env_ids, 15] = self.rb_lvels[env_ids, 1]
        self.obs_buf[env_ids, 16] = self.rb_lvels[env_ids, 2]

        if dbg_obs:
            print("rb_lvels", self.rb_lvels)

        # goal vel
        self.obs_buf[env_ids, 17] = self.wp.vel[env_ids, 0]
        self.obs_buf[env_ids, 18] = self.wp.vel[env_ids, 1]
        self.obs_buf[env_ids, 19] = self.wp.vel[env_ids, 2]
        self.obs_buf[env_ids, 20] = self.wp.velnorm[env_ids, 0]

        if dbg_obs:
            print("wp.vel", self.wp.vel)
            print("wp.velnorm", self.wp.velnorm)

        # robot angular velocities
        self.obs_buf[env_ids, 21] = self.rb_avels[env_ids, 0]
        self.obs_buf[env_ids, 22] = self.rb_avels[env_ids, 1]
        self.obs_buf[env_ids, 23] = self.rb_avels[env_ids, 2]

        # goal ang vel
        self.obs_buf[env_ids, 24] = self.wp.angvel[env_ids, 0]
        self.obs_buf[env_ids, 25] = self.wp.angvel[env_ids, 1]
        self.obs_buf[env_ids, 26] = self.wp.angvel[env_ids, 2]

        # print("wp.angvel", self.wp.angvel)

        # prev actuators
        self.obs_buf[env_ids, 27] = self.prev_actuator[env_ids, 0]  # thrust
        self.obs_buf[env_ids, 28] = self.prev_actuator[env_ids, 1]  # stick
        self.obs_buf[env_ids, 29] = self.prev_actuator[env_ids, 2]  # bot thrust

        if dbg_obs:
            print("prev_actuator", self.prev_actuator)

        # previous actions
        self.obs_buf[env_ids, 30] = self.prev_actions[env_ids, 0]
        self.obs_buf[env_ids, 31] = self.prev_actions[env_ids, 1]
        self.obs_buf[env_ids, 32] = self.prev_actions[env_ids, 2]
        self.obs_buf[env_ids, 33] = self.prev_actions[env_ids, 3]

        if dbg_obs:
            print("prev_actions", self.prev_actions)

        return self.obs_buf.clone()

    def reset(self):
        self.agent.comp.reset()
        self.agent.prev_traj.clear()

        env_ids = 0

        # sample new waypoint
        self.wp.sample(env_ids)
        self.wp.ang[env_ids, 0:2] = 0
        self.wp.angvel[env_ids, 0:2] = 0

        self.prev_actions[env_ids] = torch.zeros((1, 4), device=self.device)
        self.prev_actuator[env_ids] = torch.zeros((1, 3), device=self.device)

        # reset gazebo world
        if not real_exp:
            self.pause_sim()
            self.reset_world()
            self.unpause_sim()

        # refresh new observation after reset
        return self.get_obs().clone()

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
            # self.root.update_idletasks()
            # self.root.update()

            print(self.agent.task.Eval.W[0])


def modify_cfg(cfg_dict):
    # don't change these
    cfg_dict["agent"]["phase"] = 4  # phase: [encoder, adaptor, fine-tune, deploy]
    cfg_dict["agent"]["norm_task_by_sf"] = False
    cfg_dict["agent"]["load_model"] = False

    cfg_dict["env"]["num_envs"] = 1
    cfg_dict["env"]["save_model"] = False
    cfg_dict["env"]["mode"] = "play"
    cfg_dict["env"]["sim"]["headless"] = True
    cfg_dict["env"]["task"]["rand_task"] = False
    cfg_dict["env"]["task"]["rand_vel_targets"] = False

    if "aero" in cfg_dict["env"]:
        cfg_dict["env"]["aero"]["wind_mag"] = 0.0
    if "domain_rand" in cfg_dict["env"]["task"]:
        cfg_dict["env"]["task"]["domain_rand"] = False

    cfg_dict["buffer"]["n_env"] = cfg_dict["env"]["num_envs"]
    cfg_dict["buffer"]["min_n_experience"] = 0
    cfg_dict["buffer"]["capacity"] = 20

    # change these
    cfg_dict["agent"]["exploit_method"] = "sfgpi"

    cfg_dict["env"]["goal"]["trigger_dist"] = trigger_dist
    cfg_dict["env"]["goal"]["type"] = "fix"
    cfg_dict["env"]["goal"]["style"] = goal_style

    print_dict(cfg_dict)

    return cfg_dict


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    cfg_dict = omegaconf_to_dict(cfg)

    wandb.init(mode="disabled")
    wandb_dict = fix_wandb(wandb.config)

    cfg_path = model_folder + "/cfg"
    model_path = model_folder + "/" + model_checkpoint 

    cfg_dict = None
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    update_dict(cfg_dict, wandb_dict)
    cfg_dict = modify_cfg(cfg_dict)

    torch.manual_seed(456)
    np.random.seed(456)

    playob = PlayUI(cfg_dict, model_path)
    playob.play()

    wandb.finish()


if __name__ == "__main__":
    launch_rlg_hydra()
