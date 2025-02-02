# Multi Agent And Task Reinforcement Learning

****

## Work Allocations and TODO

### Nilaksh

Immediate:
- [ ] Figure out humanoid-env
- [ ] Implement PPO to try out single tasks
      
Future :
- [ ] Implement multi-task framework for feasible single tasks

### Satpaty

Immediate:
- [ ] Integrate Dreamer V3 in Time Chamber
      
Future :
- [ ] Implement MWM for proprio

### Viswesh

Immediate:
- [ ] Extend the environment to multi agent settings
- [ ] Check ways to run multiple seeds parallely for MARL
      
Future :
- [ ] Try out multi-agent frameworks
***

## Installation

Download and extract the [Isaac Gym preview release](https://developer.nvidia.com/isaac-gym). Supported Python versions are 3.7 or 3.8. Next create a `conda` or `venv` virtual environment and launch it. 

```
python3.8 -m venv rl-env
source rl-env/bin/activate
```

In the `python` subdirectory of the extracted folder, run:

```
pip install -e .
```

This will install the `isaacgym` package and all of its dependencies in the active Python environment. 
Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples`
directory, like `joint_monkey.py`. If you have any trouble running the samples, please follow troubleshooting steps
described in the [Isaac Gym Preview Release 3/4 installation instructions](https://developer.nvidia.com/isaac-gym).  
Then install this repo:

```bash
pip install -r requirements.txt
```

## Quick Start

****

### Tasks

Source code for tasks can be found in  `timechamber/tasks`,The detailed settings of state/action/reward are
in [here](./docs/environments.md).
More interesting tasks will come soon.

#### Humanoid Strike

Humanoid Strike is a 3D environment with two simulated humanoid physics characters. Each character is equipped with a sword and shield with 37 degrees-of-freedom.
The game will be restarted if one agent goes outside the arena. We measure how much the player damaged the opponent and how much the player was damaged by the opponent in the terminated step to determine the winner.

<div align=center>
<img src="assets/images/humanoid_strike.gif" align="center" width="600"/>
</div> 



#### Ant Sumo

Ant Sumo is a 3D environment with simulated physics that allows pairs of ant agents to compete against each other.
To win, the agent has to push the opponent out of the ring. Every agent has 100 hp . Each step, If the agent's body
touches the ground, its hp will be reduced by 1.The agent whose hp becomes 0 will be eliminated.
<div align=center>
<img src="assets/images/ant_sumo.gif" align="center" width="600"/>
</div> 

#### Ant Battle

Ant Battle is an expanded environment of Ant Sumo. It supports more than two agents competing against with
each other. The battle ring radius will shrink, the agent going out of the ring will be eliminated.
<div align=center>
<img src="assets/images/ant_battle.gif" align="center" width="600"/>
</div>  

### Self-Play Training

To train your policy for tasks, for example:

```bash
# run self-play training for Humanoid Strike task
python train.py task=MA_Humanoid_Strike headless=True
```

```bash
# run self-play training for Ant Sumo task
python train.py task=MA_Ant_Sumo train=MA_Ant_SumoPPO headless=True
```

```bash
# run self-play training for Ant Battle task
python train.py task=MA_Ant_Battle train=MA_Ant_BattlePPO headless=True
```

Key arguments to the training script
follow [IsaacGymEnvs Configuration and command line arguments](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/README.md#configuration-and-command-line-arguments)
.
Other training arguments follow [rl_games config parameters](https://github.com/Denys88/rl_games#config-parameters),
you can change them in `timechamber/tasks/train/*.yaml`. There are some specific arguments for self-play training:

- `num_agents`: Set the number of agents for Ant Battle environment, it should be larger than 1.
- `op_checkpoint`: Set to path to the checkpoint to load initial opponent agent policy.
  If it's empty, opponent agent will use random policy.
- `update_win_rate`: Win_rate threshold to add the current policy to opponent's player pool.
- `player_pool_length`: The max size of player pool, following FIFO rules.
- `games_to_check`: Warm up for training, the player pool won't be updated until the current policy plays such number of
  games.
- `max_update_steps`: If current policy update iterations exceed that number, the current policy will be added to
  opponent player_pool.

### Policies Evaluation

To evaluate your policies, for example:

```bash
# run testing for Ant Sumo policy
python train.py task=MA_Ant_Sumo train=MA_Ant_SumoPPO test=True num_envs=4 minibatch_size=32 headless=False checkpoint='models/ant_sumo/policy.pth'
```

```bash
# run testing for Humanoid Strike policy
python train.py task=MA_Humanoid_Strike train=MA_Humanoid_StrikeHRL test=True num_envs=4 minibatch_size=32 headless=False checkpoint='models/Humanoid_Strike/policy.pth' op_checkpoint='models/Humanoid_Strike/policy_op.pth'
```

You can set the opponent agent policy using `op_checkpoint`. If it's empty, the opponent agent will use the same policy
as `checkpoint`.  
We use vectorized models to accelerate the evaluation of policies. Put policies into checkpoint dir, let them compete
with each
other in parallel:

```bash
# run testing for Ant Sumo policy
python train.py task=MA_Ant_Sumo train=MA_Ant_SumoPPO test=True headless=True checkpoint='models/ant_sumo' player_pool_type=vectorized
```

There are some specific arguments for self-play evaluation, you can change them in `timechamber/tasks/train/*.yaml`:

- `games_num`: Total episode number of evaluation.
- `record_elo`: Set `True` to record the ELO rating of your policies, after evaluation, you can check the `elo.jpg` in
  your checkpoint dir.

<div align=center>
  <img src="assets/images/elo.jpg" align="center" width="400"/>
</div>

- `init_elo`: Initial ELO rating of each policy.

### Building Your Own Task

You can build your own task
follow [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/README.md#creating-an-environment)
, make sure the obs shape is correct and`info` contains `win`,`lose`and`draw`:

```python
import isaacgym
import timechamber
import torch

envs = timechamber.make(
    seed=0,
    task="MA_Ant_Sumo",
    num_envs=2,
    sim_device="cuda:0",
    rl_device="cuda:0",
)
# the obs shape should be (num_agents*num_envs,num_obs).
# the obs of training agent is (:num_envs,num_obs)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
for _ in range(20):
    obs, reward, done, info = envs.step(
        torch.rand((2 * 2,) + envs.action_space.shape, device="cuda:0")
    )
# info:
# {'win': tensor([Bool, Bool])
# 'lose': tensor([Bool, Bool])
# 'draw': tensor([Bool, Bool])}

```


## Credits

- [TimeChamber](https://github.com/inspirai/TimeChamber)
- [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
