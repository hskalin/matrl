# Multi Agent And Task Reinforcement Learning

****

## TODO
Immediate:
- [ ] Figure out humanoid-env
- [ ] Implement PPO to try out single tasks
      
Future :
- [ ] Implement multi-task framework for feasible single tasks

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

## Credits

- [TimeChamber](https://github.com/inspirai/TimeChamber)
- [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
