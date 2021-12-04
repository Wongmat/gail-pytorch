import os
import json
import pickle
import argparse

import torch
import gym

from models.nets import Expert
from models.gail import GAIL
from models.mg_gail import GAIL as MG_GAIL
from models.mg_expert import MiniGridExpert


def main(args):
    env_name = args.env_name
    expert = args.expert
    minigrid = args.minigrid
    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    if env_name not in [
            "MountainCar-v0", "CartPole-v1", "Pendulum-v0", "BipedalWalker-v3",
            "MiniGrid-DoorKey-5x5-v0"
    ]:
        print("The environment name is wrong!")
        return

    expert_ckpt_path = "experts"
    expert_ckpt_path = os.path.join(expert_ckpt_path, env_name)

    with open(os.path.join(expert_ckpt_path, "model_config.json")) as f:
        expert_config = json.load(f)

    ckpt_path = os.path.join(ckpt_path, env_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)[env_name]

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    env = gym.make(env_name)
    obs = env.reset()

    state_dim = len(env.observation_space.high) if not minigrid else 0
    if env_name in ["CartPole-v1", "MiniGrid-DoorKey-5x5-v0"]:
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if expert:
        if not minigrid:
            actor = Expert(state_dim, action_dim, discrete,
                           **expert_config).to(device)
            actor.pi.load_state_dict(
                torch.load(os.path.join(expert_ckpt_path, "policy.ckpt"),
                           map_location=device))
            print('Expert agent loaded')
        else:
            actor = MiniGridExpert(env, env_name, **expert_config).to(device)

    elif minigrid:
        actor = MG_GAIL(env.observation_space, env.action_space, env_name,
                        discrete, config).to(device)
        actor.pi.load_state_dict(
            torch.load(os.path.join(ckpt_path, "policy.ckpt"),
                       map_location=device))
        print('Trained minigrid agent loaded')

    else:
        actor = GAIL(state_dim, action_dim, discrete, config).to(device)
        actor.pi.load_state_dict(
            torch.load(os.path.join(ckpt_path, "policy.ckpt"),
                       map_location=device))
        print('Trained agent loaded')

    for _ in range(1000):
        env.render()
        action = actor.act(obs)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name",
                        type=str,
                        default="CartPole-v1",
                        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3]")
    parser.add_argument("--expert",
                        action="store_true",
                        default=False,
                        help="Visualize expert trajectories")
    parser.add_argument("--minigrid",
                        action="store_true",
                        default=False,
                        help="Visualize minigrid agent trajectories")
    args = parser.parse_args()

    main(args)
