import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json
import pickle
import argparse

import torch
import gym

from models.nets import Expert
from models.gail import GAIL

def main(env_name, visualize, save_name):
    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)

    if env_name not in [
            "MiniGrid-DoorKey-5x5-v0", "CartPole-v1", "Pendulum-v0",
            "BipedalWalker-v3"
    ]:
        print("The environment name is wrong!")
        return

    expert_ckpt_path = "experts"
    expert_ckpt_path = os.path.join(expert_ckpt_path, env_name)

    with open(os.path.join(expert_ckpt_path, "model_config.json")) as f:
        expert_config = json.load(f)

    # ckpt_path = os.path.join(ckpt_path, env_name)
    ckpt_path = os.path.join(ckpt_path, env_name, args.save_name)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)[env_name]

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    env = gym.make(env_name)
    env.reset()

    state_dim = len(env.observation_space.high)
    if env_name in ["CartPole-v1"]:
        discrete = True
        action_dim = env.action_space.n
    else:
        discrete = False
        action_dim = env.action_space.shape[0]

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    expert = Expert(state_dim, action_dim, discrete,
                    **expert_config).to(device)
    expert.pi.load_state_dict(
        torch.load(os.path.join(expert_ckpt_path, "policy.ckpt"),
                   map_location=device))

    log_dir = os.path.join(ckpt_path, "log")
    os.makedirs(log_dir, exist_ok=True)
    model = GAIL(state_dim, action_dim, discrete, log_dir=log_dir, train_config=config).to(device)

    results = model.train(env, expert, render=visualize)

    env.close()

    with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(model.pi.state_dict(), os.path.join(ckpt_path,
                                                       "policy.ckpt"))
    if hasattr(model, "v"):
        torch.save(model.v.state_dict(), os.path.join(ckpt_path, "value.ckpt"))
    if hasattr(model, "d"):
        torch.save(model.d.state_dict(),
                   os.path.join(ckpt_path, "discriminator.ckpt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name",
                        type=str,
                        default="CartPole-v1",
                        help="Type the environment name to run. \
            The possible environments are \
                [CartPole-v1, Pendulum-v0, BipedalWalker-v3]")
    parser.add_argument("--visualize",
                        action="store_true",
                        default=False,
                        help="Visualize expert trajectories")
    parser.add_argument("--save_name",
                        type=str)
    args = parser.parse_args()

    main(**vars(args))
