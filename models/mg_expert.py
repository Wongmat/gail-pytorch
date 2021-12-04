import torch

from torch.nn import Module, Sequential, Linear, Tanh, Parameter, Embedding
from torch.distributions import Categorical, MultivariateNormal

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

from mg_a2c.utils import agent, get_model_dir


class MiniGridExpert(Module):
    def __init__(self, env, model, train_config=None) -> None:
        super().__init__()

        self.train_config = train_config

        model_dir = get_model_dir(model)
        self.agent = agent.Agent(env.observation_space,
                                 env.action_space,
                                 model_dir,
                                 argmax=True,
                                 use_memory=False,
                                 use_text=False)
        print("Expert minigrid agent loaded\n")

    def act(self, state):
        action = self.agent.get_action(state)

        return action
