import torch

import torch.nn as nn
from torch.nn import Module, Sequential, Linear, Tanh, Parameter, Embedding
from torch.distributions import Categorical, MultivariateNormal

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class PolicyNetwork(Module):
    def __init__(self, state_dim, action_dim, discrete) -> None:
        super().__init__()

        self.net = Sequential(
            Linear(state_dim, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, action_dim),
        )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if not self.discrete:
            self.log_std = Parameter(torch.zeros(action_dim))

    def forward(self, states):
        if self.discrete:
            probs = torch.softmax(self.net(states), dim=-1)
            distb = Categorical(probs)
        else:
            mean = self.net(states)

            std = torch.exp(self.log_std)
            cov_mtx = torch.eye(self.action_dim) * (std**2)

            distb = MultivariateNormal(mean, cov_mtx)

        return distb


class ValueNetwork(Module):
    def __init__(self, state_dim) -> None:
        super().__init__()

        self.net = Sequential(
            Linear(state_dim, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, 1),
        )

    def forward(self, states):
        return self.net(states)


class Discriminator(Module):
    def __init__(self, state_dim, action_dim, discrete) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        if self.discrete:
            self.act_emb = Embedding(action_dim, state_dim)
            self.net_in_dim = 2 * state_dim
        else:
            self.net_in_dim = state_dim + action_dim

        self.net = Sequential(
            Linear(self.net_in_dim, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, 50),
            Tanh(),
            Linear(50, 1),
        )

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        if self.discrete:
            actions = self.act_emb(actions.long())

        if len(actions.shape) == 3:
            actions = actions.squeeze(1)
        sa = torch.cat([states, actions], dim=-1)

        return self.net(sa)


class Expert(Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 discrete,
                 train_config=None) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)

    def get_networks(self):
        return [self.pi]

    def act(self, state):
        self.pi.eval()

        state = FloatTensor(state)
        distb = self.pi(state)

        action = distb.sample().detach().cpu().numpy()

        return action

class NatureCNN(Module):
    def __init__(self, out_dim=64):
        """
        Uses same convolutional layers as original NatureCNN from
        stable_baselines3 library except it doesn't expect constant input shape. Instead, it uses 1x1
        convolution and global average pooling to keep output shape constant regardless
        of input shape.
        """
        super(NatureCNN, self).__init__()
        self.out_dim = out_dim
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=4, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_dim, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
        
        self.linear = nn.Sequential(
            nn.Linear(out_dim, out_dim, bias=True),
            nn.ReLU()
        )
    
    def forward(self, x):
        # [B, H, W, C]
        feats = self.cnn_feature_extractor(x)
        # [B, C]
        feats = feats.mean(-1).mean(-1)
        feats = self.linear(feats)
        return feats

class SimpleCNN(Module):
    def __init__(self, out_dim=64):
        super(SimpleCNN, self).__init__()
        self.out_dim = out_dim
        self.cnn_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, out_dim, kernel_size=1, stride=1),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(out_dim, out_dim, bias=True),
            nn.ReLU()
        )
    
    def forward(self, x):
        # [B, H, W, C]
        feats = self.cnn_feature_extractor(x)
        # [B, C]
        feats = feats.mean(-1).mean(-1)
        feats = self.linear(feats)
        return feats




class CNNPolicyNetwork(Module):
    def __init__(self, action_dim, discrete):
        super(CNNPolicyNetwork, self).__init__()
        self.discrete = discrete

        self.action_dim = action_dim

        self.cnn = NatureCNN()

        self.mlp = nn.Sequential(
            nn.Linear(self.cnn.out_dim, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, action_dim, bias=True)
        )

        if not self.discrete:
            self.log_std = Parameter(torch.zeros(action_dim))
    
    #  if self.discrete:
    #         probs = torch.softmax(self.net(states), dim=-1)
    #         distb = Categorical(probs)
    #     else:
    #         mean = self.net(states)

    #         std = torch.exp(self.log_std)
    #         cov_mtx = torch.eye(self.action_dim) * (std**2)

    #         distb = MultivariateNormal(mean, cov_mtx)

    #     return distb

    def forward(self, states):
        if self.discrete:
            feats = self.cnn(states)
            out = self.mlp(feats)
            probs = torch.softmax(out, dim=-1)
            # if torch.isnan(probs).any():
            #     import pdb; pdb.set_trace()
            #     print("isnan")
            # probs[probs < 0.0] = 0.0
            distb = Categorical(probs)

        else:
            feats = self.cnn(states)
            mean = self.mlp(feats)

            std = torch.exp(self.log_std)
            cov_mtx = torch.eye(self.action_dim) * (std**2)

            distb = MultivariateNormal(mean, cov_mtx)

        return distb

class CNNValueNetwork(Module):
    def __init__(self):
        super(CNNValueNetwork, self).__init__()

        self.cnn = NatureCNN()

        self.mlp = nn.Sequential(
            nn.Linear(self.cnn.out_dim, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 1, bias=True)
        ) 
    
    def forward(self, states):
        feats = self.cnn(states)
        out = self.mlp(feats)
        return out

class CNNDiscriminator(Module):
    def __init__(self, state_dim, action_dim, discrete):
        super(CNNDiscriminator, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        self.cnn = NatureCNN()

        # if discrete output, learn an embedding to map actions to state dimensionality
        # could also encode actions using do one-hot encoding, but it's whatever
        if self.discrete:
            self.act_emb = Embedding(action_dim, state_dim)
            self.net_in_dim = self.cnn.out_dim + state_dim
        else:
            self.net_in_dim = self.cnn_out_dim + action_dim


        self.mlp = nn.Sequential(
           nn.Linear(self.net_in_dim, 32, bias=True),
           nn.ReLU(),
           nn.Linear(32, 32, bias=True),
           nn.ReLU(),
           nn.Linear(32, 1, bias=True)
        )

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        feats = self.cnn(states)
        if self.discrete:
            actions = self.act_emb(actions.long())

        if len(actions.shape) == 3:
            actions = actions.squeeze(1)
        sa = torch.cat([feats, actions], dim=-1)

        return self.mlp(sa)

class SimpleCNNDiscriminator(Module):
    def __init__(self, state_dim, action_dim, discrete):
        super(SimpleCNNDiscriminator, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete

        self.cnn = SimpleCNN()

        # if discrete output, learn an embedding to map actions to state dimensionality
        # could also encode actions using do one-hot encoding, but it's whatever
        if self.discrete:
            self.act_emb = Embedding(action_dim, state_dim)
            self.net_in_dim = self.cnn.out_dim + state_dim
        else:
            self.net_in_dim = self.cnn_out_dim + action_dim


        self.mlp = nn.Sequential(
           nn.Linear(self.net_in_dim, 1, bias=True),
        )

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        feats = self.cnn(states)
        if self.discrete:
            actions = self.act_emb(actions.long())

        if len(actions.shape) == 3:
            actions = actions.squeeze(1)
        sa = torch.cat([feats, actions], dim=-1)

        return self.mlp(sa)

class SimplestCNNDiscriminator(Module):
    def __init__(self, state_dim, action_dim, discrete, out_dim=16):
        super(SimplestCNNDiscriminator, self).__init__()
        self.out_dim = out_dim
        self.discrete = discrete
        self.cnn_feature_extractor=nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, out_dim, kernel_size=1, stride=1)
        )

        # if discrete output, learn an embedding to map actions to state dimensionality
        # could also encode actions using do one-hot encoding, but it's whatever
        if self.discrete:
            self.act_emb = Embedding(action_dim, state_dim)
            self.net_in_dim = self.out_dim + state_dim
        else:
            self.net_in_dim = self.out_dim + action_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.net_in_dim, 1, bias=True)
        )

    def forward(self, states, actions):
        return torch.sigmoid(self.get_logits(states, actions))

    def get_logits(self, states, actions):
        feats = self.cnn_feature_extractor(states)
        feats = feats.mean(-1).mean(-1)
        if self.discrete:
            actions = self.act_emb(actions.long())
        if len(actions.shape) == 3:
            actions = actions.squeeze(1)
        sa = torch.cat([feats, actions], dim=-1)

        return self.mlp(sa)

class SimplestCNNDiscriminatorV2(Module):
    def __init__(self, state_dim, action_dim, out_dim=8):
        super(SimpleCNN, self).__init__()
        self.out_dim = out_dim
        self.cnn_feature_extractor=nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(8, out_dim, kernel_size=1, stride=1),
        )

        # if discrete output, learn an embedding to map actions to state dimensionality
        # could also encode actions using do one-hot encoding, but it's whatever
        if self.discrete:
            self.act_emb = Embedding(action_dim, action_dim)
            self.net_in_dim = self.cnn.out_dim + action_dim
        else:
            self.net_in_dim = self.cnn_out_dim + action_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.net_in_dim, 1, bias=True)
        )

        def forward(self, x):
            feats = self.cnn_feature_extractor(x)
            feats = feats.mean(-1).mean(-1)
            return torch.sigmoid(self.get_logits(feats))

        def get_logits(self, states, actions):
            feats = self.cnn_feature_extractor(states)
            if self.discrete:
                actions = self.act_emb(actions.long())
            if len(actions.shape) == 3:
                actions = actions.squeeze(1)
            sa = torch.cat([feats, actions], dim=-1)

            return self.mlp(sa)


if __name__ == "__main__":
    img = torch.rand(4,3,80,80)
    actions = torch.ones((4))
    action_dim = 7
    state_dim = 64

    v_net = CNNValueNetwork() 
    p_net = CNNPolicyNetwork(action_dim, discrete=True)
    d_net = CNNDiscriminator(state_dim, action_dim, discrete=True)
    d_net_simple = SimpleCNNDiscriminator(state_dim, action_dim, discrete=True)

    v_out = v_net(img)
    import pdb; pdb.set_trace()
    p_out = p_net(img)
    d_out = d_net(img, actions)
    d_out_simple = d_net_simple(img, actions)
    import pdb; pdb.set_trace()
    print("done")



