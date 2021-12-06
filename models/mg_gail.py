import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Module
from mg_a2c.utils import get_obss_preprocessor, get_vocab
from models.nets import PolicyNetwork, ValueNetwork, Discriminator, CNNPolicyNetwork, CNNValueNetwork, CNNDiscriminator, SimpleCNNDiscriminator
from utils.funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch

TILE_SIZE = 32
DOWNSAMPLE_SIZE = 2 # how many times to downsample image before feeding to CNN

def preprocess_img(img, max_shape, output_shape, color=[147,147,147]):
    img = img.copy()
    max_height = max_shape[0]
    max_width = max_shape[1]
    output_height = output_shape[0]
    output_width = output_shape[1]
    height = img.shape[0]
    width = img.shape[1]
    if height > max_height and width > max_width:
        print("img.shape: ", img.shape)
        print("max_shape: ", max_shape)
        img = cv2.resize(img, (max_width, max_height))
    
    elif height > max_height and width <= max_width:
        print("img.shape: ", img.shape)
        print("max_shape: ", max_shape)
        img = cv2.resize(img, (width, max_height))
        width_diff = max_width - width
        left = np.floor(width_diff/2)
        right = np.ceil(width_diff/2)
        img = cv2.copyMakeBorder(img, 0, 0, int(left), int(right), cv2.BORDER_CONSTANT, value=color)
        
    elif height <= max_height and width > max_width:
        print("img.shape: ", img.shape)
        print("max_shape: ", max_shape)
        img = cv2.resize(img, (max_width, height))
        height_diff = max_height - height
        bottom = np.floor(height_diff/2)
        top = np.ceil(height_diff/2)
        img = cv2.copyMakeBorder(img, int(top), int(bottom), 0, 0, cv2.BORDER_CONSTANT, value=color)
    
    else:
        width_diff = max_width - width
        left = np.floor(width_diff/2)
        right = np.ceil(width_diff/2)
        height_diff = max_height - height
        bottom = np.floor(height_diff/2)
        top = np.ceil(height_diff/2)
        img = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT, value=color)

    img = cv2.resize(img, (output_width, output_height))
    img = img.transpose(2,0,1)
    
    return img

def reset(env):
    ob = env.reset()
    full_img, observable_img = env.render('rgb_array', tile_size=TILE_SIZE)
    ob['full_res_observable_img'] = preprocess_img(observable_img, (env.height * TILE_SIZE, env.width * TILE_SIZE), (env.height * TILE_SIZE // DOWNSAMPLE_SIZE, env.width * TILE_SIZE // DOWNSAMPLE_SIZE))
    return ob

def step(env, act):
    ob, rwd, done, info = env.step(act) 
    full_img, observable_img = env.render('rgb_array', tile_size=TILE_SIZE)
    ob['full_res_observable_img'] = preprocess_img(observable_img, (env.height * TILE_SIZE, env.width * TILE_SIZE), (env.height * TILE_SIZE // DOWNSAMPLE_SIZE, env.width * TILE_SIZE // DOWNSAMPLE_SIZE))
    return ob, rwd, done, info



if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor


class GAIL(Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 model_dir,
                 discrete,
                 log_dir,
                 train_config=None) -> None:
        super().__init__()

        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        obs_space, self.preprocess_obss = get_obss_preprocessor(obs_space)
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n - 1) // 2 - 2) * (
            (m - 1) // 2 - 2) * 64
        self.state_dim = self.image_embedding_size
        self.action_dim = action_space.n
        self.discrete = discrete
        self.train_config = train_config
        self.writer = SummaryWriter(log_dir)
        self.ckpt_path = os.path.abspath(os.path.join(log_dir, os.pardir))
        self.ckpt_path = os.path.join(self.ckpt_path, "models")
        os.makedirs(self.ckpt_path)

        # self.image_conv = nn.Sequential(nn.Conv2d(3, 16, (2, 2)), nn.ReLU(),
        #                                 nn.MaxPool2d((2, 2)),
        #                                 nn.Conv2d(16, 32, (2, 2)), nn.ReLU(),
        #                                 nn.Conv2d(32, 64, (2, 2)), nn.ReLU())

        # self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete)
        # self.v = ValueNetwork(self.state_dim)

        # self.d = Discriminator(self.state_dim, self.action_dim, self.discrete)

        self.pi = CNNPolicyNetwork(self.action_dim, self.discrete)
        self.v = CNNValueNetwork()
        if self.train_config["discrim_net"] == "SimpleCNN":
            print("Using SimpleCNN for Discriminator")
            self.d = SimpleCNNDiscriminator(self.state_dim, self.action_dim, self.discrete)
        elif self.train_config["discrim_net"] == "NatureCNN":
            print("Using NatureCNN for Discriminator")
            self.d = CNNDiscriminator(self.state_dim, self.action_dim, self.discrete)
        else:
            raise RunTimeError("not implemented: ", self.train_config["discrim_net"])

        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(get_vocab(model_dir))

    def get_networks(self):
        return [self.pi, self.v]

    # def embed(self, state):
    #     preprocessed_state = self.preprocess_obss(state)
    #     x = preprocessed_state.image.transpose(1, 3).transpose(2, 3)
    #     x = self.image_conv(x)
    #     state = x.reshape(x.shape[0], -1)
    #     return state

    def act(self, state):
        # if type(state) is not list:
        #     state = [state]
        # state = self.embed(state)
        self.pi.eval()

        state = FloatTensor(state)
        distb = self.pi(state)

        try:
            action = distb.sample().cpu().numpy()
        except:
            import pdb; pdb.set_trace()
            print("hi")

        return action

    def train(self, env, expert, render=False):
        num_iters = self.train_config["num_iters"]
        num_steps_per_iter = self.train_config["num_steps_per_iter"]
        horizon = self.train_config["horizon"]
        lambda_ = self.train_config["lambda"]
        gae_gamma = self.train_config["gae_gamma"]
        gae_lambda = self.train_config["gae_lambda"]
        eps = self.train_config["epsilon"]
        max_kl = self.train_config["max_kl"]
        cg_damping = self.train_config["cg_damping"]
        normalize_advantage = self.train_config["normalize_advantage"]

        opt_d = torch.optim.Adam(self.d.parameters(), lr=self.train_config["discrim_lr"])

        exp_rwd_iter = []

        exp_obs = []
        exp_acts = []

        steps = 0
        while steps < num_steps_per_iter:
            ep_obs = []
            ep_rwds = []

            t = 0
            done = False
            # ob = env.reset()
            # full_img, observable_img = env.render('rgb_array', tile_size=TILE_SIZE)
            # ob['full_res_observable_img'] = observable_img
            ob = reset(env)

            while not done and steps < num_steps_per_iter:
                act = expert.act(ob)
                #act = np.ones_like(act)

                # with torch.no_grad():
                #     #emb_ob = np.array(self.embed([ob])[0], dtype=np.float)
                #     emb_ob = self.embed([ob])[0].cpu().numpy()
                # ep_obs.append(emb_ob)
                # exp_obs.append(emb_ob)

                ep_obs.append(ob['full_res_observable_img'])
                exp_obs.append(ob['full_res_observable_img'])
                exp_acts.append(act)

                # if render:
                #     env.render()
                # ob, rwd, done, info = env.step(act)
                ob, rwd, done, info = step(env, act)

                ep_rwds.append(rwd)

                t += 1
                steps += 1

                if horizon is not None:
                    if t >= horizon:
                        done = True
                        break

            if done:
                exp_rwd_iter.append(np.sum(ep_rwds))
            ep_obs = FloatTensor(np.array(ep_obs))
            ep_rwds = FloatTensor(ep_rwds)

        exp_rwd_mean = np.mean(exp_rwd_iter)
        print("Expert Reward Mean: {}".format(exp_rwd_mean))

        exp_obs = FloatTensor(np.array(exp_obs))
        # exp_obs = np.array(exp_obs)
        exp_acts = FloatTensor(np.array(exp_acts))

        rwd_iter_means = []
        for i in range(num_iters):
            rwd_iter = []

            obs = []
            acts = []
            rets = []
            advs = []
            gms = []

            steps = 0
            while steps < num_steps_per_iter:
                ep_obs = []
                ep_acts = []
                ep_rwds = []
                ep_costs = []
                ep_disc_costs = []
                ep_gms = []
                ep_lmbs = []

                t = 0
                done = False
                # ob = env.reset()
                ob = reset(env)

                while not done and steps < num_steps_per_iter:
                    img = ob['full_res_observable_img']
                    img = np.expand_dims(img, 0)
                    act = self.act(img)[0]

                    # with torch.no_grad():
                    #     #emb_ob = np.array(self.embed([ob])[0], dtype=np.float)
                    #     emb_ob = self.embed([ob])[0].cpu().numpy()
                    # ep_obs.append(emb_ob)
                    # obs.append(emb_ob)

                    ep_obs.append(ob['full_res_observable_img'])
                    obs.append(ob['full_res_observable_img'])

                    ep_acts.append(act)
                    acts.append(act)

                    if render:
                        env.render()
                    # ob, rwd, done, info = env.step(act)
                    ob, rwd, done, info = step(env, act)

                    ep_rwds.append(rwd)
                    ep_gms.append(gae_gamma**t)
                    ep_lmbs.append(gae_lambda**t)

                    t += 1
                    steps += 1

                    if horizon is not None:
                        if t >= horizon:
                            done = True
                            break

                if done:
                    rwd_iter.append(np.sum(ep_rwds))

                ep_obs = FloatTensor(np.array(ep_obs))
                ep_acts = FloatTensor(np.array(ep_acts))
                ep_rwds = FloatTensor(ep_rwds)
                # ep_disc_rwds = FloatTensor(ep_disc_rwds)
                ep_gms = FloatTensor(ep_gms)
                ep_lmbs = FloatTensor(ep_lmbs)
                ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts))\
                    .squeeze().detach()
                ep_disc_costs = ep_gms * ep_costs

                ep_disc_rets = FloatTensor(
                    [sum(ep_disc_costs[i:]) for i in range(t)])
                ep_rets = ep_disc_rets / ep_gms

                rets.append(ep_rets)

                self.v.eval()
                curr_vals = self.v(ep_obs).detach()
                next_vals = torch.cat(
                    (self.v(ep_obs)[1:], FloatTensor([[0.]]))).detach()
                ep_deltas = ep_costs.unsqueeze(-1)\
                    + gae_gamma * next_vals\
                    - curr_vals

                ep_advs = FloatTensor([
                    ((ep_gms * ep_lmbs)[:t - j].unsqueeze(-1) *
                     ep_deltas[j:]).sum() for j in range(t)
                ])
                advs.append(ep_advs)

                gms.append(ep_gms)

            rwd_iter_means.append(np.mean(rwd_iter))

            obs = FloatTensor(np.array(obs))
            acts = FloatTensor(np.array(acts))
            rets = torch.cat(rets)
            advs = torch.cat(advs)
            gms = torch.cat(gms)

            if normalize_advantage:
                advs = (advs - advs.mean()) / advs.std()

            self.d.train()
            exp_scores = self.d.get_logits(exp_obs, exp_acts)
            nov_scores = self.d.get_logits(obs, acts)

            opt_d.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                exp_scores, torch.zeros_like(exp_scores)
            ) \
                + torch.nn.functional.binary_cross_entropy_with_logits(
                    nov_scores, torch.ones_like(nov_scores)
                )
            loss.backward()
            opt_d.step()

            exp_probs = self.d(exp_obs, exp_acts)
            exp_acc = (exp_probs >= 0.5).sum() / exp_probs.shape[0]
            pred_probs = self.d(obs, acts)
            pred_acc = (pred_probs <= 0.5).sum() / pred_probs.shape[0]
            self.writer.add_scalar("avg reward", np.mean(rwd_iter), i)
            self.writer.add_scalar("disc_loss", loss, i)
            self.writer.add_scalar("disc expert accuracy", exp_acc, i)
            self.writer.add_scalar("disc pred accuracy", pred_acc, i)  

            self.v.train()
            old_params = get_flat_params(self.v).detach()
            old_v = self.v(obs).detach()

            print("Iterations: {},   Reward Mean: {:.3f}, D-Loss: {:.6f}, disc expert acc: {:.3f}, disc pred acc: {:.3f}".format(
                i + 1, np.mean(rwd_iter), loss, exp_acc, pred_acc))

            def constraint():
                return ((old_v - self.v(obs))**2).mean()

            grad_diff = get_flat_grads(constraint(), self.v)

            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_diff, v), self.v)\
                    .detach()

                return hessian

            g = get_flat_grads(
                ((-1) * (self.v(obs).squeeze() - rets)**2).mean(),
                self.v).detach()
            s = conjugate_gradient(Hv, g).detach()

            Hs = Hv(s).detach()
            alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))

            new_params = old_params + alpha * s

            set_params(self.v, new_params)

            self.pi.train()
            old_params = get_flat_params(self.pi).detach()
            old_distb = self.pi(obs)

            def L():
                distb = self.pi(obs)

                return (advs * torch.exp(
                    distb.log_prob(acts) - old_distb.log_prob(acts).detach())
                        ).mean()

            def kld():
                distb = self.pi(obs)

                if self.discrete:
                    old_p = old_distb.probs.detach()
                    p = distb.probs

                    return (old_p * (torch.log(old_p) - torch.log(p)))\
                        .sum(-1)\
                        .mean()

                else:
                    old_mean = old_distb.mean.detach()
                    old_cov = old_distb.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)

                    return (0.5) * ((old_cov / cov).sum(-1) +
                                    (((old_mean - mean)**2) / cov).sum(-1) -
                                    self.action_dim + torch.log(cov).sum(-1) -
                                    torch.log(old_cov).sum(-1)).mean()

            grad_kld_old_param = get_flat_grads(kld(), self.pi)

            def Hv(v):
                hessian = get_flat_grads(torch.dot(grad_kld_old_param, v),
                                         self.pi).detach()

                return hessian + cg_damping * v

            g = get_flat_grads(L(), self.pi).detach()

            s = conjugate_gradient(Hv, g).detach()
            Hs = Hv(s).detach()

            new_params = rescale_and_linesearch(g, s, Hs, max_kl, L, kld,
                                                old_params, self.pi)

            disc_causal_entropy = ((-1) * gms * self.pi(obs).log_prob(acts))\
                .mean()
            grad_disc_causal_entropy = get_flat_grads(disc_causal_entropy,
                                                      self.pi)
            new_params += lambda_ * grad_disc_causal_entropy

            set_params(self.pi, new_params)

            if i % self.train_config["log_every"] == 0 and i > 0:
                torch.save(self.pi.state_dict(), os.path.join(self.ckpt_path,
                                                        "policy_%d.ckpt" % i))
                torch.save(self.v.state_dict(), os.path.join(self.ckpt_path, "value_%d.ckpt" % i))
                torch.save(self.d.state_dict(),
                        os.path.join(self.ckpt_path, "discriminator_%d.ckpt" % i))

        return exp_rwd_mean, rwd_iter_means
