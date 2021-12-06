import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.nn import Module
from mg_a2c.utils import get_obss_preprocessor, get_vocab

TILE_SIZE = 32
DOWNSAMPLE_SIZE = 2 # how many times to downsample image before feeding to CNN

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

class StateHandler(Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 model_dir,
                 use_text=False) -> None:
        super().__init__()
        self.use_text = use_text
        obs_space, self.preprocess_obss = get_obss_preprocessor(obs_space)
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(get_vocab(model_dir))

        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.embedding_size = ((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64

        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"],
                                               self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size,
                                   self.text_embedding_size,
                                   batch_first=True)
            self.embedding_size += self.text_embedding_size

        self.image_conv = nn.Sequential(nn.Conv2d(3, 16, (2, 2)), nn.ReLU(),
                                        nn.MaxPool2d((2, 2)),
                                        nn.Conv2d(16, 32, (2, 2)), nn.ReLU(),
                                        nn.Conv2d(32, 64, (2, 2)), nn.ReLU())

    def embed(self, state):
        preprocessed_state = self.preprocess_obss(state)
        x = preprocessed_state.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        embedding = x.reshape(x.shape[0], -1)

        if self.use_text:
            embed_text = self._get_embed_text(state.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        return embedding

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
