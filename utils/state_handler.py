import torch
import torch.nn as nn
from torch.nn import Module
from mg_a2c.utils import get_obss_preprocessor, get_vocab


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
