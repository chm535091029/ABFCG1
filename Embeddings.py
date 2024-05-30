import os
# from data_process import *
import torch
import torch.nn as nn
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import json
from tqdm import tqdm
import random
from py2neo import Graph, Node, Relationship, NodeMatcher
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.init as init
import math
from transformers import AutoModel
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
# from model_gan_bert import nl_encoder



class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        # init.xavier_uniform_(self.weight)
        # init.xavier_uniform_(self.bias)
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias



class Embedding(nn.Module):
    def  __init__(self, vocab_size,hidden_size,max_position_embeddings,pretrain_model,hidden_dropout_prob):
        super(Embedding, self).__init__()

        e_model = pretrain_model
        # self.word_embeddings = nn.Embedding(
        #     vocab_size, hidden_size, padding_idx=1
        # )

        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=1
        )


        # self.position_embeddings = nn.Embedding(
        #     max_position_embeddings,hidden_size,padding_idx=1
        # )
        self.position_embeddings = nn.Embedding(
            514, hidden_size, padding_idx=1
        )
        if pretrain_model:
            word_embeddings_pretrained_state_dict = e_model.embeddings.word_embeddings.state_dict()
            self.word_embeddings.load_state_dict(word_embeddings_pretrained_state_dict)
            position_embeddings_pretrained_state_dict = e_model.embeddings.position_embeddings.state_dict()
            self.position_embeddings.load_state_dict(position_embeddings_pretrained_state_dict)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # if self.task_specific_tokens:
        #     self.task_embeddings = nn.Embedding(20, hidden_size)
        # init.xavier_uniform_(self.word_embeddings.weight)
        # init.xavier_uniform_(self.position_embeddings.weight)
        # init.xavier_uniform_(self.dropout.weight)
    def forward(self, input_ids):
        # print(input_ids.shape)
        seq_length = input_ids.size(-1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_is = position_ids.unsqueeze(0).expand_as(input_ids)
        # print(input_ids.shape)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_is)

        embeddings = words_embeddings + position_embeddings


        embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)
        # (batch_size,seq_length,hidden_size)
        return embeddings

