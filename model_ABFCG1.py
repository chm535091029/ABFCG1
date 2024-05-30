
import torch

import math
# copy: 用于对模型进行深拷贝
import copy
from torch import optim
# from config import config
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
import torch.nn.functional as F
from Embeddings import Embedding
# from fusion_utils import GanGenerator,GanDiscriminator
from retnet_shared import *
from Embedding4ast import *
#



import warnings
import json
# 设置忽略警告

warnings.filterwarnings("ignore")
with open("processed_datasets/CSN-JAVA/csn-java_nl_vocab.json", "r") as f:
    nl_vocab_stoi = json.load(f)["nl_vocab_stoi"]
f.close()
with open("processed_datasets/CSN-JAVA/csn-java_nl_vocab.json", "r") as f:
    nl_vocab_itos = json.load(f)["nl_vocab_itos"]
f.close()
with open("processed_datasets/CSN-JAVA/csn-java_code_tokens_vocab.json", "r") as f:
    code_tokens_vocab_stoi = json.load(f)["code_tokens_vocab_stoi"]
f.close()
with open("processed_datasets/CSN-JAVA/csn-java_code_tokens_vocab.json", "r") as f:
    code_tokens_vocab_itos = json.load(f)["code_tokens_vocab_itos"]
f.close()

BOS = code_tokens_vocab_stoi["<bos>"]
EOS = code_tokens_vocab_stoi["<eos>"]
PAD = code_tokens_vocab_stoi["<pad>"]
UNK = code_tokens_vocab_stoi["<unk>"]
stoi = code_tokens_vocab_stoi
itos = code_tokens_vocab_itos

class EncoderDecoder(nn.Module):

    def __init__(self,
                 src_vocab=50265, tgt_vocab=50265, nl_len=32,code_len=128, N=6, d_model=768, d_ff=2048, h=8, dropout=0.1):

        super(EncoderDecoder, self).__init__()
        print("Start")
        # 用于将模型深度拷贝一份（相当于全新的new一个）
        c = copy.deepcopy
        # 1. 构建多头注意力机制
        attn = MultiHeadedAttention(h, d_model)
        # 2. 构建前馈神经网络
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 3. 构建位置编码

        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N+N//2)
        # self.nl_embed = Embedding(src_vocab,d_model,nl_len,None,dropout)
        self.code_embed = Embedding(tgt_vocab,d_model,code_len,None,dropout)

        self.constractive = Constrctive_loss(src_vocab,tgt_vocab,nl_len,code_len,N,
                                             d_model,d_ff,h,dropout)
        # self.fusion_init = Decoder(EncoderLayer_fusion(d_model, c(attn), c(attn),c(ff), dropout),1)
        self.fusion = Decoder(EncoderLayer_fusion(d_model, c(attn), c(attn),c(ff), dropout),3)
        self.generator = Generator(d_model, tgt_vocab)
        self.linear = nn.Linear(d_model,tgt_vocab)
        self.mse_loss = nn.MSELoss()
        self.mlp = MLP(d_model)
        self.mlp2 = MLP(d_model)

    def forward(self, nl, target_code,related_code,related_nl,related_ast_tokens,related_ast_matrice):

        # nl_embedding_output = self.nl_embed(nl)
        code_embedding_output = self.code_embed(target_code)
        #
        nl_mask = encoder_mask(nl, PAD, BOS, EOS)
        encoder_output_nl,encoder_output_related_code,encoder_output_ast,related_code,fusion_loss = self.constractive(nl,related_code,related_nl,related_ast_tokens,related_ast_matrice)
        z = torch.cat((encoder_output_ast,encoder_output_related_code),dim=1)

        z = self.mlp(z)

        z_fuse = self.fusion(encoder_output_nl,z,src_mask=None,tgt_mask=nl_mask)
        z_fuse = self.mlp2(z_fuse)
        tgt_mask = decoder_mask(target_code, EOS)
        decoder_o = self.decoder(code_embedding_output,z_fuse,src_mask=None,tgt_mask=tgt_mask)


        return self.linear(decoder_o), fusion_loss

    def encode(self, nl,related_code,related_nl,related_ast_tokens,related_ast_matrice):


        nl_mask = encoder_mask(nl, PAD, BOS, EOS)
        encoder_output_nl,encoder_output_related_code,encoder_output_ast,related_code,fusion_loss = self.constractive(nl,related_code,related_nl,related_ast_tokens,related_ast_matrice)
        z = torch.cat((encoder_output_ast,encoder_output_related_code),dim=1)

        z = self.mlp(z)

        z_fuse = self.fusion(encoder_output_nl,z,src_mask=None,tgt_mask=nl_mask)
        z_fuse = self.mlp2(z_fuse)
        return z_fuse




    def decode(self, memory, src_mask, tgt, tgt_mask):

        decoder_o = self.decoder(self.code_embed(tgt), memory, src_mask, tgt_mask)
        return self.linear(decoder_o)


class MLP(nn.Module):
    def __init__(self,d_model):
        super(MLP,self).__init__()

        self.linear1 = nn.Linear(d_model, d_model)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.relu = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(d_model, d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
    def forward(self,feature):
        feature = self.linear1(feature).transpose(-1,-2).contiguous()
        # print(feature.shape)
        feature = self.bn1(feature).transpose(-1,-2).contiguous()
        # print(feature.shape)
        # print(type(feature))
        feature = self.relu(feature)
        feature = self.linear2(feature).transpose(-1,-2).contiguous()
        feature = self.bn2(feature).transpose(-1,-2).contiguous()
        return feature
class Constrctive_loss(nn.Module):
    def __init__(self,src_vocab=50265, tgt_vocab=50265, nl_len=32,code_len=128, N=6, d_model=768, d_ff=2048, h=8, dropout=0.1,temp=0.07):
        super(Constrctive_loss, self).__init__()
        c = copy.deepcopy

        attn = MultiHeadedAttention(h, d_model)

        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.loss = nn.CrossEntropyLoss()

        self.temp = temp

        # self.nl_encoder = RetNet_wo_ffn(N,768,4096,16)
        # self.code_encoder = RetNet_wo_ffn(N,768,4096,16)

        self.shared_encoder = RetNet_shared(N,d_model,d_ff*2,h*2)

        self.nl_embed = Embedding(src_vocab,d_model,nl_len,None,dropout)
        self.code_embed = Embedding(tgt_vocab,d_model,code_len,None,dropout)
        # self.ast_embed = GATModel(5,d_ff,d_model,h)
        self.ast_embed = Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout=0.1),N)
        # self.related_nl_embed = Embedding(src_vocab,d_model,nl_len,dropout)
        self.code_len = code_len
        self.nl_len = nl_len
        # self._optimizer = optim.Adam(self.parameters(), 0.00001)

    def _select_most_similar_code(self,related_nl_embedding_output,nl_embedding_output,related_code,related_ast_tokens,related_ast_matrice):
        relate_code = []

        related_ast = []
        related_att = []
        batch_size = nl_embedding_output.size(0)
        for batch_i in range(0,batch_size):
            max_sim,max_index = 0,0
            for row in range(related_code.size(1)):
                if torch.all(related_code[batch_i, row] == 0).item() == 0:
                    cosine_sim = torch.cosine_similarity(nl_embedding_output[batch_i], related_nl_embedding_output[batch_i,row],dim=-1).mean(dim=-1)
                    if max_sim<cosine_sim:
                        # max_index = cosine_sim.argmax().item()
                        max_index = row
                        max_sim = cosine_sim
            relate_code.append(related_code[batch_i,max_index].tolist())
            related_ast.append(related_ast_tokens[batch_i,max_index].tolist())
            related_att.append(related_ast_matrice[batch_i,max_index].tolist())

        return torch.tensor(relate_code,dtype=related_code.dtype,device=device),\
                torch.tensor(related_ast,dtype=torch.long,device=device),\
                torch.tensor(related_att,dtype=torch.bool,device=device),\

    def optimize(self):
        self._optimizer.step()
    def forward(self, nl,related_code,related_nl,related_ast_tokens,related_ast_matrice):
        nl_embedding_output = self.nl_embed(nl)
        # code_embedding_output = self.code_embed(target_code)
        related_nl_embedding_output = self.nl_embed(related_nl)

        related_code, related_ast, related_ast_adj = self._select_most_similar_code(related_nl_embedding_output, nl_embedding_output,
                                                                   related_code,related_ast_tokens,related_ast_matrice)

        related_code_embedding_output = self.code_embed(related_code)
        related_code_mask = encoder_mask(related_ast, PAD, BOS, EOS)
        related_ast_embedding_output = self.code_embed(related_ast)
        encoder_output_nl = self.shared_encoder(nl_embedding_output,'nl')

        related_ast_embedding_output = self.ast_embed(related_ast_embedding_output,related_code_mask,related_ast_adj)
        encoder_output_related_code = self.shared_encoder(related_code_embedding_output,'code')

        # f1 = F.normalize(encoder_output_nl[:,0,:])
        # f2 = F.normalize(encoder_output_related_code[:, 0, :])
        # f3 = F.normalize(related_ast_embedding_output[:, 0, :])
        f1 = F.normalize(torch.mean(encoder_output_nl,dim=1))
        f2 = F.normalize(torch.mean(encoder_output_related_code,dim=1))
        f3 = F.normalize(torch.mean(related_ast_embedding_output,dim=1))
        sim_targets = torch.zeros(f1.size(0),f1.size(0)).to(device)
        sim_targets.fill_diagonal_(1)

        sim_12 = f1 @ f2.transpose(0,1).contiguous() / self.temp
        sim_21 = f2 @ f1.transpose(0,1).contiguous() / self.temp

        loss_12 = -torch.sum(F.log_softmax(sim_12, dim=1) * sim_targets, dim=1).mean()
        loss_21 = -torch.sum(F.log_softmax(sim_21, dim=1) * sim_targets, dim=1).mean()

        sim_13 = f1 @ f3.transpose(0,1).contiguous() / self.temp
        sim_31 = f3 @ f1.transpose(0,1).contiguous() / self.temp
        loss_13 = -torch.sum(F.log_softmax(sim_13, dim=1) * sim_targets, dim=1).mean()
        loss_31 = -torch.sum(F.log_softmax(sim_31, dim=1) * sim_targets, dim=1).mean()

        sim_23 = f2 @ f3.transpose(0,1).contiguous() / self.temp
        sim_32 = f3 @ f2.transpose(0,1).contiguous() / self.temp
        loss_23 = -torch.sum(F.log_softmax(sim_23, dim=1) * sim_targets, dim=1).mean()
        loss_32 = -torch.sum(F.log_softmax(sim_32, dim=1) * sim_targets, dim=1).mean()

        loss = (loss_12 + loss_23 + loss_13 + loss_21 + loss_31 + loss_32)/6
        return encoder_output_nl,encoder_output_related_code,related_ast_embedding_output,related_code,loss
class Generator(nn.Module):

    def __init__(self, d_model, vocab):

        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):

        return self.proj(x)

def clones(module, N):

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):


    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        assert N % 2 == 0
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.n_layers = N
    def forward(self, x, mask,adj=None):

        if adj is None:
            adj = mask
        # else:
        #     adj = adj.unsqueeze(1)
        for i in range(self.n_layers//2):
            x = self.layers[i](x, mask)
            x_ = self.layers[i](x, adj)
            x = x + x_
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class EncoderLayer_fusion(nn.Module):

    def __init__(self,size, self_attn, src_attn, feed_forward, dropout):
        super(EncoderLayer_fusion, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 克隆出N个DecoderLayer
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def encoder_mask(input_ids,pad_token_index,bos_token_index,eos_token_index):
    mask = (input_ids != torch.tensor(pad_token_index).to(device)) & (input_ids != torch.tensor(bos_token_index).to(device)) & (input_ids!=torch.tensor(eos_token_index).to(device))


    mask = mask.float()

    mask = mask.unsqueeze(1)
    return mask

def subsequent_mask(max_len,seq_len):
    attn_shape = (max_len, max_len)
    subsequent_mask = 1 - torch.triu(torch.ones(attn_shape), diagonal=1)
    subsequent_mask[:, seq_len:] = 0
    return subsequent_mask

def decoder_mask(input_ids,end_symbol):
    mask = []
    for batch_sample in input_ids:
        end_pos = batch_sample.tolist().index(end_symbol) if end_symbol in batch_sample.tolist() else input_ids.size(-1)
        mask.append(subsequent_mask(input_ids.size(-1),
                    end_pos).tolist())
    # print(mask)
    return torch.tensor(mask, dtype=torch.float)


def attention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:

        scores = scores.masked_fill(~mask.bool().to(device), -1e4)


    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):


        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)


        query, key, value = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]


        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )


        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )

        del query
        del key
        del value

        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):


    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):

        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
