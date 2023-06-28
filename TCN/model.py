# load packages
import pandas as pd

import numpy as np

from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

import torch
import torch.nn.functional as F
from torch.utils import data
from torchinfo import summary
import torch.nn as nn
import torch.optim as optim
from torch import nn, einsum
import torch.nn.functional as F
from TCN.tcn import TemporalConvNet
from torch.utils.data import WeightedRandomSampler
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # (b, n(65), dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # q, k, v   (b, h, n, dim_head(64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x=x.permute(0,2,1)
        y1 = self.tcn(x)
        return self.linear(y1.permute(0,2,1))

class TABL(nn.Module):
    def __init__(self,n_b,d_b,n_a,d_a,beta=0.99):
        super(TABL, self).__init__()
        self.n_b=n_b
        self.d_b=d_b
        self.n_a=n_a
        self.d_a=d_a
        self.beta=beta
        self.W1=nn.Parameter(torch.randn(1,self.n_a,self.n_b))
        self.W=nn.Parameter(torch.randn(1,self.d_b,self.d_b))
        self.softmax=nn.Softmax(dim=-1)
        self.W2=nn.Linear(self.d_b,self.d_a)
        self.activ=nn.ReLU()
    def forward(self,x):
        b,n,d=x.shape
        W1=self.W1.repeat(b,1,1)
        W=self.W.repeat(b,1,1)
        x=torch.bmm(W1,x)
        E=torch.bmm(x,W)
        A=self.softmax(E)
        x=self.beta*(x*A)+(1-self.beta)*x
        y=self.activ(self.W2(x))
        return y

class C_TABL(nn.Module):
    def __init__(self,time_slices,dim,num_classes):
        super(C_TABL, self).__init__()
        self.tcn=nn.Sequential(
            TABL(time_slices,dim,60,10),
            nn.ReLU(),
            TABL(60,10,120,5),
            nn.ReLU(),
            TABL(120,5,3,1),
            nn.ReLU(),
        )
    def forward(self,x):
        x=self.tcn(x)
        x=x.squeeze(2)
        forecast_y = torch.softmax(x, dim=1)
        return forecast_y


class DTNN(nn.Module):
    def __init__(self, *, time_slices, num_classes=2, dim=30, kernel_size=2, num_channels=None, depth=6, heads=3,
                 mlp_dim=512,
                 pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        assert pool in {'cls', 'mean'}
        self.pos_embedding = nn.Parameter(torch.zeros(1, time_slices + 1, dim))
        #self.cls_token = nn.Parameter(torch.ones(1, 1, dim)*1e-3)  # nn.Parameter()定义可学习参数
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        tcn_channel=[2*dim]*3 if not num_channels else num_channels
        self.tcn_emb = TCN(dim, dim, tcn_channel, kernel_size=kernel_size, dropout=dropout)
        self.tabl_emb=nn.Sequential(
            TABL(time_slices,dim,time_slices,2*dim),
            TABL(time_slices,2*dim,2*time_slices,2*dim),
            TABL(2*time_slices,2*dim,time_slices,dim)
        )
        self.emb = nn.Linear(dim*3, dim)
        nn.init.eye_(self.emb.weight)
        nn.init.constant_(self.emb.bias, 0)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.GELU(),
        )

    def forward(self, x):
        b, n, _ = x.shape  # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值
        x1=self.tcn_emb(x)
        x2=self.tabl_emb(x)
        x=torch.cat((x,x1,x2),dim=2)
        x = self.emb(x)
        for i in reversed(range(1, n)):
            x[:, i, :] = x[:, i, :] - x[:, i - 1, :]
        cls_tokens = repeat(self.cls_token, '() n d -> b n d',
                            b=b)  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # 将cls_token拼接到patch token中去       (b, n+1, dim)
        x = x + self.pos_embedding[:, :(n + 1)]  # 加位置嵌入（直接加）      (b, n+1, dim)
        x = self.dropout(x)
        x = self.transformer(x)  # (b, n+1, dim)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]  # (b,dim)
        x = self.to_latent(x)  # Identity (b, dim)
        x = self.mlp_head(x)
        forecast_y = torch.softmax(x, dim=1)
        return forecast_y  # (b, num_classes)
class SVM(nn.Module):
    def __init__(self, time_slices,dim, num_classes):
        super(SVM, self).__init__()
        self.linear = nn.Linear(time_slices * dim, num_classes)

    def forward(self, x):
        b, n, d = x.shape
        x=x.resize(b,n*d)
        return self.linear(x)


class MLP(torch.nn.Module):
    def __init__(self, time_slices,dim,num_classes, n_hidden=128):
        super(MLP, self).__init__()
        # 两层感知机
        self.hidden = torch.nn.Linear(time_slices * dim, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, num_classes)
        self.activation=torch.nn.LeakyReLU()

    def forward(self, x):
        b, n, d = x.shape
        x = x.resize(b, n * d)
        x = self.activation(self.hidden(x))
        x = self.predict(x)
        return x

class LSTM(torch.nn.Module):
    def __init__(self,time_slices,dim,num_classes,n_hidden=40):
        super(LSTM, self).__init__()
        self.lstm=torch.nn.LSTM(input_size=dim,hidden_size=n_hidden,batch_first=True)
        self.activation=torch.nn.LeakyReLU()
        self.classifier=torch.nn.Linear(time_slices*n_hidden,num_classes)
    def forward(self, x):
        output,(ht, ct)=self.lstm(x)
        output=self.activation(output)
        b, n, d = output.shape
        output = output.resize(b,n*d)
        output=self.classifier(output)
        return output

class CNN(torch.nn.Module):
    def __init__(self,time_slices,dim,num_classes):
        super(CNN, self).__init__()
        self.layer1=torch.nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(4,dim))#(b,1,100,40)->(b,16,97,1)
        self.layer2=torch.nn.Conv1d(in_channels=16,out_channels=16,kernel_size=4)#(b,16,97)->(b,16,94)
        self.layer3=torch.nn.MaxPool1d(kernel_size=2)#(b,16,94)->(b,16,47)
        self.layer4=torch.nn.Conv1d(in_channels=16,out_channels=32,kernel_size=3)#(b,16,47)->(b,32,45)
        self.layer5 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3)#(b,32,45)->(b,32,43)
        self.layer6 = torch.nn.MaxPool1d(kernel_size=2)#(b,32,43)->(b,32,21)
        finchanl=int((int((time_slices-6)/2+1)-4)/2+1)
        self.activation = torch.nn.LeakyReLU()
        self.Fullcon=nn.Sequential(nn.Linear(32*finchanl,32),
                                   nn.Linear(32,num_classes))

    def forward(self, x):
        b, n, d = x.shape
        x=x.unsqueeze(1)#(b,1,n,d)
        x=self.layer1(x)
        x=self.squeeze()
        x=self.layer2(x)
        x=self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x=self.activation(x)
        return self.Fullcon(x)

class CNN_LSTM(torch.nn.Module):
    def __init__(self, time_slices, dim, num_classes):
        super(CNN_LSTM, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, dim),padding=(2,0))  # (b,1,100,40)->(b,16,100,1)
        self.layer2 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=5,padding=2)  # (b,16,100)->(b,16,100)
        self.layer3 = torch.nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5,padding=2)  # (b,16,100)->(b,32,100)
        self.layer4 = torch.nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5,padding=2)  # (b,32,100)->(b,32,100)
        self.activation = torch.nn.PReLU()
        self.lstm=torch.nn.LSTM(input_size=time_slices,hidden_size=32,batch_first=True)
        self.classifier = torch.nn.Sequential(nn.Linear(32*32,64),
                                              nn.Linear(64,num_classes))

    def forward(self, x):
        b, n, d = x.shape
        x=self.layer1(x.unsqueeze(1))
        x=self.layer2(x.squeeze(3))
        x = self.layer3(x)
        x = self.layer4(x)
        output, (ht, ct) = self.lstm(x)
        output=self.activation(output)
        return self.classifier(output.reshape(b,-1))