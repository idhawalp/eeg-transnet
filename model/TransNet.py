import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange

def attention(query, key, value):
    dim = query.size(-1)
    scores = torch.einsum('bhqd,bhkd->bhqk', query, key) / dim**.5
    attn = F.softmax(scores, dim=-1)
    out = torch.einsum('bhqk,bhkd->bhqd', attn, value)
    return out, attn

class VarPoold(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self, x):
        t = x.shape[2]
        out_shape = (t - self.kernel_size) // self.stride + 1
        out = []
        for i in range(out_shape):
            index = i*self.stride
            input_slice = x[:, :, index:index+self.kernel_size]
            # Numeric Overflow Fix: Compute variance in float32 for FP16 stability
            var_val = input_slice.var(dim=-1, keepdim=True).float() 
            output = torch.log(torch.clamp(var_val, 1e-6, 1e6)).to(x.dtype)
            out.append(output)
        out = torch.cat(out, dim=-1)
        return out

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.d_k = self.d_v = d_model // n_head
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, n_head*self.d_k)
        self.w_k = nn.Linear(d_model, n_head*self.d_k)
        self.w_v = nn.Linear(d_model, n_head*self.d_v)
        self.w_o = nn.Linear(n_head*self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, query, key, value):
        q = rearrange(self.w_q(query), "b n (h d) -> b h n d", h=self.n_head)
        k = rearrange(self.w_k(key), "b n (h d) -> b h n d", h=self.n_head)
        v = rearrange(self.w_v(value), "b n (h d) -> b h n d", h=self.n_head)
        out, _ = attention(q, k, v)
        out = rearrange(out, 'b h q d -> b q (h d)')
        return self.dropout(self.w_o(out))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.w_2(self.dropout(self.act(self.w_1(x)))))

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, fc_ratio, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.multihead_attention = MultiHeadedAttention(embed_dim, num_heads, attn_drop)
        self.feed_forward = FeedForward(embed_dim, embed_dim*fc_ratio, fc_drop)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
    def forward(self, data):
        out = data + self.multihead_attention(self.layernorm1(data), self.layernorm1(data), self.layernorm1(data))
        return out + self.feed_forward(self.layernorm2(out))

class TransNet(nn.Module):
    def __init__(self, num_classes=4, num_samples=800, num_channels=62, embed_dim=32, pool_size=50, 
    pool_stride=15, num_heads=8, fc_ratio=4, depth=4, attn_drop=0.5, fc_drop=0.5):
        super().__init__()
        self.temp_conv1 = nn.Conv2d(1, embed_dim//4, (1, 15), padding=(0, 7))
        self.temp_conv2 = nn.Conv2d(1, embed_dim//4, (1, 25), padding=(0, 12))
        self.temp_conv3 = nn.Conv2d(1, embed_dim//4, (1, 51), padding=(0, 25))
        self.temp_conv4 = nn.Conv2d(1, embed_dim//4, (1, 65), padding=(0, 32))
        self.bn1 = nn.BatchNorm2d(embed_dim)
        self.spatial_conv = nn.Conv2d(embed_dim, embed_dim, (num_channels, 1))
        self.bn2 = nn.BatchNorm2d(embed_dim)
        self.elu = nn.ELU()
        self.var_pool = VarPoold(pool_size, pool_stride)
        self.avg_pool = nn.AvgPool1d(pool_size, pool_stride)
        temp_embedding_dim = (num_samples - pool_size) // pool_stride + 1
        self.dropout = nn.Dropout()
        self.transformer_encoders = nn.ModuleList([TransformerEncoder(embed_dim, num_heads, fc_ratio, attn_drop, fc_drop) for _ in range(depth)])
        self.conv_encoder = nn.Sequential(nn.Conv2d(temp_embedding_dim, temp_embedding_dim, (2, 1)), nn.BatchNorm2d(temp_embedding_dim), nn.ELU())
        self.classify = nn.Linear(embed_dim*temp_embedding_dim, num_classes)

    def forward(self, x):
        # Compact multi-scale temporal bank
        x = self.bn1(torch.cat((self.temp_conv1(x.unsqueeze(1)), self.temp_conv2(x.unsqueeze(1)), self.temp_conv3(x.unsqueeze(1)), self.temp_conv4(x.unsqueeze(1))), 1))
        x = self.elu(self.bn2(self.spatial_conv(x))).squeeze(2)
        x1, x2 = rearrange(self.dropout(self.avg_pool(x)), 'b d n -> b n d'), rearrange(self.dropout(self.var_pool(x)), 'b d n -> b n d')
        for encoder in self.transformer_encoders:
            x1, x2 = encoder(x1), encoder(x2)
        x = self.conv_encoder(torch.cat((x1.unsqueeze(2), x2.unsqueeze(2)), 2))
        return self.classify(x.reshape(x.size(0), -1))