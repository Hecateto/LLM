import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
import pandas as pd

from torch.utils.data import IterableDataset, Dataset
import json
import numpy as np
from transformers import  PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, DataCollatorForTokenClassification, AutoConfig
from dataset import SFTDataset, LLMDataset

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f'{tuple(self.weight.shape)}, eps={self.eps}'

'''
[a,b,c,d] -> [a,b],[c,d] -> [-b,a],[-d,c]
'''
def rotate_half(x: torch.Tensor):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

'''
q*cos = [a*cos, b*cos]
r(q)*sin = [-b*sin, a*sin]
sum = [a*cos - b*sin, b*cos + a*sin]
'''
def apply_rotate_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int=2):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)    # history positions
    # train/prefill
    if q.shape[1] == k.shape[1]:
        q_embed = (q * cos) + (rotate_half(q) * sin)
    # inference
    else:
        q_embed = (q * cos[:, -1:]) + (rotate_half(q) * sin[:, -1:])    # 最新位置的旋转信息
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        freqs = t @ inv_freq.unsqueeze(0)
        freqs = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def forward(self, q, k):
        cos = self.cos_cached[:k.shape[1], :].unsqueeze(0)
        sin = self.sin_cached[:k.shape[1], :].unsqueeze(0)
        return apply_rotate_pos_emb(q, k, cos, sin)

def l2norm(x: torch.FloatTensor, dim: int=-1, eps: float=1e-6):
    inv_norm = torch.rsqrt(x.pow(2).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm

def repeat_kv(hidden_states: torch.Tensor, n_rep: int):
    batch, slen, num_kv_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_kv_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_kv_heads * n_rep, head_dim)

'''
X = Conv1D(X_{t-k+1:t}), k=4
X = SiLU(X)
'''
def torch_causal_conv1d_update(
        hidden_states,
        conv_state,
        weight,
        bias=None
):
    # [b, h, s]
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    # weight: [h, 1, kernel_size], groups=hidden_size表示每个通道独立进行卷积, padding=0确保因果性
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias=bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out

class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.k_cache, self.v_cache = None, None
        self.is_causal = True
        self.flash_attn = self.config.flash_attn
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.gate_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.residual_dropout = nn.Dropout(self.dropout)
        self.attention_dropout = nn.Dropout(self.dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, attention_mask: torch.Tensor=None, cache_params=None):
        b, s = x.shape[:2]
        q, k, v, z = self.q_proj(x), self.k_proj(x), self.v_proj(x), self.gate_proj(x)

        if cache_params is not None:
            if cache_params.key_cache[self.layer_idx] is not None:
                k = torch.cat([cache_params.key_cache[self.layer_idx], k], dim=1)
                v = torch.cat([cache_params.value_cache[self.layer_idx], v], dim=1)
            cache_params.key_cache[self.layer_idx] = k
            cache_params.value_cache[self.layer_idx] = v

        q = q.view(b, q.shape[1], self.num_heads, self.head_dim)
        k = k.view(b, k.shape[1], self.num_key_value_heads, self.head_dim)
        v = v.view(b, v.shape[1], self.num_key_value_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(q, k)
        k = repeat_kv(k, self.num_key_value_heads)
        v = repeat_kv(v, self.num_key_value_heads)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn and s > 1:
            attn_mask = None
            if attention_mask is not None:
                mask = torch.triu(torch.ones(b, s, s), diagonal=1).bool().to(attention_mask.device) # [b,s,s]
                attention_mask = attention_mask.bool().unsqueeze(-1)    # [b,s,1]
                attn_mask = mask | ~attention_mask
                attn_mask = attn_mask.float().masked_fill(attn_mask, float('-inf')).unsqueeze(1)

            output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,
                                                    dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            mask = torch.triu(torch.ones(b, s, s), diagonal=1).bool()
            if attention_mask is not None:
                mask = mask.to(attention_mask.device)
                mask = mask | ~attention_mask.bool().unsqueeze(-1)
            mask = mask.float().masked_fill(mask, float('-inf')).unsqueeze(-1)

            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            scores += mask
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            scores = self.attention_dropout(scores)
            output = torch.matmul(scores, v)

        output = output.transpose(1, 2).contiguous().view(b, s, -1)
        z = z.sigmoid()
        output = output * z
        output = self.o_proj(output)
        return output

class DynamicCache:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv_states = [None for _ in range(config.n_layers)]
        self.recurrent_states = [None for _ in range(config.n_layers)]

        self.key_cache = [None for _ in range(config.n_layers)]
        self.value_cache = [None for _ in range(config.n_layers)]

class GatedDeltaNet(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2    # QKVZ
        projection_size_ba = self.num_v_heads * 2   # BA

        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx

        self.conv_dim = self.key_dim * 2 + self.value_dim   # QKV
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = RMSNorm(self.value_dim)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cache_params: DynamicCache=None, attention_mask: torch.Tensor=None):
        bs, s, _ = x.shape
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        projected_states_qkvz = self.in_proj_qkvz(x)
        projected_states_ba = self.in_proj_ba(x)

        query, key, value, z = torch.split(
            projected_states_qkvz,
            [self.key_dim, self.key_dim, self.value_dim, self.value_dim], dim=-1
        )

        b, a = torch.split(
            projected_states_ba,
            [self.num_v_heads, self.num_v_heads], dim=-1
        )

        mixed_qkv = torch.cat([query, key, value], dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)   # [b, k*2+v, s]

        if cache_params is not None:
            if s > 1:
                # 前填充
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :s])
            else:
                # 将当前token拼接到缓存末尾,将当前token拼接到缓存末尾,更新缓存
                conv_state = cache_params.conv_states[self.layer_idx]
                conv_state = torch.cat([conv_state, mixed_qkv], dim=-1)
                state_len = conv_state.shape[-1]

                mixed_qkv = F.silu(self.conv1d(conv_state)[:, :, state_len-1:state_len])
                cache_params.conv_states[self.layer_idx] = conv_state

        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :s])

        mixed_qkv = mixed_qkv.transpose(1,2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )

        query = query.reshape(bs, s, -1, self.head_k_dim)
        key = key.reshape(bs, s, -1, self.head_k_dim)
        value = value.reshape(bs, s, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

        query = query.transpose(1, 2).contiguous().to(torch.float32)
        key = key.transpose(1, 2).contiguous().to(torch.float32)
        value = value.transpose(1, 2).contiguous().to(torch.float32)
        beta = beta.transpose(1, 2).contiguous().to(torch.float32)
        g = g.transpose(1, 2).contiguous().to(torch.float32)

        _, num_heads, _, head_k_dim = key.shape
        head_v_dim = value.shape[-1]
        core_attn_out = torch.zeros(bs, num_heads, s, head_v_dim).to(value)

        recurrent_state = None
        if cache_params is not None:
            recurrent_state = cache_params.recurrent_states[self.layer_idx]
        last_recurrent_state = torch.zeros(bs, num_heads, head_k_dim, head_v_dim).to(value) \
            if recurrent_state is None else recurrent_state.to(value)

        for i in range(s):
            q_t = query[:, :, i]
            k_t = key[:, :, i]
            v_t = value[:, :, i]
            g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
            beta_t = beta[:, :, i].unsqueeze(-1)

            last_recurrent_state = last_recurrent_state * g_t  # S_{t-1} * alpha_t
            kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)  # S_{t-1} * alpha_t * k_t
            delta = beta_t * (v_t - kv_mem)  # beta_t * (v_t - S_{t-1} * alpha_t * k_t)
            last_recurrent_state += k_t.unsqueeze(-1) * delta.unsqueeze(-2) # S_{t-1} * alpha_t * (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
            core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

        core_attn_out = core_attn_out.transpose(1,2).contiguous().to(query.dtype)
        core_attn_out = core_attn_out.reshape(bs, s, -1)

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        z = F.silu(z)
        core_attn_out = self.norm(core_attn_out)
        core_attn_out = core_attn_out * z
        output = self.out_proj(core_attn_out)
        return output

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.up_proj(x)) * self.up_proj(x))

class Gating(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.topk = config.topk
        self.expert_num = config.expert_num
        self.gate = nn.Linear(self.hidden_size, self.expert_num)

    def forward(self, x):
        logits = self.gate(x) # [b,s,h] -> [b,s,expert_num]
        # 取topk专家的logits, 归一化
        logits_topk, indices = logits.topk(self.topk, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(dim=-1, index=indices, src=logits_topk)
        sparse_logtis = F.softmax(sparse_logits, dim=-1)
        gate_logit = logits.view(-1, self.expert_num)   # [b*s, expert_num]
        return sparse_logtis, indices, gate_logit

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.expert_num)])
        self.gating = Gating(config)

    def forward(self, x):
        sparse_logits, indices, gate_logit = self.gating(x)
        final_outputs = torch.zeros_like(x)
        x_flat = x.view(-1, x.shape[-1])  # [b*s, h]
        sparse_logits_flat = sparse_logits.view(-1, sparse_logits.shape[-1])    # [b*s, expert_num]

        for i, expert in enumerate(self.experts):
            expert_mask = (indices == i).any(-1)    # [b,s]
            expert_mask_flat = expert_mask.view(-1)  # [b*s]
            if expert_mask_flat.any():
                expert_input = x_flat[expert_mask_flat]  # [n, h], n means number of tokens routed to this expert
                expert_output = expert(expert_input)  # [n, h]
                gate_scores = sparse_logits_flat[expert_mask, i].unsqueeze(1) # [n] -> [n, 1]
                weighted_output = expert_output * gate_scores # [n, h]
                final_outputs[expert_mask] += weighted_output
        return final_outputs, gate_logit    # [b,s,h], [b*s, expert_num]

class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.moe = MoE(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.layer_idx = layer_idx
        self.softmax_attn_index = config.softmax_attn_index

        if layer_idx in self.softmax_attn_index:
            self.self_attn = Attention(config, layer_idx)
        else:
            self.linear_attn = GatedDeltaNet(config, layer_idx)

    def forward(self, x, attention_mask: torch.Tensor=None, cache_params: DynamicCache=None):
        residual = x
        x = self.input_layernorm(x)

        if self.layer_idx in self.softmax_attn_index:
            x = self.self_attn(x, attention_mask, cache_params)
        else:
            x = self.linear_attn(x, cache_params, attention_mask)

        x = x + residual
        residual = x
        x = self.post_attention_layernorm(x)

        # 交替使用MLP和MoE
        if self.layer_idx % 2 == 0:
            x = self.mlp(x)
            gate_logit = None
        else:
            x, gate_logit = self.moe(x)
        outputs = x + residual
        return outputs, gate_logit

class Config(PretrainedConfig):
    model_type = 'moe_model'
    def __init__(self,
                 hidden_size=256,
                 num_attention_heads=16,
                 num_key_value_heads=4,
                 linear_num_value_heads=16,
                 linear_num_key_heads=8,
                 linear_key_head_dim=16,
                 linear_value_head_dim=16,
                 linear_conv_kernel_dim=4,
                 flash_attn=True,
                 attention_bias=False,
                 max_seq_len=512,
                 intermediate_size=1024,
                 mlp_bias=False,
                 vocab_size=6400,
                 n_layers=8,
                 softmax_attn_index=[3, 7],
                 dropout=0.0,
                 expert_num=8,
                 topk=2,
                 output_router_logits=True,
                 aux_loss_coef=0.01,
                 **kwargs):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads          # q heads
        self.num_key_value_head_dim = num_key_value_heads       # kv heads

        self.linear_num_value_heads = linear_num_value_heads    # v heads
        self.linear_num_key_heads = linear_num_key_heads        # k heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim

        self.linear_conv_kernel_dim = linear_conv_kernel_dim

        self.flash_attn = flash_attn
        self.attention_bias = attention_bias
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.softmax_attn_index = softmax_attn_index
        self.dropout = dropout
        self.expert_num = expert_num
        self.topk = topk
        self.output_router_logits = output_router_logits
        self.aux_loss_coef = aux_loss_coef
        super().__init__(**kwargs)

def load_balancing_loss_func(gate_logits, num_experts, top_k):
    concatenated_logits = torch.cat([layer_gate for layer_gate in gate_logits], dim=0)  # [num_layers * b * s, expert_num]
    routing_weights = F.softmax(concatenated_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)  # [num_layers * b * s, top_k]

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)    # [total_tokens, top_k, expert_num]
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)  # 专家i被使用的比例
    router_prob_per_expert = torch.mean(routing_weights, dim=0) # 专家被分配的平均权重
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts

class LLM(PreTrainedModel):
    config_class = Config
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.expert_num = config.expert_num
        self.topk = config.topk

        self.token_emb = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([DecoderLayer(config, i) for i in range(self.n_layers)])
        self.norm = RMSNorm(self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.token_emb.weight = self.output.weight

        self.apply(self._init_weights)
        self.loss = None
        self.aux_loss = None

        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, labels=None, cache_params=None):
        all_router_logits = () if self.config.output_router_logits else None
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        for idx, layer in enumerate(self.layers):
            x, gate_logit = layer(x, attention_mask, cache_params)
            if gate_logit is not None:
                all_router_logits += (gate_logit,)
        x = self.norm(x)

        if labels is not None:
            logits = self.output(x)
            self.loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
        else:
            logits = self.output(x[:, [-1], :])
            self.loss = None

        if self.config.output_router_logits:
            self.aux_loss = load_balancing_loss_func(all_router_logits, self.expert_num, self.topk)
            if labels is not None:
                self.loss += self.config.aux_loss_coef * self.aux_loss
        return CausalLMOutputWithPast(self.loss, logits)

    @torch.inference_mode
    def generate(self, inputs, eos, max_new_tokens, temperature=0.7, top_k=None, stream=True, repetition_penalty=1., use_cache=True):
        cache_params = None
        if use_cache:
            cache_params = DynamicCache(self.config)
        input_ids = inputs['input_ids'] # [b, s]
        attention_mask = None
        labels = None
        s = input_ids.shape[1]

        idx_next = None
        while input_ids.shape[1] < max_new_tokens - 1:
            if idx_next is None:
                inference_res = self(input_ids, attention_mask, labels, cache_params)
            else:
                inference_res = self(idx_next, attention_mask, labels, cache_params)
            logits = inference_res.logits   # [b, 1, vocab_size]
            logits = logits[:, -1, :]
            # 降低已生成token的概率, 避免重复
            for token in set(input_ids.tolist()[0]):
                logits[:, token] /= repetition_penalty
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits /= temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # [b, top_k]
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                '''
                torch.argmax：总是选择概率最高的词（贪婪解码），生成文本可能单调、缺乏多样性
                torch.multinomial：根据概率随机选择，使生成的文本更自然、更丰富
                '''
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)
            if idx_next == eos:
                break

            input_ids = torch.cat([input_ids, idx_next], dim=-1)
            if stream:
                yield input_ids[:, s:]
        if not stream:
            yield input_ids[:, s:]

if __name__ == "__main__":
    config = Config()
    model = LLM(config)
    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer', use_fast=True)
    args = TrainingArguments(
        output_dir='./pretrain',
        num_train_epochs=1,
        do_train=True,
        per_device_train_batch_size=128,
        gradient_accumulation_steps=2,
        logging_steps=1,
        report_to='tensorboard',
        save_total_limit=5,
        save_steps=500,
        bf16=True,
        learning_rate=2e-4,
        lr_scheduler_type='cosine',
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        save_safetensors=False)
    data_pth = ''
    dataset = LLMDataset(data_pth, tokenizer=tokenizer, max_length=512)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, data_collator=data_collator)
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./pretrain/final_model')
    trainer.save_state()


