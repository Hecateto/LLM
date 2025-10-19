import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, x, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(x))
        self.eps = eps

    def forward(self, x):
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    # cos: [seq_len, dim] -> [seq_len, dim, 1]
    # q,k: [batch, n_heads, seq_len, dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super(RotaryEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)) # [dim/2]
        t = torch.arange(max_seq_len).float().unsqueeze(1)  # [max_seq_len, 1]
        freqs = t @ inv_freq.unsqueeze(0)   # [max_seq_len, dim/2]
        freqs = torch.cat((freqs, freqs), dim=-1)   # [max_seq_len, dim]

        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def forward(self, q, k):
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0)  # [1, seq_len, dim]
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0)
        return apply_rotary_pos_emb(q, k, cos, sin)

class MLA(nn.Module):
    def __init__(self, dim, n_heads, q_lora_rank, kv_lora_rank,
                 qk_nope_head_dim, qk_rope_head_dim, v_head_dim,
                 max_seq_len, max_batch_size, mode):
        super().__init__()
        self.dim = dim  # 隐藏层维度
        self.n_heads = n_heads
        self.q_lora_rank = q_lora_rank  # Q压缩秩
        self.kv_lora_rank = kv_lora_rank    # KV压缩秩

        self.qk_nope_head_dim = qk_nope_head_dim    # QK非旋转维度
        self.qk_rope_head_dim = qk_rope_head_dim    # QK旋转维度
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim  # QK头维度

        self.v_head_dim = v_head_dim    # 等于QK非旋转维度
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.mode = mode

        self.wq_a = nn.Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim)

        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.wk_b = nn.Linear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim)
        self.rotary_emb = RotaryEmbedding(self.qk_rope_head_dim)

        if self.mode == 'naive':
            self.register_buffer('k_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.n_heads, self.qk_head_dim), persistent=False)
            self.register_buffer('v_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.n_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer('kv_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer('pe_cache', torch.zeros(self.max_batch_size, self.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x, mask=None):
        bs, seq_len, _ = x.shape    # [bs, seq_len, dim]
        # q降维
        q = self.wq_a(x)    # [bs, seq_len, q_lora_rank]
        # q归一化
        q = self.q_norm(q)
        # q升维
        q = self.wq_b(q) # [bs, seq_len, n_heads * qk_head_dim]
        q = q.view(bs, seq_len, self.n_heads, self.qk_head_dim) # [bs, seq_len, n_heads, qk_head_dim]
        # 拆分q为非旋转和旋转部分
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # kv降维
        kv = self.wkv_a(x)
        # 拆分kv为 低维 和 k的pe部分
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.unsqueeze(2)   # [bs, seq_len, 1, qk_rope_head_dim]
        q_pe, k_pe = self.rotary_emb(q_pe, k_pe)
        # q_pe: [bs, seq_len, n_heads, qk_rope_head_dim]
        # k_pe: [bs, seq_len, 1, qk_rope_head_dim]

        if self.mode == 'naive':
            # q完整
            q = torch.cat([q_nope, q_pe], dim=-1)   # [bs, seq_len, n_heads, qk_head_dim]

            # k完整
            kv = self.kv_norm(kv)   # [bs, seq_len, kv_lora_rank]
            kv = self.wk_b(kv) # [bs, seq_len, n_heads * (qk_nope_head_dim + v_head_dim)]
            kv = kv.view(bs, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = torch.cat([k_nope, k_pe], dim=-1)   # [bs, seq_len, n_heads, qk_head_dim]
            # cache
            self.k_cache[:, :seq_len, :, :] = k
            self.v_cache[:, :seq_len, :, :] = v

            # q: [bs, seq_len, n_heads, qk_head_dim] -> [bs, n_heads, seq_len, qk_head_dim]
            # k转置: [bs, seq_len, n_heads, qk_head_dim] -> [bs, n_heads, qk_head_dim, seq_len]
            # scores: [bs, n_heads, seq_len, seq_len]
            scores = torch.matmul(q.transpose(1,2),
                                  self.k_cache[:bs, :seq_len, :, :].transpose(1,2).transpose(2,3)
                                  / math.sqrt(self.qk_head_dim))  # [bs, n_heads, seq_len, seq_len]
            scores = scores.transpose(1,2)  # [bs, seq_len, n_heads, seq_len]

        else:
            # 去掉临时头维度
            k_pe = k_pe.squeeze(2)  # [bs, seq_len, qk_rope_head_dim]
            # kv升维矩阵的权重
            wkv_b = self.wk_b.weight    # [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
            wkv_b = wkv_b.view(self.n_heads, -1, self.kv_lora_rank) # [n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank]

            # q * k(T) = q * (c * wkv_b[:, :qk_nope_head_dim, :](T))(T)
            #          = q * wkv_b[:, :qk_nope_head_dim, :] * c(T)
            #          = (q * wkv_b[:, :qk_nope_head_dim, :]) * c(T)
            #          = q_latent * c(T)
            # q_latent是对q_nope的降维
            q_nope = torch.einsum('bshd,hdc->bshc',
                                  q_nope,
                                  wkv_b[:, :self.qk_nope_head_dim, :])  # [bs, seq_len, n_heads, kv_lora_rank]
            kv = self.kv_norm(kv)

            self.kv_cache[:bs, :seq_len, :] = kv  # [bs, seq_len, kv_lora_rank]
            self.pe_cache[:bs, :seq_len, :] = k_pe  # [bs, seq_len, qk_rope_head_dim]

            scores_nope = torch.einsum('bshc, btc->bsht', q_nope, self.kv_cache[:bs, :seq_len, :])
            scores_pe = torch.einsum('bshr, btr->bsht', q_pe, self.pe_cache[:bs, :seq_len, :])
            scores = (scores_nope + scores_pe) / math.sqrt(self.qk_head_dim)

        if mask is not None:    # mask: [bs, seq_len, seq_len]
            scores += mask.unsqueeze(2)    # [bs, seq_len, 1, seq_len]

        scores = F.softmax(scores, dim=-1)

        # v_cache: [bs, seq_len, n_heads, v_head_dim]
        # wkv_b: [n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank]
        # x: [bs, seq_len, n_heads, v_head_dim]
        if self.mode == 'naive':
            x = torch.einsum('bsht, bthd->bshd', scores, self.v_cache[:bs, :seq_len])
        else:
            # scores * v = scores * c * wkv_b[:, -self.v_head_dim:]
            x = torch.einsum('bsht, btc->bshc', scores, self.v_cache[:bs, :seq_len])    # 注意力权重 * KV低维缓存
            x = torch.einsum('bshc,hdc->bshd', x, wkv_b[:, -self.v_head_dim:])  # 升维

        x = x.contiguous().view(bs, seq_len, -1)
        x = self.wo(x)
        return x


if __name__ == '__main__':
    x = torch.randn(4, 100, 4096)
    dim = 4096
    n_heads = 16
    q_lora_rank = 128
    kv_lora_rank = 64
    qk_nope_head_dim = 256
    qk_rope_head_dim = 48
    v_head_dim = 256
    max_seq_len = 512
    max_batch_size = 16
    mode = 'none'

    mla = MLA(
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mode=mode
    )
    print(mla(x))
    print(mla.kv_cache)