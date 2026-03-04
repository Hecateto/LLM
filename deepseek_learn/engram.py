from typing import List
from dataclasses import dataclass, field
import math

from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex

@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = ""
    engram_vocab_size: List[int] = field(default_factory=lambda: [151665*5, 151665*5])  # [2-gram vocab size, 3-gram vocab size]
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512    # embedding dimension for each n-gram
    n_head_per_ngram: int = 8   # 多头哈希降低碰撞概率
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])   # layers to apply engram module
    pad_id: int = 2 # tokenizer pad token id
    seed: int = 0
    kernel_size: int = 4    # kernel size for ShortConv


@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4            # Hyper-Connection 多头倍数（分组处理）
    vocab_size: int = 151665
    num_layers: int = 30


class CompressedTokenizer:
    """
    词表压缩器：通过文本归一化将语义相似的 token 映射到同一 ID，
    从而减小词表规模并提升 N-gram 哈希的效率。

    核心思想：
    - "Hello" 和 "hello" 归一化后映射到相同 ID
    - 减少稀疏性，提升 N-gram 统计的有效性
    """
    def __init__(self, tokenizer_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        SENTINEL = "\uE000"
        # 构建文本归一化流水线
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),  # Unicode 兼容性分解（全角→半角等）
            normalizers.NFD(),  # 字符分解（如 é → e + ́）
            normalizers.StripAccents(),  # 移除重音符号, 例如将 "é" 转换为 "e"
            normalizers.Lowercase(),  # 转小写（关键：合并大小写变体）
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),  # 合并空白字符
            normalizers.Replace(Regex(r"^ $"), SENTINEL),  # 临时标记纯空格
            normalizers.Strip(),  # 移除首尾空格
            normalizers.Replace(SENTINEL, " "),  # 恢复空格
        ])
        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def __len__(self):
        return self.num_new_token

    def _build_lookup_table(self):
        """
        构建原始 token_id → 压缩 token_id 的映射表

        算法流程：
        1. 遍历原始词表每个 token_id
        2. 解码为文本 → 归一化处理 → 生成归一化 key
        3. 相同 key 的 token 映射到同一新 ID
        4. 返回查找表和压缩后词表大小
        """
        old2new = {}
        key2new = {}
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)

    def _compress(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out

    def __call__(self, input_ids):
        return self._compress(input_ids)


class ShortConv(nn.Module):
    """
    短程卷积模块（Short-range Convolution）

    设计目标：
    - 在 Hyper-Connection 分组内提取局部时序特征
    - 使用深度可分离卷积（groups=channels）降低参数量
    - 配合 RMSNorm 和 SiLU 激活增强非线性表达能力
    """
    def __init__(
            self,
            hidden_size: int,
            kernel_size: int = 4,
            dilation: int = 1,
            norm_eps: float = 1e-5,
            hc_mult: int = 4,
            activation: bool = True
    ):
        super().__init__()
        self.hc_mult = hc_mult  # Hyper-Connection 分组数
        self.activation = activation

        total_channels = hidden_size * hc_mult  # 总通道数 = 隐藏维度 × 分组数
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )

        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps)
            for _ in range(hc_mult)
        ])

        if self.activation:
            self.act_fn = nn.SiLU()    # SiLU(x) = x * sigmoid(x), 平滑且非单调

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, G, C = x.shape # x: [batch_size, seq_len, hc_mult, hidden_size]
        assert G == self.hc_mult, f"输入的 hc_mult={G} 与模块配置 {self.hc_mult} 不匹配"

        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]  # [B, L, C]
            normed_chunks.append(self.norms[i](chunk))

        x_norm = torch.cat(normed_chunks, dim=-1) # [B, L, G*C]
        x_bct = x_norm.transpose(1, 2)  # [B, G*C, L]
        y_bct = self.conv(x_bct)  # 卷积, 输出长度 = L + 2*padding - (kernel-1)*dilation
        y_bct = y_bct[:, :, :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)

        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        return y


def find_next_prime(start, seen_primes) -> int:
    """
    工具函数：寻找大于 start 且未在 seen_primes 中出现的最小质数
    用途：为每个哈希头分配唯一的质数模数，确保哈希空间正交
    """
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    """
    N-gram 哈希映射器：将 token 序列转换为多尺度 N-gram 哈希索引

    核心创新：
    1. 词表压缩：减少哈希空间稀疏性
    2. 随机奇数乘子：混合不同位置的 token_id，增强哈希区分度
    3. 多头质数模哈希：每个 N-gram 使用多个质数模数，仅当所有头冲突时才视为碰撞
    4. 分层独立哈希：不同 Transformer 层使用独立随机种子，避免特征同质化
    """

    def __init__(
            self,
            engram_vocab_size,
            max_ngram_size,
            n_embed_per_ngram,
            n_head_per_ngram,
            layer_ids,
            tokenizer_name_or_path,
            pad_id,
            seed
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(tokenizer_name_or_path)
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)

        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        # 生成随机乘子
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers = {}     # # {layer_id: np.array[multipliers for n=2..max_ngram]}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)

            # 生成 max_ngram_size 个随机数 [0, half_bound)
            r = g.integers(low=0, high=half_bound, size=(self.max_ngram_size, ), dtype=np.int64)
            multipliers = r * 2 + 1
            '''
                若乘数为偶数, 二进制末尾为 0, token_id * 2 左移补0 会导致原始 token_id 的最低位信息丢失
                奇数乘法, token_id * odd 能保留原始值的最低位信息, 确保哈希函数是满射的, 减少信息熵损失
                另外, 在模 2^k 的整数环中, 奇数乘法是可逆的
            '''
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        """
        为每层、每个 N-gram、每个哈希头分配唯一的质数模数（虚拟词表大小）

        返回结构：
        {
            layer_id: [
                [head0_prime, head1_prime, ...],  # 2-gram 的各头质数
                [head0_prime, head1_prime, ...],  # 3-gram 的各头质数
                ...
            ]
        }
        """
        seen_primes = set()
        vocab_size_across_layers = {}

        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1

                # 为每个头寻找下一个可用质数
                for _ in range(num_head):
                    found_prime = find_next_prime(current_prime_search_start, seen_primes)
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime

                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

        return vocab_size_across_layers

    def _get_ngram_hashes(self, input_ids: np.ndarray, layer_id: int) -> np.ndarray:
        """
        核心哈希计算：为输入序列生成多尺度 N-gram 多头哈希索引

        Args:
            input_ids: np.ndarray, 形状 [B, T]，已压缩的 token_id
            layer_id: int, 当前 Transformer 层 ID
        Returns:
            np.ndarray, 形状 [B, T, num_heads_total], dtype=int64
            num_heads_total = (max_ngram_size-1) * n_head_per_ngram
        """
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape
        multipliers = self.layer_multipliers[layer_id]

        # 辅助函数：生成 k-step 移位序列（用于构建 N-gram）
        '''
        [10, 20, 30, 40]
        [2, 10, 20, 30]  # k=1
        [2, 2, 10, 20]   # k=2
        '''
        def shift_k(k: int) -> np.ndarray:
            if k == 0:
                return x
            # 左侧填充 pad_id，右侧截断，保持长度 T
            shifted = np.pad(x, ((0, 0), (k, 0)),
                             mode='constant', constant_values=self.pad_id[:, :T])
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]  # base_shifts[k] = x shifted by k
        all_hashes = [] # 存储所有 N-gram 所有头的哈希结果

        for n in range(2, self.max_ngram_size + 1): # n = 2, 3
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            '''
            n = 2
            tokens[0] = base_shifts[0][0] = 10
            tokens[1] = base_shifts[1][0] = 2
            '''

            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])

            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash)

        return np.stack(all_hashes, axis=2) # 堆叠所有头的结果: [B, T, num_heads_total]

    def hash(self, input_ids):
        """
        公共接口：对输入序列执行完整哈希流程

        Returns:
            dict: {layer_id: np.ndarray[B, T, H]}
        """

        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(
                input_ids=input_ids,
                layer_id=layer_id
            )
        return hash_ids_for_all_layers


class MultiHeadEmbedding(nn.Module):
    """
    多头嵌入层：将多个独立质数词表的哈希索引映射到共享嵌入空间

    设计动机：
    - 若为每个头单独创建 nn.Embedding，显存开销大且管理复杂
    - 方案：将所有头的词表拼接为一个大词表，通过 offset 实现逻辑隔离
    """
    def __init__(self, list_of_N: list[int], D: int):
        """
        Args:
            list_of_N: List[int], 每个哈希头的词表大小（质数）
            D: int, 每个头的嵌入维度（总维度 / 头数）
        """
        super().__init__()
        self.embedding_dim = D

        # 计算每个头在大词表中的起始偏移量
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        print('offsets:', offsets)

        # 总词表大小 = 所有头词表大小之和
        total_N = sum(list_of_N)
        print('总词表大小:', total_N)

        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: torch.Tensor, 形状 [B, T, num_heads]
                       每个元素是对应头的哈希索引（范围 [0, N_head)）
        Returns:
            output: torch.Tensor, 形状 [B, T, num_heads, D]
        """
        shifted_input_ids = input_ids + self.offsets    # [B,T,H] + [H] → [B,T,H]
        output = self.embedding(shifted_input_ids)  # [B,T,H,D]
        return output


class Engram(nn.Module):
    """
    Engram 模块：将 N-gram 哈希特征与主干隐藏状态融合

    工作流程：
    1. 输入 token_id → 压缩 → 多头 N-gram 哈希 → 哈希索引
    2. 哈希索引 → MultiHeadEmbedding → N-gram 嵌入向量
    3. N-gram 嵌入 → 投影为 Key/Value
    4. 主干隐藏状态 → 投影为 Query
    5. Q·K 相似度 → 门控系数 → 调制 Value
    6. 门控 Value + ShortConv(Value) → 输出残差
    """
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id

        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_config.engram_vocab_size,
            max_ngram_size=engram_config.max_ngram_size,
            n_embed_per_ngram=engram_config.n_embed_per_ngram,
            n_head_per_ngram=engram_config.n_head_per_ngram,
            layer_ids=engram_config.layer_ids,
            tokenizer_name_or_path=engram_config.tokenizer_name_or_path,
            pad_id=engram_config.pad_id,
            seed=engram_config.seed
        )

        self.multi_head_embedding = MultiHeadEmbedding(
            # 将嵌套的词汇表大小列表展平为一维列表
            list_of_N=[x for y in self.hash_mapping.vocab_size_across_layers[layer_id] for x in y],
            D=engram_config.n_embed_per_ngram // engram_config.n_head_per_ngram
        )

        self.short_conv = ShortConv(
            hidden_size=backbone_config.hidden_size,
            kernel_size=engram_config.kernel_size,
            dilation=engram_config.max_ngram_size,
            hc_mult=backbone_config.hc_mult
        )

        engram_hidden_size = (engram_config.max_ngram_size - 1) * engram_config.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, backbone_config.hidden_size)    # Engram 特征 → 主干隐藏空间

        self.key_projs = nn.ModuleList([
            nn.Linear(engram_hidden_size, backbone_config.hidden_size)
             for _ in range(backbone_config.hc_mult)
        ])

        # for K(h), Q
        self.norm1 = nn.ModuleList([
            nn.RMSNorm(backbone_config.hidden_size)
            for _ in range(backbone_config.hc_mult)
        ])
        self.norm2 = nn.ModuleList([
            nn.RMSNorm(backbone_config.hidden_size)
            for _ in range(backbone_config.hc_mult)
        ])

    def forward(self, hidden_states, input_ids):
        """
        Args:
            hidden_states: torch.Tensor, [B, L, HC_MULT, D]
                          主干网络的隐藏状态（Hyper-Connection 格式）
            input_ids: torch.Tensor, [B, L]
                      原始（压缩后）token_id 序列
        Returns:
            output: torch.Tensor, [B, L, HC_MULT, D]
        """
        # 获取当前层的哈希索引 [B, L, num_heads]
        hash_input_ids = torch.from_numpy(
            self.hash_mapping.hash(input_ids)[self.layer_id]
        )
        # 查表获取嵌入并展平多头维度
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=2)  # [B, L, num_heads, D_head] -> [B, L, H*D]

        gates = []
        for hc_idx in range(backbone_config.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)

            query = hidden_states[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)

            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()  # 保留符号, 压缩动态范围
            gate = gate.sigmoid().unsqueeze(-1) # # Sigmoid 映射到 (0,1) 并增加通道维 [B, L, 1]
            gates.append(gate)

        # 堆叠所有头的门控系数 [B, L, HC_MULT, 1]
        gates = torch.stack(gates, dim=2)

        value = gates * self.value_proj(embeddings).unsqueeze(2)    # [B, L, HC_MULT, D]
        output = value + self.short_conv(value)  # RC with ShortConv
        return output


class TransformerBlock(nn.Module):
    """
    集成 Engram 的 Transformer 块

    结构：
    [Input]
      → (可选) Engram 残差分支
      → (可选) Attention 残差分支
      → (可选) MoE 残差分支
      → [Output]
    """

    def __init__(self, layer_id):
        super().__init__()
        self.attn = lambda x: x
        self.moe = lambda x: x
        self.engram = None
        if layer_id in engram_config.layer_ids:
            self.engram = Engram(layer_id)

    def forward(self, input_ids, hidden_states):
        if self.engram is not None:
            hidden_states = self.engram(
                hidden_states=hidden_states,
                input_ids=input_ids
            ) + hidden_states

        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states


if __name__ == "__main__":

    # 初始化配置
    engram_config = EngramConfig()
    backbone_config = BackBoneConfig()

    # 压缩词表
    compressed_tokenizer = CompressedTokenizer(tokenizer_name_or_path=engram_config.tokenizer_name_or_path)
    print('压缩后的词表大小:', len(compressed_tokenizer))
    print('原始词表大小:', compressed_tokenizer.tokenizer.vocab_size)
    print('压缩率:', 1 - len(compressed_tokenizer) / compressed_tokenizer.tokenizer.vocab_size)

    input_ids = compressed_tokenizer.tokenizer.encode('hello world, Hello world')
    print('原始input_ids:', input_ids)
    compressedinput_ids = compressed_tokenizer(input_ids)
    print('压缩后的input_ids:', compressedinput_ids)

    # N-gram 哈希映射
    hash_mapping = NgramHashMapping(
        engram_vocab_size=engram_config.engram_vocab_size,
        max_ngram_size=engram_config.max_ngram_size,
        n_embed_per_ngram=engram_config.n_embed_per_ngram,
        n_head_per_ngram=engram_config.n_head_per_ngram,
        layer_ids=engram_config.layer_ids,
        tokenizer_name_or_path=engram_config.tokenizer_name_or_path,
        pad_id=engram_config.pad_id,
        seed=engram_config.seed)

    print('每层每个 N-gram 每个头的词表大小（质数）:')
    print(hash_mapping.vocab_size_across_layers)

    print('哈希映射:')
    input_ids = np.array([[101, 2000, 2022, 1037, 2204, 2154, 102]])
    hash_input_ids = hash_mapping.hash(input_ids)
    print('输入 input_ids:', input_ids)
    print('hash_input_ids', hash_input_ids)

    print('Engram 模块:')
    engram = Engram(layer_id=1)
    hidden_states = torch.randn(1, 6, 4, 1024)
    input_ids = torch.randint(0, 10000, (1, 6))
    output = engram(hidden_states, input_ids)
    print(output.shape)

    # LLM

    LLM = [
        nn.Embedding(backbone_config.vocab_size, backbone_config.hidden_size),  # 输入嵌入层
        *[TransformerBlock(layer_id=layer_id) for layer_id in range(backbone_config.num_layers)],
        nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size)      # LM Head
    ]

    text = "Only Alexander the Great could tame the horse Bucephalus."
    tokenizer = AutoTokenizer.from_pretrained(engram_config.tokenizer_name_or_path, trust_remote_code=True)
    input_ids = tokenizer(text, return_tensors='pt').input_ids

    B, L = input_ids.shape

    for idx, layer in enumerate(LLM):
        if idx == 0:
            hidden_states = LLM[0](input_ids)   # [B, L, D]
            # 扩展为 Hyper-Connection 格式：[B, L, 1, D] → [B, L, HC_MULT, D]
            hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, backbone_config.hc_mult, -1)
        elif idx == len(LLM) - 1:
            # [B, L, HC_MULT, D] → [B, L, D]
            hidden_states = hidden_states[:, :, 0, :]
            output = layer(hidden_states)
        else:
            hidden_states = layer(input_ids=input_ids, hidden_states=hidden_states)

    print("✅ Forward Complete!")
    print(f"{input_ids.shape=}\n{output.shape=}")


