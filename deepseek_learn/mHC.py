import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import MultiheadAttention


# Sinkhorn-Knopp 算法实现
def sinkhorn_knopp(matrix: torch.Tensor, num_iter: int=20, epsilon: float=1e-20) -> torch.Tensor:
    """
     Sinkhorn-Knopp 迭代算法，用于将任意非负矩阵转换为双随机矩阵（Doubly Stochastic Matrix）。

    性质：
    1. 所有元素非负。
    2. 每一行的和为 1。
    3. 每一列的和为 1。

    用途：
    在 mHC 中，用于生成分支间的混合矩阵 H_res。双随机性质确保信息在分支间交换时
    总 magnitude 保持稳定，避免某些分支被过度抑制或增强，利于训练稳定性。

    Args:
        matrix: 输入张量 [B, L, n, n]，待归一化的混合矩阵
        num_iter: 迭代次数，通常 20 次足以收敛
        epsilon: 数值稳定性常数，防止除零

    Returns:
        双随机矩阵 [B, L, n, n]
    """
    K = torch.exp(matrix)
    for _ in range(num_iter):
        # 行列归一化, 使和为1
        K = K / (K.sum(dim=-1, keepdim=True) + epsilon)
        K = K / (K.sum(dim=-2, keepdim=True) + epsilon)
    return K


class mHC(nn.Module):
    """
    Manifold-Constrained Hyper-Connections, 流形约束超连接
    """
    def __init__(self, dim, n, layer_id):
        super(mHC, self).__init__()
        self.dim = dim  # 每个分支的特征维度 D
        self.n = n  # 分支数量 n
        self.nc = n * dim  # 展平后的总通道数 (n*D)
        self.n2 = n * n  # 混合矩阵的元素总数 (n*n)

        '''
        [:n]   -> H_pre  (n)
        [n:2n] -> H_post (n)
        [2n:]  -> H_res  (n*n)
        '''
        self.phi = nn.Linear(self.nc, self.n2 + 2 * self.n, bias=False)
        self.a = nn.Parameter(torch.ones(3)*0.01)
        self.b = nn.Parameter(torch.zeros(self.n2 + 2 * self.n))

    def width_connection(self, hidden_states):
        """
        宽度连接：处理同一层内 n 个分支之间的交互。
        1. 聚合：n 分支 -> 1 分支 (供 Attention/FFN 使用)
        2. 混合：n 分支 -> n 分支 (作为残差路径)

        Args:
            hidden_states: [B, L, n, dim] 输入的多分支隐藏状态

        Returns:
            h_pre:   [B, L, 1, dim] 聚合后的单分支表示
            h_res:   [B, L, n, dim] 混合后的多分支残差
            H_post:  [B, L, n]      用于深度连接的分发权重
        """
        B, L, N, D = hidden_states.shape
        assert N == self.n, f"Expected n={self.n} branches, but got {N}"

        hidden_states_flatten = hidden_states.flatten(2)  # [B, L, n*D]
        r = hidden_states_flatten.norm(dim=-1, keepdim=True) / math.sqrt(self.nc) # [B, L, 1]
        H = self.phi(hidden_states_flatten)  # [B, L, n*n + 2*n]

        # 分割H为三部分, 使用 (1/r) 进行反归一化
        H_pre = (1/r) * H[:, :, :self.n] * self.a[0] + self.b[0:self.n]  # [B, L, n]
        H_post = (1/r) * H[:, :, self.n:self.n*2] * self.a[1] + self.b[self.n:self.n*2]  # [B, L, n], 乘2增大初始动态范围
        H_res = (1/r) * H[:, :, self.n*2:] * self.a[2] + self.b[self.n*2:]  # [B, L, n*n]

        H_pre = F.sigmoid(H_pre)
        H_post = 2 * F.sigmoid(H_post)
        H_res = H_res.reshape(B, L, self.n, self.n)  # [B, L, n, n]
        H_res = sinkhorn_knopp(H_res)

        H_pre = H_pre.unsqueeze(dim=2) # [B, L, 1, n]
        h_pre = torch.matmul(H_pre, hidden_states)  # [B, L, 1, dim] @ [B, L, n, dim] -> [B, L, 1, dim]

        h_res = torch.matmul(H_res, hidden_states)  # [B, L, n, n] @ [B, L, n, dim] -> [B, L, n, dim]

        return h_pre, h_res, H_post

    def depth_connection(self, h_res, hidden_states, H_post):
        """
        深度连接：处理层间信息传递，将处理后的单分支信号分发回多分支，并与残差融合。

        Args:
            h_res:         [B, L, n, dim] 来自 width_connection 的混合残差
            hidden_states: [B, L, dim]    来自 Attention 或 FFN 的处理后单分支信号
            H_post:        [B, L, n]      来自 width_connection 的分发权重

        Returns:
            output: [B, L, n, dim] 融合后的多分支输出
        """
        # H_post: [B, L, n] -> [B, L, n, 1]
        # hidden_states: [B, L, dim] -> [B, L, 1, dim]
        # h_post: [B, L, n, dim]
        h_post = torch.matmul(H_post.unsqueeze(dim=-1), hidden_states.unsqueeze(dim=-2))

        output = h_post + h_res
        return output


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(FFN, self).__init__()
        self.proj_up = nn.Linear(dim, hidden_dim)
        self.proj_down = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x = F.gelu(self.proj_up(x))
        x = self.proj_down(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, layer_id, n=4):
        super(DecoderLayer, self).__init__()
        self.attn_mhc = mHC(dim=dim, n=n, layer_id=layer_id)
        self.ffn_mhc = mHC(dim=dim, n=n, layer_id=layer_id)

        self.attention = MultiheadAttention(embed_dim=dim, num_heads=num_heads, bias=False, batch_first=True)
        self.ffn = FFN(dim=dim, hidden_dim=4*dim)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [B, L, n, dim] 输入的多分支隐藏状态
        Returns:
            hidden_states: [B, L, n, dim] 输出的多分支隐藏状态
        """
        # Attention 层
        # 宽度连接
        h_pre, h_res, H_post = self.attn_mhc.width_connection(hidden_states)
        # Attn 计算
        h_pre = h_pre.squeeze(dim=2)  # [B, L, 1, dim] -> [B, L, dim]
        attn_output, _ = self.attention(h_pre, h_pre, h_pre)
        # 深度连接
        hidden_states = self.attn_mhc.depth_connection(h_res, attn_output, H_post)  # [B, L, n, dim]

        # FFN 层
        h_pre, h_res, H_post = self.ffn_mhc.width_connection(hidden_states)
        ffn_output = self.ffn(h_pre.squeeze(dim=2))  # [B, L, dim]
        hidden_states = self.ffn_mhc.depth_connection(h_res, ffn_output, H_post)  # [B, L, n, dim]

        return hidden_states


class LLM(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, n=4):
        super(LLM, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            DecoderLayer(dim=dim, num_heads=num_heads, layer_id=i, n=n)
            for i in range(num_layers)
        ])
        self.output_layer = nn.Linear(dim, vocab_size)
        self.n = n

    def forward(self, input_ids):
        """
        Args:
            input_ids: [B, L] 输入的 token ids
        Returns:
            logits: [B, L, vocab_size] 输出的 token logits
        """
        hidden_states = self.word_embedding(input_ids)  # [B, L, dim]
        hidden_states = hidden_states.unsqueeze(dim=2).expand(-1, -1, self.n, -1)  # [B, L, n, dim]

        for layer in self.layers:
            hidden_states = layer(hidden_states)

        output = self.output_layer(hidden_states.mean(dim=2))   # 均值池化将所有分支的信息合并
        return output

if __name__ == "__main__":
    # 实例化模型
    model = LLM(vocab_size=5000, dim=64, num_heads=4, num_layers=2, n=4)

    # 构造随机输入
    input_ids = torch.randint(0, 5000, (1, 10))  # (B=1, L=10)

    # 前向传播
    output = model(input_ids)

    # 验证输出形状
    print(output.shape)  # 期望：torch.Size([1, 10, 5000])
