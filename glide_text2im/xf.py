"""
Transformer implementation adapted from CLIP ViT:
https://github.com/openai/CLIP/blob/4c0275784d6d9da97ca1f47eaaee31de1867da91/clip/model.py
"""

import math

import torch as th
import torch.nn as nn


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


class LayerNorm(nn.LayerNorm):
    """
    支持混合精度的LayerNorm实现。

    继承自PyTorch的LayerNorm，但支持fp16输入和fp32增益/偏置参数。
    这种设计在混合精度训练中很常见，可以保持数值稳定性。

    特性：
    - 输入：可以是fp16/fp32
    - 内部计算：使用fp32确保数值精度
    - 输出：恢复到输入的数据类型

    Args:
        width: 输入特征维度（继承自父类）
        eps: 防止除零的小常数（默认1e-5）
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def forward(self, x: th.Tensor):
        # 将输入转换为fp32进行计算，确保数值稳定性
        # 计算完成后转换回原始数据类型
        return super().forward(x.float()).to(x.dtype)


class MultiheadAttention(nn.Module):
    """
    标准的自注意力机制实现，使用QKV线性投影。

    将输入通过线性层投影为Q、K、V，然后计算注意力权重，
    最后通过输出投影层得到结果。

    Args:
        n_ctx: 序列长度/上下文窗口大小
        width: 输入特征维度
        heads: 多头注意力的头数
    """
    def __init__(self, n_ctx, width, heads):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        # QKV投影层：将输入投影为查询(Query)、键(Key)、值(Value)
        # 输出维度为width*3，因为Q、K、V各占width维度
        self.c_qkv = nn.Linear(width, width * 3)
        # 输出投影层：将多头注意力结果合并回原始维度
        self.c_proj = nn.Linear(width, width)
        # 注意力计算核心模块
        self.attention = QKVMultiheadAttention(heads, n_ctx)

    def forward(self, x):
        """
        前向传播过程：
        输入 -> QKV投影 -> 注意力计算 -> 输出投影
        """
        # 1. 计算Q、K、V（合并计算以提高效率）
        x = self.c_qkv(x) # [batch, seq_len, width*3]
        # 2. 应用多头注意力机制
        x = self.attention(x) # [batch, seq_len, width]
        # 3. 输出投影（可选，用于调整输出维度）
        x = self.c_proj(x) # [batch, seq_len, width]
        return x


class MLP(nn.Module):
    """
    前馈神经网络（Feed-Forward Network）模块。

    Transformer中的标准前馈网络，包含两个线性层和激活函数。
    通常用于每个注意力层之后，增加模型的非线性表达能力。

    Args:
        width: 输入输出特征维度
    """
    def __init__(self, width):
        super().__init__()
        self.width = width
        # 第一个线性层：扩展维度（通常扩展4倍）
        self.c_fc = nn.Linear(width, width * 4)
        # 第二个线性层：压缩回原始维度
        self.c_proj = nn.Linear(width * 4, width)
        # GELU激活函数：比ReLU更平滑的激活函数
        self.gelu = nn.GELU()

    def forward(self, x):
        """
        前向传播：线性扩展 -> 激活 -> 线性压缩
        """
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    """
    核心的多头注意力计算模块。

    接收合并的QKV张量，进行注意力权重的计算和加权求和。
    使用爱因斯坦求和约定实现高效的矩阵运算。

    Args:
        n_heads: 注意力头数
        n_ctx: 序列长度（用于注意力掩码）
    """
    def __init__(self, n_heads: int, n_ctx: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        """
        计算多头注意力。

        Args:
            qkv: 合并的QKV张量，形状为 [batch, seq_len, width*3]

        Returns:
            注意力加权后的值张量，形状为 [batch, seq_len, width]
        """
        # 获取输入形状
        bs, n_ctx, width = qkv.shape # [batch_size, seq_len, hidden_dim*3]
        # 计算每个注意力头的通道数
        attn_ch = width // self.n_heads // 3 # 每个头每个Q/K/V的维度
        # 缩放因子：1/sqrt(d_k)，用于稳定注意力分数
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        # 重塑张量：分离出注意力头
        # [batch, seq_len, heads, 3*head_dim] -> 便于分割QKV
        qkv = qkv.view(bs, n_ctx, self.n_heads, -1)
        # 分割Q、K、V（每个占head_dim维度）
        # q, k, v 形状: [batch, seq_len, heads, head_dim]
        q, k, v = th.split(qkv, attn_ch, dim=-1)
        # 计算注意力权重：Q * K^T
        # 使用爱因斯坦求和约定进行高效的矩阵乘法
        # weight形状: [batch, heads, target序列长度, 源序列长度]
        # 使用缩放后的点积比后缩放更稳定（特别是fp16）
        weight = th.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        # 保存原始数据类型（用于混合精度）
        wdtype = weight.dtype
        # 应用softmax得到注意力权重
        # 在fp32中计算softmax确保数值稳定性
        weight = th.softmax(weight.float(), dim=-1).type(wdtype)
        # 加权求和：注意力权重 * 值向量
        # result形状: [batch, seq_len, heads, head_dim]
        # 重塑回原始形状
        return th.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1) # [batch, seq_len, width]


class ResidualAttentionBlock(nn.Module):
    """
    Transformer的残差注意力块，包含自注意力和前馈网络。

    这是Transformer编码器的基本构建块，包含：
    1. 层归一化 + 多头自注意力 + 残差连接
    2. 层归一化 + 前馈网络 + 残差连接

    Args:
        n_ctx: 序列长度
        width: 特征维度
        heads: 注意力头数
    """
    def __init__(
        self,
        n_ctx: int,
        width: int,
        heads: int,
    ):
        super().__init__()

        # 自注意力子层
        self.attn = MultiheadAttention(
            n_ctx,
            width,
            heads,
        )
        # 第一个层归一化（注意力子层前）
        self.ln_1 = LayerNorm(width)
        # 前馈网络子层
        self.mlp = MLP(width)
        # 第二个层归一化（前馈子层前）
        self.ln_2 = LayerNorm(width)

    def forward(self, x: th.Tensor):
        """
        前向传播：残差连接的两个子层。

        Args:
            x: 输入张量 [batch, seq_len, hidden_dim]

        Returns:
            输出张量 [batch, seq_len, hidden_dim]
        """
        # 子层1: 层归一化 -> 自注意力 -> 残差连接
        # 公式: x = x + Attention(LayerNorm(x))
        x = x + self.attn(self.ln_1(x))
        # 子层2: 层归一化 -> 前馈网络 -> 残差连接
        # 公式: x = x + FeedForward(LayerNorm(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    Transformer编码器，用于处理文本序列编码。

    这是CLIP/GLIDE/DALL·E等模型中使用的标准Transformer编码器结构。
    不包含位置编码（位置编码在外部添加），仅包含多层Transformer块。

    功能：将输入序列转换为上下文感知的表示，捕捉序列内部的依赖关系。

    Args:
        n_ctx: 文本上下文长度（最大序列长度），如128、256
        width: Transformer隐藏层的维度（特征维度），如512、768
        layers: Transformer层数（深度），如12、24
        heads: 多头注意力机制的头数，如8、12

    输入输出形状:
        输入: [batch_size, sequence_length, hidden_dim]
        输出: [batch_size, sequence_length, hidden_dim]

    典型配置示例：
        - CLIP文本编码器: n_ctx=77, width=512, layers=12, heads=8
        - GPT-2小型: width=768, layers=12, heads=12
        - BERT基础: width=768, layers=12, heads=12
    """
    def __init__(
        self,
        n_ctx: int, # 序列长度/上下文窗口大小
        width: int, # 隐藏层维度（特征维度）
        layers: int, # Transformer层数（深度）
        heads: int, # 多头注意力头数
    ):
        super().__init__() # 调用父类nn.Module的初始化
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        # 创建Transformer层堆叠
        # nn.ModuleList: 用于管理多个子模块的容器
        self.resblocks = nn.ModuleList(
            [
                # 创建多个ResidualAttentionBlock实例
                # 每个块包含：多头注意力 + 前馈网络 + 残差连接 + 层归一化
                ResidualAttentionBlock(
                    n_ctx,
                    width,
                    heads,
                )
                for _ in range(layers) # 创建layers个块
            ]
        )

    def forward(self, x: th.Tensor):
        """
        前向传播：将输入序列通过多层Transformer块进行处理。

        Args:
            x: 输入张量，形状为 [batch_size, sequence_length, hidden_dim]
               其中：
               - batch_size: 批次大小
               - sequence_length: 序列长度，应等于n_ctx
               - hidden_dim: 特征维度，应等于width

        Returns:
            th.Tensor: 输出张量，形状同输入 [batch_size, sequence_length, hidden_dim]

        处理流程：
            输入 → [Transformer块1] → [Transformer块2] → ... → [Transformer块N] → 输出
            每个块：多头注意力 + 前馈网络 + 残差连接 + 层归一化
        """
        # 逐层处理：将输入依次通过每个Transformer块
        for block in self.resblocks:
            # 当前块的输出作为下一个块的输入
            # 每个ResidualAttentionBlock内部包含：
            # 1. 层归一化 (LayerNorm)
            # 2. 多头自注意力 (Multi-Head Self-Attention)
            # 3. 残差连接 (Add)
            # 4. 层归一化 (LayerNorm)
            # 5. 前馈网络 (Feed-Forward Network, MLP)
            # 6. 残差连接 (Add)
            x = block(x)
        # 返回最后一层的输出
        return x
