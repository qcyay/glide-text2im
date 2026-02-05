"""
Various utilities for neural networks.
"""

import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, swish, eps=1e-5):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps)
        self.swish = swish

    def forward(self, x):
        y = super().forward(x.float()).to(x.dtype)
        if self.swish == 1.0:
            y = F.silu(y)
        elif self.swish:
            y = y * F.sigmoid(y * float(self.swish))
        return y


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def normalization(channels, swish=0.0):
    """
    Make a standard normalization layer, with an optional swish activation.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(num_channels=channels, num_groups=32, swish=swish)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    生成正弦时间步嵌入（Sinusoidal Timestep Embeddings）。

    这是扩散模型（如DDPM、Stable Diffusion）中的标准时间步编码方法，
    用于将离散/连续的时间步编码为连续的向量表示，使模型能够感知扩散过程的时间。

    原理：使用不同频率的正弦和余弦函数编码时间步，类似Transformer的位置编码。

    Args:
        timesteps: 1-D张量，形状为 [N]，N是批次大小
                  包含每个样本的时间步索引，可以是整数值或小数值
                  在DDPM中通常范围是 [0, T-1]，T是总扩散步数
        dim: 输出嵌入的维度，需要是偶数（如果是奇数会补零）
        max_period: 控制嵌入的最小频率，默认10000
                   更大的max_period会产生更低频率的正弦波

    Returns:
        时间步嵌入张量，形状为 [N, dim]

    数学公式：
        embedding(t, 2i) = sin(t / 10000^(2i/dim))
        embedding(t, 2i+1) = cos(t / 10000^(2i/dim))

    示例：
        >>> timesteps = th.tensor([0, 500, 1000])
        >>> emb = timestep_embedding(timesteps, dim=128)
        >>> emb.shape
        torch.Size([3, 128])  # 3个时间步，每个128维

    应用场景：
        1. 扩散模型：告知模型当前是扩散过程的哪一步
        2. 时间序列：编码序列中的时间位置
        3. 相对位置编码：通过正弦编码的相对位置
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    # 1. 计算正弦/余弦对的数量
    # 每个维度一对（正弦+余弦），所以需要一半的维度
    half = dim // 2
    # 2. 生成频率序列
    # 计算指数衰减的频率：从高频到低频
    # 等价于：freqs = 1.0 / (max_period ** (th.arange(half) / half))
    # 形状：[half]，例如：[1/10000^(0/64), 1/10000^(1/64), ...]
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    # 3. 计算角度参数
    # timesteps[:, None]: 形状 [N, 1]，添加新维度
    # freqs[None]: 形状 [1, half]，添加新维度
    # 广播相乘：结果形状 [N, half]
    # 每个时间步乘以对应的频率系数
    # args[i, j] = timesteps[i] * 1/(max_period^(j/half))
    args = timesteps[:, None].float() * freqs[None]
    # 4. 计算正弦和余弦嵌入
    # 前一半维度：余弦，后一半维度：正弦
    # 形状从 [N, half] + [N, half] -> [N, 2*half] = [N, dim]（如果dim是偶数）
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    # 5. 处理奇数维度
    if dim % 2:
        # 如果dim是奇数，添加一个零填充的维度
        # 例如：dim=129时，half=64，需要额外添加1维
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding