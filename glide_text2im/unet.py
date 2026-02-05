import math
from abc import abstractmethod

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from glide_text2im.fp16_util import convert_module_to_f16, convert_module_to_f32
from glide_text2im.nn import avg_pool_nd, conv_nd, linear, normalization, timestep_embedding, zero_module


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    支持时间步嵌入的顺序容器模块。

    继承自 nn.Sequential 和 TimestepBlock，允许将时间步嵌入作为额外输入
    传递给支持它的子模块。这是扩散模型中连接不同层类型的关键组件。

    功能：依次调用每个子层，根据子层的类型传递不同的参数：
    1. 如果子层是 TimestepBlock: 传递 (x, emb) - 时间步条件
    2. 如果子层是 AttentionBlock: 传递 (x, encoder_out) - 注意力条件
    3. 否则: 只传递 x - 普通层

    继承结构：
        nn.Sequential              ← PyTorch的顺序容器
        TimestepBlock              ← 时间步块接口
        TimestepEmbedSequential    ← 我们的自定义容器

    在UNet中的应用：
        用于将 ResBlock、AttentionBlock 等组合在一起，
        并根据它们的类型智能地传递参数。
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, encoder_out=None):
        """
        前向传播：依次处理每个子层，根据类型传递参数。

        Args:
            x: 输入张量，通常形状为 [batch, channels, height, width]
            emb: 时间步嵌入张量，形状为 [batch, embed_dim]
                从 timestep_embedding 生成，用于条件化
            encoder_out: 编码器输出（可选），形状为 [batch, channels, seq_len]
                用于交叉注意力，如文本条件

        Returns:
            处理后的张量，形状与输入 x 相同

        处理逻辑：
            for layer in self.children():
                if layer 是 TimestepBlock: layer(x, emb)      # 时间步条件
                elif layer 是 AttentionBlock: layer(x, encoder_out)  # 交叉注意力
                else: layer(x)                                # 普通层

        示例：
            # 定义
            block = TimestepEmbedSequential(
                ResBlock(channels, emb_dim),      # TimestepBlock
                AttentionBlock(channels),         # AttentionBlock
                nn.Conv2d(channels, channels, 3)  # 普通层
            )

            # 前向
            output = block(x, emb, encoder_out)
        """
        # 遍历容器中的所有子层
        for layer in self:
            # 1. 如果是TimestepBlock（如ResBlock），传递时间步嵌入
            if isinstance(layer, TimestepBlock):
                # 时间步条件层：需要知道当前扩散时间步
                x = layer(x, emb)
            # 2. 如果是AttentionBlock，传递编码器输出（用于交叉注意力）
            elif isinstance(layer, AttentionBlock):
                # 注意力层：需要条件信息（如文本编码）
                x = layer(x, encoder_out)
            # 3. 否则是普通层（如卷积、池化、激活等），只传递输入
            else:
                # 普通层：不需要额外条件
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    上采样层，可选是否包含卷积操作。

    支持最近邻插值上采样，可选后接3x3卷积用于特征平滑和通道调整。

    Args:
        channels: 输入通道数
        use_conv: 是否在采样后应用卷积
        dims: 信号维度（1D、2D或3D）
              如果是3D，上采样在内层两个维度（H和W）进行
        out_channels: 输出通道数（如果为None则与输入相同）
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        # 存储参数
        self.channels = channels # 输入通道数
        self.out_channels = out_channels or channels # 输出通道数（默认同输入）
        self.use_conv = use_conv # 是否使用卷积
        self.dims = dims # 维度（1D/2D/3D）
        # 如果启用卷积，创建3x3卷积层
        if use_conv:
            # 卷积核大小3x3，padding=1保持空间尺寸
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        """
        前向传播：对输入进行2倍上采样。

        Args:
            x: 输入张量，形状取决于维度：
               2D: [batch, channels, height, width]
               3D: [batch, channels, depth, height, width]

        Returns:
            上采样后的张量，空间维度扩大2倍
        """
        # 验证输入通道数匹配
        assert x.shape[1] == self.channels
        # 1. 上采样操作
        if self.dims == 3:
            # 3D情况：只在内层两个维度（高度和宽度）上采样
            # 保持深度维度不变
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest") # 最近邻插值，计算简单，保持边缘清晰
        else:
            # 1D/2D情况：所有空间维度2倍上采样
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        # 2. 可选卷积操作
        if self.use_conv:
            x = self.conv(x) # 3x3卷积，可调整通道数和平滑特征
        return x


class Downsample(nn.Module):
    """
    下采样层，可选使用卷积或平均池化。

    支持2倍下采样，可以通过带步长的卷积或平均池化实现。

    Args:
        channels: 输入通道数
        use_conv: 是否使用卷积进行下采样（True=带步长卷积，False=平均池化）
        dims: 信号维度（1D、2D或3D）
              如果是3D，下采样在内层两个维度（H和W）进行
        out_channels: 输出通道数（如果为None则与输入相同）
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        # 存储参数
        self.channels = channels # 输入通道数
        self.out_channels = out_channels or channels # 输出通道数（默认同输入）
        self.use_conv = use_conv # 是否使用卷积
        self.dims = dims # 维度
        # 计算步长：3D情况下只在H和W维度下采样，D维度保持
        stride = 2 if dims != 3 else (1, 2, 2) # 3D: (D_stride=1, H_stride=2, W_stride=2)
        if use_conv:
            # 使用带步长的卷积实现下采样
            # 卷积核3x3，步长=stride，padding=1
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        else:
            # 使用平均池化实现下采样
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        """
        前向传播：对输入进行2倍下采样。

        Args:
            x: 输入张量

        Returns:
            下采样后的张量，空间维度减半
        """
        # 验证输入通道数匹配
        assert x.shape[1] == self.channels
        # 应用下采样操作（卷积或池化）
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    带有时间步条件化的残差块，支持通道数变化和上/下采样。

    这是扩散模型（如DDPM、Stable Diffusion）中的核心构建块，结合了：
    1. 残差连接（ResNet风格）
    2. 时间步条件化（扩散过程感知）
    3. 可选的上/下采样
    4. 尺度-偏移归一化（FiLM-like条件）

    Args:
        channels: 输入通道数
        emb_channels: 时间步嵌入的通道数
        dropout: dropout率
        out_channels: 输出通道数（如果为None则与输入相同）
        use_conv: 如果为True且out_channels指定，使用3x3卷积而不是1x1卷积进行跳跃连接
        use_scale_shift_norm: 是否使用尺度-偏移归一化（条件归一化）
        dims: 信号维度（1D、2D或3D）
        use_checkpoint: 是否使用梯度检查点
        up: 是否用于上采样
        down: 是否用于下采样
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels, # 输入特征通道数
        emb_channels, # 时间步嵌入维度
        dropout, # Dropout概率
        out_channels=None, # 输出通道数（可选）
        use_conv=False, # 跳跃连接是否使用卷积
        use_scale_shift_norm=False, # 是否使用尺度-偏移归一化
        dims=2, # 维度：2=2D卷积，3=3D卷积
        use_checkpoint=False, # 梯度检查点（节省内存）
        up=False, # 上采样模式
        down=False, # 下采样模式
    ):
        super().__init__() # 调用TimestepBlock父类初始化
        # 1. 存储基本参数
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels # 默认输出通道=输入通道
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm # 条件归一化标志

        # 2. 输入层：特征提取和通道调整
        # 输入层作用：特征预处理和可能的通道数调整
        self.in_layers = nn.Sequential(
            normalization(channels, swish=1.0), # 组归一化 + SiLU激活
            nn.Identity(), # 恒等映射（占位符）
            conv_nd(dims, channels, self.out_channels, 3, padding=1), # 3x3卷积
        )

        # 3. 上/下采样处理
        self.updown = up or down # 标记是否进行空间维度变换

        if up:
            # 上采样模式：特征图和跳跃连接都上采样
            self.h_upd = Upsample(channels, False, dims) # 特征上采样
            self.x_upd = Upsample(channels, False, dims) # 跳跃连接上采样
        elif down:
            # 下采样模式：特征图和跳跃连接都下采样
            self.h_upd = Downsample(channels, False, dims) # 特征下采样
            self.x_upd = Downsample(channels, False, dims) # 跳跃连接下采样
        else:
            # 无采样：使用恒等映射
            self.h_upd = self.x_upd = nn.Identity()

        # 4. 时间步嵌入处理层
        # 将时间步嵌入投影到特征空间，用于条件化
        self.emb_layers = nn.Sequential(
            nn.SiLU(), # 激活函数
            linear( # 线性投影
                emb_channels,
                # 如果使用尺度-偏移归一化，输出2倍通道（尺度+偏移）
                # 否则输出正常通道数（直接相加）
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        # 5. 输出层：最终特征处理
        self.out_layers = nn.Sequential(
            # 条件归一化：根据use_scale_shift_norm选择不同的激活策略
            normalization(self.out_channels, swish=0.0 if use_scale_shift_norm else 1.0),
            nn.SiLU() if use_scale_shift_norm else nn.Identity(), # 条件激活
            nn.Dropout(p=dropout), # 随机失活
            # 零初始化的卷积，确保训练初期残差连接占主导
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        # 6. 跳跃连接处理
        if self.out_channels == channels:
            # 输入输出通道相同：直接恒等连接
            self.skip_connection = nn.Identity()
        elif use_conv:
            # 使用3x3卷积调整通道数（保持空间信息）
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            # 使用1x1卷积调整通道数（更轻量）
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        前向传播：应用带时间步条件化的残差块。

        Args:
            x: 输入特征张量，形状为 [N, C, ...]（批次大小, 通道数, 空间维度）
            emb: 时间步嵌入张量，形状为 [N, emb_channels]（批次大小, 嵌入维度）

        Returns:
            输出张量，形状与x相同（但通道数可能改变）

        数学公式：
            h = in_layers(x)                    # 输入处理
            emb_out = emb_layers(emb)           # 时间步条件处理
            h = condition(h, emb_out)           # 条件化融合
            output = skip_connection(x) + h     # 残差连接

        处理流程：
            1. 输入特征预处理（可能包含上/下采样）
            2. 时间步嵌入处理
            3. 特征与时间步条件融合
            4. 输出处理 + 残差连接
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # 1. 输入特征处理
        if self.updown:
            # 上/下采样模式：分离卷积层进行特殊处理
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1] # 分离最后卷积层
            h = in_rest(x) # 应用除最后卷积外的所有层
            h = self.h_upd(h) # 特征上/下采样
            x = self.x_upd(x) # 跳跃连接上/下采样（保持对齐）
            h = in_conv(h) # 应用最后的卷积层
        else:
            # 普通模式：直接应用所有输入层
            h = self.in_layers(x)
        # 2. 时间步嵌入处理
        emb_out = self.emb_layers(emb).type(h.dtype) # 投影并确保数据类型一致
        # 3. 调整嵌入张量的维度以匹配特征张量
        # 例如：将 [N, C] 扩展为 [N, C, 1, 1] 以匹配2D特征 [N, C, H, W]
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None] # 在末尾添加维度
        # 4. 特征与时间步条件融合
        if self.use_scale_shift_norm:
            # 尺度-偏移归一化（FiLM机制）：更精细的条件控制
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:] # 分离归一化层
            # 将嵌入输出分割为尺度(scale)和偏移(shift)两部分
            scale, shift = th.chunk(emb_out, 2, dim=1) # 沿通道维度分割
            # 应用条件归一化：h = norm(h) * (1 + scale) + shift
            h = out_norm(h) * (1 + scale) + shift # 尺度缩放 + 偏移调整
            h = out_rest(h) # 应用剩余的输出层
        else:
            # 简单相加条件化：h = h + emb_out
            h = h + emb_out # 直接特征相加
            h = self.out_layers(h) # 应用所有输出层
        # 5. 残差连接：输出 = 跳跃连接(x) + 处理后的特征(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    注意力块：允许空间位置之间相互关注的自注意力/交叉注意力机制。

    这是扩散模型中的核心注意力组件，支持：
    1. 自注意力（Self-Attention）：输入特征内部的空间关系建模
    2. 交叉注意力（Cross-Attention）：输入特征与编码器输出的条件融合

    原始实现移植自TensorFlow版本，并适配到N维情况。

    Args:
        channels: 输入特征通道数
        num_heads: 注意力头数（如果num_head_channels=-1则使用）
        num_head_channels: 每个注意力头的通道数（优先级高于num_heads）
        use_checkpoint: 是否使用梯度检查点节省内存
        encoder_channels: 编码器输出通道数（如果提供则启用交叉注意力）
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels, # 输入特征通道数
        num_heads=1, # 注意力头数（默认1头）
        num_head_channels=-1, # 每头通道数（-1表示自动计算）
        use_checkpoint=False, # 梯度检查点
        encoder_channels=None, # 编码器通道数（None表示自注意力）
    ):
        super().__init__()
        self.channels = channels
        # 1. 计算注意力头数
        if num_head_channels == -1:
            # 使用指定的头数
            self.num_heads = num_heads
        else:
            # 使用每头通道数计算头数
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels # 总通道数 / 每头通道数

        self.use_checkpoint = use_checkpoint
        # 2. 归一化层：层归一化，不使用swish激活
        self.norm = normalization(channels, swish=0.0)
        # 3. QKV投影层：1x1卷积将输入投影为Q、K、V三部分
        # 输出通道 = channels * 3 (Q, K, V各占1/3)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        # 4. 注意力计算核心模块
        self.attention = QKVAttention(self.num_heads)

        # 5. 编码器KV投影（交叉注意力）：如果提供编码器输出
        if encoder_channels is not None:
            # 将编码器输出投影为K和V（各占channels）
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)
        # 6. 输出投影层：零初始化的1x1卷积
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x, encoder_out=None):
        """
        前向传播：应用注意力机制。

        Args:
            x: 输入特征张量，形状为 [batch, channels, *spatial_dims]
            encoder_out: 编码器输出（条件信息），形状为 [batch, encoder_channels, seq_len]
                       如果为None则使用自注意力，否则使用交叉注意力

        Returns:
            注意力增强后的特征，形状与输入x相同

        处理流程：
            1. 特征归一化
            2. QKV投影
            3. 注意力计算（自注意力或交叉注意力）
            4. 输出投影
            5. 残差连接
        """
        # 获取输入形状信息
        b, c, *spatial = x.shape # b=batch_size, c=channels, spatial=空间维度列表
        # 1. 特征归一化并展平空间维度
        # x形状: [b, c, *spatial] → [b, c, total_spatial_elements]
        # 2. QKV投影：将特征投影为查询(Query)、键(Key)、值(Value)
        # 输出形状: [b, c*3, total_spatial_elements]
        qkv = self.qkv(self.norm(x).view(b, c, -1)) # 空间位置总数（H*W或H*W*D）
        # 3. 注意力计算
        if encoder_out is not None:
            # 交叉注意力模式：使用编码器输出作为条件
            # 将编码器输出投影为K和V
            encoder_out = self.encoder_kv(encoder_out) # [b, c*2, seq_len]
            # 注意力计算：qkv作为Q，encoder_kv作为K和V
            h = self.attention(qkv, encoder_out)
        else:
            # 自注意力模式：Q、K、V都来自输入特征
            h = self.attention(qkv)
        # 4. 输出投影：将注意力输出映射回原始通道空间
        h = self.proj_out(h) # [b, c, total_spatial_elements]
        return x + h.reshape(b, c, *spatial) # 残差连接：输入 + 注意力增强


class QKVAttention(nn.Module):
    """
    QKV注意力计算模块：执行多头注意力机制的核心计算。

    匹配传统QKV注意力计算 + 输入/输出头的形状处理。
    支持自注意力和交叉注意力的统一计算。

    Args:
        n_heads: 注意力头数
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads # 注意力头数

    def forward(self, qkv, encoder_kv=None):
        """
        应用QKV注意力计算。

        Args:
            qkv: 包含Q、K、V的合并张量，形状为 [N, (H * 3 * C), T]
                 其中：N=batch_size, H=头数, C=每头通道数, T=序列长度
            encoder_kv: 编码器的K和V（交叉注意力时提供），形状为 [N, (H * 2 * C), S]

        Returns:
            注意力加权后的输出，形状为 [N, (H * C), T]

        数学公式：
            Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        # 获取输入形状
        bs, width, length = qkv.shape # batch_size, 总通道数, 序列长度
        # 验证通道数可被3*头数整除（Q、K、V各占1/3）
        assert width % (3 * self.n_heads) == 0
        # 计算每头通道数
        ch = width // (3 * self.n_heads) # 每头的通道数
        # 1. 分割Q、K、V并重塑为多头格式
        # 原始: [bs, 3*H*C, T] → 重塑: [bs*H, 3*C, T] → 分割: 各[bs*H, C, T]
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1) # 沿通道维度分割为三等份
        # 2. 交叉注意力处理：拼接编码器的K和V
        if encoder_kv is not None:
            # 验证编码器通道数
            assert encoder_kv.shape[1] == self.n_heads * ch * 2
            # 分割编码器的K和V
            ek, ev = encoder_kv.reshape(bs * self.n_heads, ch * 2, -1).split(ch, dim=1) # 各[bs*H, C, S]
            # 拼接：编码器K+V + 输入K+V
            k = th.cat([ek, k], dim=-1) # 沿序列维度拼接：[bs*H, C, S+T]
            v = th.cat([ev, v], dim=-1) # [bs*H, C, S+T]
        # 3. 缩放点积注意力计算
        # 缩放因子：1/sqrt(sqrt(ch))，数值稳定性优化
        scale = 1 / math.sqrt(math.sqrt(ch))
        # 计算注意力权重：Q·K^T
        # 使用爱因斯坦求和约定进行高效的矩阵乘法
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards # 输出形状: [bs*H, T, S+T]
        # 4. 应用softmax得到注意力权重
        # 先转换为float32确保数值稳定性，再转换回原类型
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        # 5. 注意力加权求和：权重 · 值
        a = th.einsum("bts,bcs->bct", weight, v) # [bs*H, C, T]
        # 6. 重塑回原始格式：合并多头
        return a.reshape(bs, -1, length) # [bs, H*C, T]


class UNetModel(nn.Module):
    """
    完整的UNet模型，包含注意力机制和时间步嵌入。

    这是扩散模型（如DDPM、Stable Diffusion）的核心架构，用于从噪声中重建图像。
    支持条件生成（类别条件、文本条件等）和注意力机制。

    Args:
        in_channels: 输入张量的通道数
        model_channels: 模型的基础通道数
        out_channels: 输出张量的通道数
        num_res_blocks: 每个下采样级别的残差块数量
        attention_resolutions: 应用注意力的下采样率集合
           例如：[4, 8] 表示在4倍和8倍下采样时应用注意力
        dropout: dropout概率
        channel_mult: UNet每个级别的通道倍增因子
           例如：(1, 2, 4, 8) 表示每下采样一次通道数翻倍
        conv_resample: 如果为True，使用学习的卷积进行上/下采样
        dims: 信号维度（1D、2D或3D）
        num_classes: 如果指定，则为类别条件模型
        use_checkpoint: 使用梯度检查点减少内存使用
        use_fp16: 使用半精度浮点数
        num_heads: 每个注意力层的注意力头数
        num_head_channels: 如果指定，忽略num_heads，使用固定的每头通道宽度
        num_heads_upsample: 上采样时使用不同的头数（已弃用）
        use_scale_shift_norm: 使用FiLM-like条件机制
        resblock_updown: 对上/下采样使用残差块
        encoder_channels: 编码器通道数（用于交叉注意力）
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
        self,
        in_channels, # 输入通道数
        model_channels, # 基础通道数
        out_channels, # 输出通道数
        num_res_blocks, # 每个下采样级别的残差块数
        attention_resolutions, # 应用注意力的分辨率
        dropout=0, # dropout率
        channel_mult=(1, 2, 4, 8), # 通道倍增因子
        conv_resample=True, # 是否使用卷积重采样
        dims=2, # 维度（1D/2D/3D）
        num_classes=None, # 类别数（用于条件生成）
        use_checkpoint=False, # 梯度检查点
        use_fp16=False, # 半精度
        num_heads=1, # 注意力头数
        num_head_channels=-1, # 每头通道数
        num_heads_upsample=-1, # 上采样头数
        use_scale_shift_norm=False, # 尺度偏移归一化
        resblock_updown=False, # 残差上/下采样
        encoder_channels=None, # 编码器通道数
    ):
        super().__init__()

        # 1. 初始化参数
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads # 默认使用相同头数

        # 存储模型参数
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32 # 数据类型
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # 2. 时间步嵌入网络
        # 将时间步编码为高维向量，用于条件生成
        time_embed_dim = model_channels * 4 # 通常扩展4倍
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim), # 线性投影
            nn.SiLU(), # 激活函数
            linear(time_embed_dim, time_embed_dim), # 再次投影
        )

        # 3. 类别条件嵌入（如果提供类别）
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # 4. 输入块构建（编码器路径）
        ch = input_ch = int(channel_mult[0] * model_channels) # 初始通道数
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch # 跟踪特征图大小
        input_block_chans = [ch] # 存储每个块的输出通道数（用于跳跃连接）
        ds = 1 # 下采样因子
        # 遍历通道倍增级别
        for level, mult in enumerate(channel_mult):
            # 每个级别的残差块
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch, # 输入通道
                        time_embed_dim, # 时间嵌入维度
                        dropout, # dropout率
                        out_channels=int(mult * model_channels), # 输出通道
                        dims=dims, # 维度
                        use_checkpoint=use_checkpoint, # 梯度检查点
                        use_scale_shift_norm=use_scale_shift_norm, # 条件归一化
                    )
                ]
                ch = int(mult * model_channels) # 更新当前通道数
                # 在指定分辨率添加注意力层
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, # 输入通道
                            use_checkpoint=use_checkpoint, # 梯度检查点
                            num_heads=num_heads, # 注意力头数
                            num_head_channels=num_head_channels, # 每头通道
                            encoder_channels=encoder_channels, # 编码器通道（交叉注意力）
                        )
                    )
                # 添加到输入块
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch # 更新特征大小
                input_block_chans.append(ch) # 存储通道数
            # 如果不是最后一个级别，添加下采样
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        # 使用残差块下采样或普通下采样
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True, # 下采样模式
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch) # 存储下采样后的通道
                ds *= 2 # 更新下采样因子
                self._feature_size += ch

        # 5. 中间块（瓶颈层）
        self.middle_block = TimestepEmbedSequential(
            # 残差块1
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # 注意力块
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                encoder_channels=encoder_channels,
            ),
            # 残差块2
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        # print(input_block_chans)

        # 6. 输出块构建（解码器路径）
        self.output_blocks = nn.ModuleList([])
        # 反向遍历通道倍增级别（从深层到浅层）
        for level, mult in list(enumerate(channel_mult))[::-1]:
            # 每个级别的残差块（比输入路径多一个块，用于跳跃连接）
            for i in range(num_res_blocks + 1):
                # 从输入路径获取对应层的通道数（跳跃连接）
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich, # 当前通道 + 跳跃连接通道
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult), # 输出通道
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                # 在指定分辨率添加注意力层
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample, # 可能使用不同的头数
                            num_head_channels=num_head_channels,
                            encoder_channels=encoder_channels,
                        )
                    )
                # 如果不是第一个级别且是最后一个块，添加上采样
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True, # 上采样模式
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2 # 减少下采样因子（上采样）
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # 7. 输出层
        self.out = nn.Sequential(
            normalization(ch, swish=1.0), # 归一化层
            nn.Identity(), # 恒等映射
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)), # 输出卷积
        )
        self.use_fp16 = use_fp16

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None):
        """
        对输入批次应用模型。

        Args:
            x: 输入张量，形状为 [N, C, ...]
            timesteps: 1D时间步张量，形状为 [N]
            y: 条件标签张量，形状为 [N]（如果是类别条件模型）

        Returns:
            输出张量，形状为 [N, C, ...]
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # 1. 验证条件输入
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # 2. 时间步和条件嵌入
        hs = [] # 存储跳跃连接特征
        # 生成时间步嵌入
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # 如果使用类别条件，添加标签嵌入
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y) # 条件嵌入相加

        # 3. 编码器路径（下采样）
        h = x.type(self.dtype) # 转换为模型数据类型
        for module in self.input_blocks:
            h = module(h, emb) # 应用块（传入时间嵌入）
            hs.append(h) # 存储特征用于跳跃连接
        # 4. 瓶颈层
        h = self.middle_block(h, emb)
        # 5. 解码器路径（上采样）
        for module in self.output_blocks:
            # 跳跃连接：拼接当前特征和对应编码器特征
            h = th.cat([h, hs.pop()], dim=1) # 通道维度拼接
            h = module(h, emb) # 应用块
        # 6. 输出层
        h = h.type(x.dtype) # 转换回输入数据类型
        return self.out(h) # 最终输出

class SuperResUNetModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, low_res=None, **kwargs):
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)

    
class InpaintUNetModel(UNetModel):
    """
    A UNetModel which can perform inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 2 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 2 + 1
        super().__init__(*args, **kwargs)

    def forward(self, x, timesteps, inpaint_image=None, inpaint_mask=None, **kwargs):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        return super().forward(
            th.cat([x, inpaint_image * inpaint_mask, inpaint_mask], dim=1),
            timesteps,
            **kwargs,
        )


class SuperResInpaintUNetModel(UNetModel):
    """
    A UNetModel which can perform both upsampling and inpainting.
    """

    def __init__(self, *args, **kwargs):
        if "in_channels" in kwargs:
            kwargs = dict(kwargs)
            kwargs["in_channels"] = kwargs["in_channels"] * 3 + 1
        else:
            # Curse you, Python. Or really, just curse positional arguments :|.
            args = list(args)
            args[1] = args[1] * 3 + 1
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x,
        timesteps,
        inpaint_image=None,
        inpaint_mask=None,
        low_res=None,
        **kwargs,
    ):
        if inpaint_image is None:
            inpaint_image = th.zeros_like(x)
        if inpaint_mask is None:
            inpaint_mask = th.zeros_like(x[:, :1])
        _, _, new_height, new_width = x.shape
        upsampled = F.interpolate(low_res, (new_height, new_width), mode="bilinear")
        return super().forward(
            th.cat([x, inpaint_image * inpaint_mask, inpaint_mask, upsampled], dim=1),
            timesteps,
            **kwargs,
        )

if __name__ == '__main__':
    model = UNetModel(
        in_channels=3,
        model_channels=32,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=[4, 2, 1],
        channel_mult=[1, 2, 4],
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=8,
        num_heads_upsample=8,
    )
    print(model)
    print(len(model.input_blocks))
    print(len(model.output_blocks))
    for module in model.input_blocks:
        print(f'module: {module}')