import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import timestep_embedding
from .unet import UNetModel
from .xf import LayerNorm, Transformer, convert_module_to_f16


class Text2ImUNet(UNetModel):
    """
    基于文本条件化的U-Net扩散模型，用于文本到图像生成任务。
    这是GLIDE/Stable Diffusion等文本引导生成模型的核心组件。

    继承自基础U-Net模型，通过Transformer编码文本条件并注入到U-Net中。

    Args:
        text_ctx: 文本上下文长度（文本token数量）
        xf_width: Transformer编码器的隐藏层维度
        xf_layers: Transformer编码器的层数
        xf_heads: Transformer的多头注意力头数
        xf_final_ln: 是否在Transformer输出后使用LayerNorm
        tokenizer: 文本分词器，用于词汇表大小和token编码
        cache_text_emb: 是否缓存文本嵌入以提高效率
        xf_ar: 自回归训练相关参数（默认为0，禁用）
        xf_padding: 是否支持文本填充处理
        share_unemb: 是否共享token嵌入和输出层的权重
        *args, **kwargs: 传递给父类UNetModel的参数

    A UNetModel that conditions on text with an encoding transformer.

    Expects an extra kwarg `tokens` of text.

    :param text_ctx: number of text tokens to expect.
    :param xf_width: width of the transformer.
    :param xf_layers: depth of the transformer.
    :param xf_heads: heads in the transformer.
    :param xf_final_ln: use a LayerNorm after the output layer.
    :param tokenizer: the text tokenizer for sampling/vocab size.
    """

    def __init__(
        self,
        text_ctx, # 文本上下文长度
        xf_width, # Transformer宽度
        xf_layers, # Transformer层数
        xf_heads, # Transformer头数
        xf_final_ln, # 最终层归一化
        tokenizer, # 文本分词器对象
        *args,
        cache_text_emb=False, # 文本嵌入缓存
        xf_ar=0.0, # 自回归训练权重（0表示禁用）
        xf_padding=False, # 填充选项
        share_unemb=False, # 权重共享标志
        **kwargs,
    ):
        # 初始化文本相关参数
        self.text_ctx = text_ctx
        self.xf_width = xf_width
        self.xf_ar = xf_ar
        self.xf_padding = xf_padding
        self.tokenizer = tokenizer

        # 调用父类UNetModel初始化
        # 根据是否有文本编码器决定是否传递encoder_channels
        if not xf_width:
            # 无文本编码器情况：用于无条件生成
            super().__init__(*args, **kwargs, encoder_channels=None)
        else:
            # 有文本编码器：传递Transformer维度作为编码器通道数
            super().__init__(*args, **kwargs, encoder_channels=xf_width)
        # 如果配置了文本编码器（xf_width > 0），初始化Transformer组件
        if self.xf_width:
            # 1. 初始化Transformer编码器
            self.transformer = Transformer(
                text_ctx, # 序列长度
                xf_width, # 隐藏维度
                xf_layers, # 层数
                xf_heads, # 注意力头数
            )
            # 2. 可选的最终层归一化
            if xf_final_ln:
                self.final_ln = LayerNorm(xf_width) # 层归一化
            else:
                self.final_ln = None

            # 3. Token嵌入层：将离散token转换为连续向量
            # 输入形状: [batch, text_ctx] -> 输出形状: [batch, text_ctx, xf_width]
            self.token_embedding = nn.Embedding(self.tokenizer.n_vocab, xf_width) # (词汇表大小, 嵌入维度)
            # 4. 位置编码：学习序列中每个位置的位置信息
            # 可学习参数，形状: [text_ctx, xf_width]
            self.positional_embedding = nn.Parameter(th.empty(text_ctx, xf_width, dtype=th.float32))
            # 5. 投影层：将Transformer输出投影到U-Net条件空间
            # 将xf_width维度投影到model_channels*4（用于时间嵌入融合）
            self.transformer_proj = nn.Linear(xf_width, self.model_channels * 4)

            # 6. 填充嵌入处理（用于变长文本序列）
            if self.xf_padding:
                self.padding_embedding = nn.Parameter(
                    th.empty(text_ctx, xf_width, dtype=th.float32)
                )
            # 7. 自回归相关组件（用于文本生成任务）
            if self.xf_ar:
                # 输出层：将隐藏状态投影回词汇表空间
                self.unemb = nn.Linear(xf_width, self.tokenizer.n_vocab)
                # 权重共享：减少参数数量
                if share_unemb:
                    self.unemb.weight = self.token_embedding.weight

        # 8. 文本嵌入缓存机制（推理优化）
        self.cache_text_emb = cache_text_emb
        self.cache = None # 缓存存储

    def convert_to_fp16(self):
        super().convert_to_fp16()
        if self.xf_width:
            self.transformer.apply(convert_module_to_f16)
            self.transformer_proj.to(th.float16)
            self.token_embedding.to(th.float16)
            self.positional_embedding.to(th.float16)
            if self.xf_padding:
                self.padding_embedding.to(th.float16)
            if self.xf_ar:
                self.unemb.to(th.float16)

    def get_text_emb(self, tokens, mask):
        """
        获取文本嵌入表示，支持缓存优化。

        Args:
            tokens: 文本token序列，形状 [batch, text_ctx]
            mask: 注意力掩码，形状 [batch, text_ctx]（用于变长序列）

        Returns:
            dict: 包含文本嵌入的字典
                - xf_proj: 投影后的条件向量 [batch, model_channels*4]
                - xf_out: 完整的序列输出 [batch, xf_width, text_ctx]
        """
        assert tokens is not None

        # 缓存检查：如果启用缓存且缓存存在，直接返回缓存结果
        if self.cache_text_emb and self.cache is not None:
            # 验证token是否与缓存一致（确保条件一致）
            assert (
                tokens == self.cache["tokens"]
            ).all(), f"Tokens {tokens.cpu().numpy().tolist()} do not match cache {self.cache['tokens'].cpu().numpy().tolist()}"
            return self.cache

        # 1. Token嵌入：将离散token转换为连续向量
        # tokens.long(): [batch, text_ctx] -> xf_in: [batch, text_ctx, xf_width]
        xf_in = self.token_embedding(tokens.long())
        # 2. 添加位置编码
        # positional_embedding: [text_ctx, xf_width] -> 广播加到每个批次
        xf_in = xf_in + self.positional_embedding[None] # [None]增加批次维度
        # 3. 处理文本填充（用于变长序列）
        if self.xf_padding:
            assert mask is not None
            # 将填充位置的嵌入替换为可学习的填充嵌入
            # mask 为 False 的位置，不用真实 token embedding，而改用可学习 padding embedding
            xf_in = th.where(mask[..., None], xf_in, self.padding_embedding[None])
        # 4. 通过Transformer编码器
        # xf_in: [batch, text_ctx, xf_width] -> xf_out: [batch, text_ctx, xf_width]
        xf_out = self.transformer(xf_in.to(self.dtype))
        # 5. 可选的最终层归一化
        if self.final_ln is not None:
            xf_out = self.final_ln(xf_out)
        # 6. 投影到条件空间（使用最后一个token的表示）
        # xf_out[:, -1]: 取序列最后一个token -> [batch, xf_width]
        # xf_proj: [batch, model_channels * 4]（用于时间步条件）
        # TODO 这里为什么是取最后一个token的表示？
        xf_proj = self.transformer_proj(xf_out[:, -1])
        xf_out = xf_out.permute(0, 2, 1)  # NLC -> NCL

        # 准备输出字典
        outputs = dict(xf_proj=xf_proj, xf_out=xf_out)

        # 8. 缓存处理（推理优化）
        if self.cache_text_emb:
            self.cache = dict(
                tokens=tokens,
                xf_proj=xf_proj.detach(), # 分离梯度，仅用于推理
                xf_out=xf_out.detach() if xf_out is not None else None,
            )

        return outputs

    def del_cache(self):
        self.cache = None

    def forward(self, x, timesteps, tokens=None, mask=None):
        """
        前向传播：文本条件化的U-Net扩散过程。

        Args:
            x: 噪声图像 latent，形状 [batch, channels, height, width]
            timesteps: 扩散时间步，形状 [batch]
            tokens: 文本token序列，形状 [batch, text_ctx]
            mask: 注意力掩码，形状 [batch, text_ctx]

        Returns:
            predicted_noise: 预测的噪声，形状同x
        """
        # 存储各层特征用于跳跃连接
        hs = []
        # 1. 时间步嵌入
        # timesteps: [batch] -> emb: [batch, time_embed_dim(model_channels * 4)]
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # 2. 文本条件处理
        if self.xf_width:
            # 获取文本嵌入
            text_outputs = self.get_text_emb(tokens, mask)
            # xf_proj,尺寸为[B,C],xf_out,尺寸为[B,C,N]
            xf_proj, xf_out = text_outputs["xf_proj"], text_outputs["xf_out"]
            # 将文本条件融合到时间步嵌入中
            # xf_proj: [batch, model_channels*4] -> 与时间嵌入相加
            emb = emb + xf_proj.to(emb)
        else:
            # 无条件生成：不使用文本条件
            xf_out = None
        # 3. U-Net编码器路径（下采样）
        h = x.type(self.dtype) # 确保数据类型一致
        # 输入块处理：逐步下采样，保存跳跃连接特征
        for module in self.input_blocks:
            # 每个模块接收：当前特征、时间嵌入、文本条件
            h = module(h, emb, xf_out) # 文本条件通过交叉注意力注入
            hs.append(h) # 保存特征用于跳跃连接
        # 4. U-Net瓶颈层（最底层）
        h = self.middle_block(h, emb, xf_out)
        # 5. U-Net解码器路径（上采样）
        for module in self.output_blocks:
            # 跳跃连接：拼接当前特征和对应编码器特征
            h = th.cat([h, hs.pop()], dim=1) # 通道维度拼接
            # 上采样块处理
            h = module(h, emb, xf_out)
        # 6. 最终输出层
        h = h.type(x.dtype) # 恢复原始数据类型
        h = self.out(h)  # 最终卷积，输出预测的噪声
        return h


class SuperResText2ImUNet(Text2ImUNet):
    """
    A text2im model that performs super-resolution.
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
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear", align_corners=False
        )
        x = th.cat([x, upsampled], dim=1)
        return super().forward(x, timesteps, **kwargs)


class InpaintText2ImUNet(Text2ImUNet):
    """
    A text2im model which can perform inpainting.
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


class SuperResInpaintText2ImUnet(Text2ImUNet):
    """
    A text2im model which can perform both upsampling and inpainting.
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
        upsampled = F.interpolate(
            low_res, (new_height, new_width), mode="bilinear", align_corners=False
        )
        return super().forward(
            th.cat([x, inpaint_image * inpaint_mask, inpaint_mask, upsampled], dim=1),
            timesteps,
            **kwargs,
        )
