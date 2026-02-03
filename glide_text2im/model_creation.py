from glide_text2im.gaussian_diffusion import get_named_beta_schedule
from glide_text2im.respace import SpacedDiffusion, space_timesteps
from glide_text2im.text2im_model import (
    InpaintText2ImUNet,
    SuperResInpaintText2ImUnet,
    SuperResText2ImUNet,
    Text2ImUNet,
)
from glide_text2im.tokenizer.bpe import get_encoder


def model_and_diffusion_defaults():
    return dict(
        image_size=64,
        num_channels=192,
        num_res_blocks=3,
        channel_mult="",
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        attention_resolutions="32,16,8",
        dropout=0.1,
        text_ctx=128,
        xf_width=512,
        xf_layers=16,
        xf_heads=8,
        xf_final_ln=True,
        xf_padding=True,
        diffusion_steps=1000,
        noise_schedule="squaredcos_cap_v2",
        timestep_respacing="",
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        cache_text_emb=False,
        inpaint=False,
        super_res=False,
    )


def model_and_diffusion_defaults_upsampler():
    result = model_and_diffusion_defaults()
    result.update(
        dict(
            image_size=256,
            num_res_blocks=2,
            noise_schedule="linear",
            super_res=True,
        )
    )
    return result


def create_model_and_diffusion(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    text_ctx,
    xf_width,
    xf_layers,
    xf_heads,
    xf_final_ln,
    xf_padding,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    cache_text_emb,
    inpaint,
    super_res,
):
    """
    创建扩散模型和对应的扩散过程

    这是GLIDE/Stable Diffusion等文本到图像模型的核心初始化函数
    负责构建U-Net主干网络和定义扩散过程的时间调度

    Args:
        # ============== 模型架构参数 ==============
        image_size: int, 输入图像尺寸（如256, 512）
        num_channels: int, 模型初始通道数（如128, 256）
        num_res_blocks: int, 每个分辨率级别的残差块数量

        # 通道倍增因子，控制U-Net各层通道数
        # 例如: (1, 2, 4, 8) 表示每下采样一次通道数翻倍
        channel_mult: tuple, 通道数倍增列表

        # ============== 注意力机制参数 ==============
        num_heads: int, 多头注意力的头数
        num_head_channels: int, 每个注意力头的通道数
        num_heads_upsample: int, 上采样阶段使用的注意力头数

        # 哪些分辨率级别使用注意力机制
        # 例如: (16, 8) 表示在16×16和8×8分辨率使用注意力
        attention_resolutions: tuple, 注意力分辨率列表

        # ============== 正则化参数 ==============
        dropout: float, Dropout概率（0.0-1.0）

        # ============== 文本编码器参数 ==============
        text_ctx: int, 文本上下文长度（如128, 256）
        xf_width: int, Transformer编码器隐藏层维度
        xf_layers: int, Transformer编码器层数
        xf_heads: int, Transformer注意力头数
        xf_final_ln: bool, 是否在Transformer输出前使用LayerNorm
        xf_padding: bool, 是否对文本进行填充

        # ============== 扩散过程参数 ==============
        diffusion_steps: int, 扩散总步数（如1000）
        noise_schedule: str, 噪声调度策略（'linear', 'cosine', 'sigmoid'）

        # 时间步重采样，用于加速推理
        # 例如: "100" 表示用100步代替1000步
        timestep_respacing: str, 时间步重采样策略

        # ============== 高级架构选项 ==============
        use_scale_shift_norm: bool, 是否在残差块中使用Scale-Shift归一化
        resblock_updown: bool, 是否在上采样块中使用残差连接
        use_fp16: bool, 是否使用混合精度训练
        cache_text_emb: bool, 是否缓存文本嵌入以节省内存

        # ============== 特殊任务标志 ==============
        inpaint: bool, 是否为图像修复任务
        super_res: bool, 是否为超分辨率任务

    Returns:
        tuple: (model, diffusion)
            - model: 配置好的U-Net扩散模型
            - diffusion: 高斯扩散过程对象
    """
    # 1. 创建U-Net扩散模型
    model = create_model(
        image_size, # 图像尺寸
        num_channels, # 基础通道数
        num_res_blocks, # 残差块数
        # 网络容量控制
        channel_mult=channel_mult, # 通道倍增
        attention_resolutions=attention_resolutions, # 注意力分辨率
        # 注意力机制配置
        num_heads=num_heads, # 注意力头数
        num_head_channels=num_head_channels, # 每头通道数
        num_heads_upsample=num_heads_upsample, # 上采样头数
        # 归一化选项
        use_scale_shift_norm=use_scale_shift_norm, # Scale-Shift归一化
        dropout=dropout, # 丢弃率
        # 文本编码器配置
        text_ctx=text_ctx, # 文本上下文长度
        xf_width=xf_width, # Transformer宽度
        xf_layers=xf_layers, # Transformer层数
        xf_heads=xf_heads, # Transformer头数
        xf_final_ln=xf_final_ln, # 最终层归一化
        xf_padding=xf_padding, # 填充选项
        # 架构细节
        resblock_updown=resblock_updown, # 上采样残差连接
        use_fp16=use_fp16, # 混合精度
        cache_text_emb=cache_text_emb, # 文本嵌入缓存
        # 任务特定配置
        inpaint=inpaint, # 图像修复模式
        super_res=super_res, # 超分辨率模式
    )
    # 2. 创建高斯扩散过程
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps, # 扩散总步数
        noise_schedule=noise_schedule, # 噪声调度策略
        timestep_respacing=timestep_respacing, # 时间步重采样
    )
    # 3. 返回模型和扩散过程对象
    return model, diffusion


def create_model(
    image_size, # 图像尺寸
    num_channels, # 基础通道数
    num_res_blocks, # 残差块数
    channel_mult, # 通道倍增
    attention_resolutions, # 注意力分辨率
    num_heads, # 注意力头数
    num_head_channels, # 每头通道数
    num_heads_upsample, # 上采样头数
    use_scale_shift_norm, # Scale-Shift归一化
    dropout, # 丢弃率
    text_ctx, # 文本上下文长度
    xf_width, # Transformer宽度
    xf_layers, # Transformer层数
    xf_heads, # Transformer头数
    xf_final_ln, # 最终层归一化
    xf_padding, # 填充选项
    resblock_updown, # 上采样残差连接
    use_fp16, # 混合精度
    cache_text_emb, # 文本嵌入缓存
    inpaint, # 图像修复模式
    super_res, # 超分辨率模式
):
    if channel_mult == "":
        if image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))
        assert 2 ** (len(channel_mult) + 2) == image_size

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    if inpaint and super_res:
        model_cls = SuperResInpaintText2ImUnet
    elif inpaint:
        model_cls = InpaintText2ImUNet
    elif super_res:
        model_cls = SuperResText2ImUNet
    else:
        model_cls = Text2ImUNet
    return model_cls(
        text_ctx=text_ctx,
        xf_width=xf_width,
        xf_layers=xf_layers,
        xf_heads=xf_heads,
        xf_final_ln=xf_final_ln,
        tokenizer=get_encoder(),
        xf_padding=xf_padding,
        in_channels=3,
        model_channels=num_channels,
        out_channels=6,
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        cache_text_emb=cache_text_emb,
    )


def create_gaussian_diffusion(
    steps,
    noise_schedule,
    timestep_respacing,
):
    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
    )
