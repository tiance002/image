"""NAFNet：非线性激活无关网络（Chen et al., ECCV 2022）。

这是当前图像复原（去噪/去模糊/去雨）最常用的强基线骨干，官方在 SIDD/GoPro
上都是 SOTA 水平，参数量与计算量比 Transformer 系小得多，适合单卡 8GB 训练。

本实现严格按官方结构：

    NAFBlock = LayerNorm
               + 空间分支（1x1 -> 3x3 深度卷积 -> SimpleGate -> 通道注意力 -> 1x1）
               + 通道分支（1x1 -> SimpleGate -> 1x1）
               两段都带可学习缩放系数 beta / gamma

其中 SimpleGate 用"通道对半相乘"替代 GELU/ReLU，是 NAFNet 的核心改动；
通道注意力用全局平均池化后的 1x1 卷积，替代 SE 的两层 MLP。

相比原项目的 FidelityPreservingRestorer（base_channels=16，约几十万参数），
这里的容量提升了两个数量级，是"真实数据上复原质量不够"的直接对策之一。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """在通道维做 LayerNorm（PyTorch 原生 LayerNorm 只支持最后一维）。"""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 必须在 float32 下算：fp16 的方差很容易落到 1e-5 以下，
        # 而 eps=1e-6 在 fp16 里已是次正规数，sqrt(var+eps) 会把误差放大成
        # NaN/Inf。实测 fp16 训练在第 0 步就出现 NaN 梯度并塌成全白输出。
        dtype = x.dtype
        xf = x.float()
        mu = xf.mean(dim=1, keepdim=True)
        var = (xf - mu).pow(2).mean(dim=1, keepdim=True)
        y = (xf - mu) / torch.sqrt(var + self.eps)
        y = y * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return y.to(dtype)


class SimpleGate(nn.Module):
    """通道对半切开后逐元素相乘，替代逐元素非线性激活。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop_rate: float = 0.0):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)
        # 简化通道注意力：全局池化 + 单层 1x1 卷积
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )
        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.sg = SimpleGate()
        self.dropout1 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class NAFNet(nn.Module):
    """U 形 NAFNet。width=32、enc=[1,1,1,16] 约 10M 参数。"""

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        width: int = 32,
        enc_blk_nums: tuple[int, ...] = (1, 1, 1, 16),
        middle_blk_num: int = 4,
        dec_blk_nums: tuple[int, ...] = (1, 1, 1, 1),
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.intro = nn.Conv2d(in_ch, width, 3, padding=1, bias=True)
        self.ending = nn.Conv2d(width, out_ch, 3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2))
            )
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

        if residual:
            # 残差模式下把输出层权重清零，使网络在初始化时严格等于恒等映射。
            # 否则随机初始化的输出会给输入加噪声，前期收敛明显变慢。
            nn.init.zeros_(self.ending.weight)
            nn.init.zeros_(self.ending.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        y = self.intro(x)
        skips: list[torch.Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            y = encoder(y)
            skips.append(y)
            y = down(y)

        y = self.middle_blks(y)

        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            y = up(y)
            y = y + skip
            y = decoder(y)

        out = self.ending(y)
        out = out[..., :h, :w]
        # x可能为pad后的尺寸；残差也必须裁回原始高宽。
        return torch.clamp(x[..., :h, :w] + out, 0.0, 1.0) if self.residual else out


def build_nafnet(width: int = 32, variant: str = "standard") -> NAFNet:
    """按官方常见配置构造网络，方便在实验里用名字切换。"""
    presets = {
        "tiny": dict(width=16, enc_blk_nums=(1, 1, 1, 8), middle_blk_num=2, dec_blk_nums=(1, 1, 1, 1)),
        "standard": dict(width=32, enc_blk_nums=(1, 1, 1, 16), middle_blk_num=4, dec_blk_nums=(1, 1, 1, 1)),
        "wide": dict(width=48, enc_blk_nums=(1, 1, 1, 16), middle_blk_num=4, dec_blk_nums=(1, 1, 1, 1)),
    }
    return NAFNet(**presets[variant])


if __name__ == "__main__":
    for name in ("tiny", "standard", "wide"):
        net = build_nafnet(variant=name)
        n = sum(p.numel() for p in net.parameters())
        print(f"{name}: {n / 1e6:.2f}M 参数")
