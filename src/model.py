import torch
import torch.nn as nn

STYLE_DIM = 64


def conv_bn_act(in_ch, out_ch, stride=1, kernel=3):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2),
        nn.InstanceNorm2d(out_ch, affine=True),
        nn.LeakyReLU(0.2, inplace=True),
    )


class StyleEncoder(nn.Module):
    def __init__(self, in_ch: int = 4, style_dim: int = STYLE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn_act(in_ch, 32, stride=2),   # 128 -> 64
            conv_bn_act(32, 64, stride=2),      # 64  -> 32
            conv_bn_act(64, 128, stride=2),     # 32  -> 16
            conv_bn_act(128, 128, stride=2),    # 16  -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, style_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        feat = self.pool(feat).flatten(1)
        return self.proj(feat)  # (B, style_dim)


class MSLKA(nn.Module):
    def __init__(self, channels: int, style_dim: int = STYLE_DIM, kernel: int = 5):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel, padding=(kernel // 2) * d,
                      dilation=d, groups=channels)
            for d in (1, 4, 9)
        ])
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid(),  # channel-wise gate in (0, 1)
        )

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        fused = self.fuse(torch.cat(feats, dim=1))
        gate = self.style_mlp(style).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        modulated = fused * gate
        return x + modulated  


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = conv_bn_act(in_ch, out_ch, stride=2)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, style_dim=STYLE_DIM):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = conv_bn_act(in_ch, out_ch)
        self.mslka = MSLKA(out_ch, style_dim=style_dim)

    def forward(self, x, style, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.mslka(x, style)
        return x


class Generator(nn.Module):
    def __init__(self, in_ch: int = 4, base_ch: int = 32, style_dim: int = STYLE_DIM):
        super().__init__()
        self.stem = conv_bn_act(in_ch, base_ch)               # 128
        self.down1 = DownBlock(base_ch, base_ch * 2)           # 64
        self.down2 = DownBlock(base_ch * 2, base_ch * 4)       # 32

        self.bottleneck = nn.Sequential(
            conv_bn_act(base_ch * 4, base_ch * 4),
            conv_bn_act(base_ch * 4, base_ch * 4),
        )

        # up1 input = bottleneck channels (4*base) + skip from down1 (2*base)
        self.up1 = UpBlock(base_ch * 4 + base_ch * 2, base_ch * 2, style_dim)
        # up2 input = up1 output (2*base) + skip from stem (1*base)
        self.up2 = UpBlock(base_ch * 2 + base_ch, base_ch, style_dim)

        self.to_raw = nn.Conv2d(base_ch, in_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)          # (B, base, 128, 128)
        s1 = self.down1(s0)        # (B, 2*base, 64, 64)
        s2 = self.down2(s1)        # (B, 4*base, 32, 32)

        # This is simplified version. it uses convolution based bottleneck layer instead of transformers 
        b = self.bottleneck(s2)    # (B, 4*base, 32, 32)

        u1 = self.up1(b, style, skip=s1)   # -> 64x64
        u2 = self.up2(u1, style, skip=s0)  # -> 128x128

        out = self.to_raw(u2)
        return torch.sigmoid(out)  # keep output in valid RAW range [0, 1]


class MERIT(nn.Module):
    def __init__(self, in_ch: int = 4, style_dim: int = STYLE_DIM):
        super().__init__()
        self.style_encoder = StyleEncoder(in_ch, style_dim)
        self.generator = Generator(in_ch, style_dim=style_dim)

    def translate(self, source: torch.Tensor, target_ref: torch.Tensor) -> torch.Tensor:
        """Translate `source` into the domain represented by `target_ref`."""
        style = self.style_encoder(target_ref)
        return self.generator(source, style)