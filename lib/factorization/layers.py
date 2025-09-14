import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, rank: int, bias: bool = True
    ):
        super(LowRankLinear, self).__init__()
        self.w0 = nn.Parameter(torch.randn(in_features, rank))
        self.w1 = nn.Parameter(torch.randn(out_features, rank))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None

    def forward(self, x: torch.Tensor):
        # X in R^{BATCH x IN}, W0 in R^{IN x RANK}, W1 in R^{RANK x OUT}
        w0, w1 = self.w0, self.w1
        return torch.nn.functional.linear(x @ w0, w1, bias=self.bias)

    def __repr__(self):
        return f"LowRankLinear(in_features={self.w0.shape[0]}, out_features={self.w1.shape[0]}, rank={self.w0.shape[1]}, bias={self.bias is not None})"

    def to_linear(self):
        res = nn.Linear(self.w0.shape[0], self.w1.shape[1], bias=self.bias is not None)
        res.weight = nn.Parameter((self.w0 @ self.w1).t().contiguous())
        if self.bias is not None:
            res.bias = self.bias
        return res


class LowRankConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rank: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(LowRankConv2d, self).__init__()
        H_k = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        W_k = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size
        assert in_channels % groups == 0
        assert out_channels % groups == 0
        self.w0 = nn.Parameter(
            torch.randn(rank * groups, in_channels // groups, H_k, W_k)
        )
        self.w1 = nn.Parameter(torch.randn(out_channels, rank, 1, 1))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.rank = rank
        self.input_channels = in_channels
        self.in_channels = in_channels
        self.kernel_size = (H_k, W_k)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor):
        # stage 1: spatial conv to rank per group
        conv_out = torch.nn.functional.conv2d(
            x,
            self.w0,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )  # (B, rank*groups, H, W)
        # stage 2: grouped 1x1 to mix the rank channels within each group
        y = torch.nn.functional.conv2d(
            conv_out,
            self.w1,
            bias=self.bias,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.groups,
        )  # (B, out_channels, H, W)
        return y

    def __repr__(self):
        return f"LowRankConv2d(in_channels={self.input_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, rank={self.rank}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None})"
