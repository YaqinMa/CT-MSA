import torch
import torch.nn as nn
import torch.nn.functional as F
import models.Transformer64 as Transformer64
import models.CNN64 as CNN64
import numpy as np





class CT-MSA(nn.Module):
    def __init__(self, args):
        super(CT-MSA, self).__init__()
        self.args = args
        self.n_input = args.n_input
        self.bottom_width = args.bottom_width
        self.embed_dim = args.gf_dim
        self.outdim = int(np.ceil((args.img_size ** 2) // (args.bottom_width ** 2)))
	self.dropout = nn.Dropout(0.2)  # 25%及以上采样率
            

        # 初始化卷积
        self.iniconv = nn.Sequential(
            nn.Conv2d(self.n_input, 128, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0)
        )

        # 测量矩阵
        self.Phi = nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(self.n_input, 256)))
        self.PhiT = nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(256, self.n_input)))

        # 主分支
        self.td = Transformer64.Transformer(args)
        self.gs = CNN64.Generator(args)

        # 多尺度融合模块
        self.cmsf_layers = nn.ModuleList([
            CMSF(dim=self.embed_dim // (2 ** i), reduction=16)
            for i in range(4)
        ])

        self.mkga_layers = nn.ModuleList([
            MKGA(n_feats=self._adjust_channels_for_mkga(self.embed_dim // (2 ** i)))
            for i in range(4)
        ])

        # 特征融合后处理
        self.fusion_post = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.embed_dim // (2 ** i), self.embed_dim // (2 ** i), 1),
                nn.LeakyReLU(0.2, inplace=True)
            )
            for i in range(4)
        ])

    def _adjust_channels_for_mkga(self, channels):
        """确保通道数能被3整除，这是mkga模块的要求"""
        if channels % 3 == 0:
            return channels
        else:
            # 调整到最近的能被3整除的数
            return ((channels // 3) + 1) * 3

    def together(self, inputs, S, H, L):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0)  # 将列表转为张量（如果是 list 类型）
        inputs = inputs.squeeze(1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=H * S, dim=0), dim=2)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=S, dim=0), dim=1)
        inputs = inputs.unsqueeze(1)
        return inputs

    def forward(self, inputs):
        # 分块处理
        H = int(inputs.shape[2] / 64)
        L = int(inputs.shape[3] / 64)
        S = inputs.shape[0]

        inputs = torch.squeeze(inputs, dim=1)
        inputs = torch.cat(inputs.chunk(H, dim=1), dim=0)
        inputs = torch.cat(inputs.chunk(L, dim=2), dim=0)
        inputs = inputs.unsqueeze(1)

        # 压缩感知采样
        PhiWeight = self.Phi.contiguous().view(self.n_input, 1, 16, 16)
        y = F.conv2d(inputs, PhiWeight, padding=0, stride=16, bias=None)

        # 初始重建
        PhiTWeight = self.PhiT.contiguous().view(256, self.n_input, 1, 1)
        PhiTb = F.conv2d(y, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(16)(PhiTb)

        # 特征提取
        x = self.iniconv(y)
        x = torch.nn.PixelShuffle(2)(x)
        x = x.flatten(2).transpose(1, 2).contiguous()

        # 获取CNN分支特征
        gsfeatures = self.gs(x)

        # 多尺度特征融合
        enhanced_features = []
        for i, (cmsf, mkga) in enumerate(zip(self.cmsf_layers, self.mkga_layers)):
            curr_feat = gsfeatures[i]

            # 确保4D格式
            if len(curr_feat.shape) == 3:
                B, L, C = curr_feat.shape
                H = W = int(np.sqrt(L))
                curr_feat = curr_feat.transpose(1, 2).reshape(B, C, H, W)

            # 调整PhiTb大小
            curr_phitb = F.interpolate(
                PhiTb,
                size=curr_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

            try:
                # 特征增强
                enhanced = cmsf(curr_feat, curr_phitb)

                # 如果通道数需要调整以适应mkga
                original_channels = enhanced.shape[1]
                mkga_channels = self._adjust_channels_for_mkga(original_channels)

                if original_channels != mkga_channels:
                    # 添加通道维度调整
                    channel_adjust = nn.Conv2d(original_channels, mkga_channels, 1).to(enhanced.device)
                    enhanced_adjusted = channel_adjust(enhanced)
                    enhanced_mkga = mkga(enhanced_adjusted)
                    # 调整回原来的通道数
                    channel_restore = nn.Conv2d(mkga_channels, original_channels, 1).to(enhanced.device)
                    enhanced = channel_restore(enhanced_mkga)
                else:
                    enhanced = mkga(enhanced)

                # 如果需要，转回3D格式
                if len(gsfeatures[i].shape) == 3:
                    enhanced = enhanced.flatten(2).transpose(1, 2)

                enhanced_features.append(enhanced)

            except Exception as e:
                print(f"\nError at stage {i}")
                print(f"curr_feat shape: {curr_feat.shape}")
                print(f"curr_phitb shape: {curr_phitb.shape}")
                raise e

        # Transformer处理
        output = self.td(x, enhanced_features, PhiTb)

        # 合并输出
        merge_output = self.together(output, S, H, L)
        merge_PhiTb = self.together(PhiTb, S, H, L)

        return merge_output, merge_PhiTb, output, PhiTb


def print_tensor_info(name, tensor):
    print(f"{name} shape: {tensor.shape}, dtype: {tensor.dtype}, device: {tensor.device}")


class SamplingLayer(nn.Module):
    def __init__(self, n_input, n_output=256):
        super(SamplingLayer, self).__init__()
        # 深度可分离卷积采样
        self.depthwise = nn.Conv2d(n_input, n_input, kernel_size=3, stride=2, padding=1, groups=n_input)
        self.pointwise = nn.Conv2d(n_input, n_output, kernel_size=1)
        # 进一步压缩采样
        self.conv = nn.Conv2d(n_output, n_output, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        y = self.conv(x)
        return y


class CMSF(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.dim = dim
        # 添加批归一化提高稳定性
        self.bn = nn.BatchNorm2d(dim)
        # 空间自适应池化
        self.pool_xw = nn.AdaptiveAvgPool2d((1, None))  # H维度池化
        self.pool_xh = nn.AdaptiveAvgPool2d((None, 1))  # W维度池化

        # 特征变换
        self.q = nn.Sequential(
            nn.Conv2d(1, dim // reduction, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )

        self.k = nn.Sequential(
            nn.Conv2d(1, dim // reduction, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )

        self.v = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

    def _print_shape(self, name, tensor):
        print(f"{name} shape: {tensor.shape}")

    def forward(self, x, y):
        B, C, H, W = x.shape

        # 确保y的形状正确
        if len(y.shape) == 3:
            y = y.reshape(B, C, H, W)

        # 空间池化
        xw = self.pool_xw(x)  # (B,C,1,W)
        yw = self.pool_xw(y)  # (B,C,1,W)
        xh = self.pool_xh(x)  # (B,C,H,1)
        yh = self.pool_xh(y)  # (B,C,H,1)

        # 压缩维度
        xw = xw.squeeze(2)  # (B,C,W)
        yw = yw.squeeze(2)  # (B,C,W)
        xh = xh.squeeze(3)  # (B,C,H)
        yh = yh.squeeze(3)  # (B,C,H)

        # 计算注意力
        f1 = torch.bmm(xh.permute(0, 2, 1), xw)  # (B,H,W)
        f1 = f1.unsqueeze(1)  # (B,1,H,W)
        f1 = self.q(f1)  # (B,C,H,W)

        f2 = torch.bmm(yh.permute(0, 2, 1), yw)  # (B,H,W)
        f2 = f2.unsqueeze(1)  # (B,1,H,W)
        f2 = self.k(f2)  # (B,C,H,W)

        # Value分支
        xt = self.v(x)  # (B,C,H,W)

        # 最终输出
        out = xt * f1 * f2
        return out


# ==================== 新的mkga模块 ====================

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



class SGAC(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        i_feats = n_feats * 2

        self.Conv1 = nn.Conv2d(n_feats, i_feats, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

    def forward(self, x):
        shortcut = x.clone()

        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)

        return x * self.scale + shortcut


# multi-scale large kernel attention (MRFC)
class MRFC(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        if n_feats % 3 != 0:
            raise ValueError("n_feats must be divisible by 3 for MRFC.")

        i_feats = 2 * n_feats

        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, stride=1, padding=(9 // 2) * 4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, stride=1, padding=(7 // 2) * 3, groups=n_feats // 3, dilation=3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, stride=1, padding=(5 // 2) * 2, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))

        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3)
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3)
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3)

        self.proj_first = nn.Sequential(
            nn.Conv2d(n_feats, i_feats, 1, 1, 0))

        self.proj_last = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        a = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3)],
                      dim=1)
        x = self.proj_last(x * a) * self.scale + shortcut
        return x


# multi-scale attention blocks (mkga)
class MKGA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.LKA = MRFC(n_feats)
        self.LFE = SGAC(n_feats)

    def forward(self, x):
        x = self.LKA(x)
        x = self.LFE(x)
        return x


# ==================== 保留原有的CA模块以防需要 ====================
class CA(nn.Module):
    def __init__(self, channel, reduction):
        super().__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv(x)
        return x * y