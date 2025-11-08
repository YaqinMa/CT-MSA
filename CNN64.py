import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import functools

def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W
# 卷积操作的 FLOPs 计算
def conv_flops(input_tensor, output_tensor, kernel_size, in_channels, out_channels):
    H_in, W_in = input_tensor.shape[2], input_tensor.shape[3]  # 输入图像的高度和宽度
    H_out, W_out = output_tensor.shape[2], output_tensor.shape[3]  # 输出图像的高度和宽度
    Kh, Kw = kernel_size  # 卷积核大小
    # 每个输出元素的 FLOPs 数量
    flops = Kh * Kw * in_channels * out_channels * H_out * W_out
    return flops
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel,up_mode='bicubic'):
        super(Upsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
        )
        self.up_mode = up_mode

    def forward(self, x):
        #up
        x = F.interpolate(x, scale_factor=2, mode=self.up_mode)
        #↓dim
        out = self.conv(x) # B H*W C
        return out

class ResGenerator(nn.Module):
    def __init__(self, args,upsample=Upsample):
        super(ResGenerator, self).__init__()
        self.args = args
        self.bottom_width = args.bottom_width
        self.embed_dim = conv_dim = args.gf_dim
        self.dec1 = nn.Sequential(
            ResBlock(in_channels=conv_dim, out_channels=conv_dim,norm_fun = args.cnnnorm_type),
            ResBlock(in_channels=conv_dim, out_channels=conv_dim,norm_fun = args.cnnnorm_type))# 8*8*128 --> 32*32*256
        self.upsample_1 = upsample(conv_dim, conv_dim // 2)
        self.dec2 = nn.Sequential(
            ResBlock(in_channels=conv_dim// 2, out_channels=conv_dim// 2,norm_fun = args.cnnnorm_type),
            ResBlock(in_channels=conv_dim// 2, out_channels=conv_dim// 2,norm_fun = args.cnnnorm_type))  # 16*16*128 --> 32*32*256
        self.upsample_2 = upsample(conv_dim//2, conv_dim // 4)
        self.dec3 =nn.Sequential(
            ResBlock(in_channels=conv_dim// 4, out_channels=conv_dim// 4,norm_fun = args.cnnnorm_type),
            ResBlock(in_channels=conv_dim// 4, out_channels=conv_dim// 4,norm_fun = args.cnnnorm_type))  # 32*32*128 --> 32*32*256
        self.upsample_3 = upsample(conv_dim // 4, conv_dim // 8)
        self.dec4 = nn.Sequential(
            ResBlock(in_channels=conv_dim// 8, out_channels=conv_dim// 8,norm_fun = args.cnnnorm_type),
            ResBlock(in_channels=conv_dim// 8, out_channels=conv_dim// 8,norm_fun = args.cnnnorm_type))  # 64*64*128 --> 32*32*256
        # self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.embed_dim)

        self.to_rgb = nn.ModuleList()
        self.padding3 = (3 + (3 - 1) * (1 - 1) - 1) // 2
        self.padding7 = (7 + (7 - 1) * (1 - 1) - 1) // 2


    def forward(self, x):
        features = []
        rgb = []
        x = x.permute(0,2,1).contiguous().view(-1,self.embed_dim, self.bottom_width,self.bottom_width)

        #8x8
        x = self.dec1(x)
        features.append(x)
        # rgb.append(self.to_rgb[0](x))

        #16x16
        x = self.upsample_1(x)
        x = self.dec2(x)
        features.append(x)
        # rgb.append(self.to_rgb[1](x))

        #32x32
        x = self.upsample_2(x)
        x = self.dec3(x)
        features.append(x)
        # rgb.append(self.to_rgb[2](x))

        #64x64
        x = self.upsample_3(x)
        x = self.dec4(x)
        features.append(x)

        return features,rgb
class MSCA(nn.Module):
    def __init__(self, dim, reduction=8):
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

        # 打印输入shape
        # self._print_shape("Input x", x)
        # self._print_shape("Input y", y)

        # 空间池化
        xw = self.pool_xw(x)  # (B,C,1,W)
        yw = self.pool_xw(y)  # (B,C,1,W)
        xh = self.pool_xh(x)  # (B,C,H,1)
        yh = self.pool_xh(y)  # (B,C,H,1)

        # self._print_shape("After pool xw", xw)
        # self._print_shape("After pool xh", xh)

        # 压缩维度
        xw = xw.squeeze(2)  # (B,C,W)
        yw = yw.squeeze(2)  # (B,C,W)
        xh = xh.squeeze(3)  # (B,C,H)
        yh = yh.squeeze(3)  # (B,C,H)

        # 计算注意力
        f1 = torch.bmm(xh.permute(0, 2, 1), xw)  # (B,H,W)
        f1 = f1.unsqueeze(1)  # (B,1,H,W)
        f1 = self.q(f1)  # (B,C,H,W)

        self._print_shape("After attention f1", f1)

        f2 = torch.bmm(yh.permute(0, 2, 1), yw)  # (B,H,W)
        f2 = f2.unsqueeze(1)  # (B,1,H,W)
        f2 = self.k(f2)  # (B,C,H,W)

        self._print_shape("After attention f2", f2)

        # Value分支
        xt = self.v(x)  # (B,C,H,W)
        self._print_shape("Value branch xt", xt)

        # 最终输出
        out = xt * f1 * f2
        self._print_shape("Final output", out)
        return out

class DCAB(nn.Module):
    def __init__(self, dim, reduction):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            # 修改这里: 对于4D输入需要指定正确的normalized_shape
            LayerNorm(dim, eps=1e-6, data_format="channels_first"),  # 使用自定义的LayerNorm
            nn.Conv2d(dim, 4 * dim, 1),
            nn.GELU(),
            nn.Conv2d(4 * dim, dim, 1),
            CA(dim, reduction)
        )

    def forward(self, x):
        return self.block(x) + x
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert data_format in ["channels_last", "channels_first"]
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
class EnhancedGenerator(nn.Module):
    def __init__(self, args):
        super(EnhancedGenerator, self).__init__()
        self.args = args
        self.bottom_width = args.bottom_width
        self.embed_dim = conv_dim = args.gf_dim

        # 第一个尺度 8x8
        self.dec1 = nn.Sequential(
            ConvBlock(in_channels=conv_dim, out_channels=conv_dim),
            MSCA(dim=conv_dim),  # 添加MSCA
            DCAB(dim=conv_dim)   # 添加DCAB
        )
        self.upsample_1 = Upsample(conv_dim, conv_dim // 2)

        # 第二个尺度 16x16
        self.dec2 = nn.Sequential(
            ConvBlock(in_channels=conv_dim//2, out_channels=conv_dim//2),
            MSCA(dim=conv_dim//2),  # 添加MSCA
            DCAB(dim=conv_dim//2)   # 添加DCAB
        )
        self.upsample_2 = Upsample(conv_dim//2, conv_dim // 4)

        # 第三个尺度 32x32
        self.dec3 = nn.Sequential(
            ConvBlock(in_channels=conv_dim//4, out_channels=conv_dim//4),
            MSCA(dim=conv_dim//4),  # 添加MSCA
            DCAB(dim=conv_dim//4)   # 添加DCAB
        )
        self.upsample_3 = Upsample(conv_dim // 4, conv_dim // 8)

        # 第四个尺度 64x64
        self.dec4 = nn.Sequential(
            ConvBlock(in_channels=conv_dim//8, out_channels=conv_dim//8),
            MSCA(dim=conv_dim//8),  # 添加MSCA
            DCAB(dim=conv_dim//8)   # 添加DCAB
        )

    def forward(self, x):
        features = []
        # 输入处理
        x = x.permute(0,2,1).contiguous().view(-1, self.embed_dim,
                                              self.bottom_width, self.bottom_width)

        # 8x8 尺度
        x = self.dec1(x)  # 卷积+MSCA+DCAB处理
        features.append(x)  # 保存第一个尺度的特征

        # 16x16 尺度
        x = self.upsample_1(x)  # 上采样到16x16
        x = self.dec2(x)  # 卷积+MSCA+DCAB处理
        features.append(x)  # 保存第二个尺度的特征

        # 32x32 尺度
        x = self.upsample_2(x)  # 上采样到32x32
        x = self.dec3(x)  # 卷积+MSCA+DCAB处理
        features.append(x)  # 保存第三个尺度的特征

        # 64x64 尺度
        x = self.upsample_3(x)  # 上采样到64x64
        x = self.dec4(x)  # 卷积+MSCA+DCAB处理
        features.append(x)  # 保存第四个尺度的特征

        return features

class Generator(nn.Module):
    def __init__(self, args,upsample=Upsample):
        super(Generator, self).__init__()
        self.args = args
        self.bottom_width = args.bottom_width
        self.embed_dim = conv_dim = args.gf_dim
        self.dec1 = ConvBlock(in_channels=conv_dim, out_channels=conv_dim,norm_fun = args.cnnnorm_type)  # 8*8*128 --> 32*32*256
        self.upsample_1 = upsample(conv_dim, conv_dim // 2)
        self.dec2 = ConvBlock(in_channels=conv_dim//2, out_channels=conv_dim//2,norm_fun = args.cnnnorm_type)  # 16*16*128 --> 32*32*256
        self.upsample_2 = upsample(conv_dim//2, conv_dim // 4)
        self.dec3 = ConvBlock(in_channels=conv_dim//4, out_channels=conv_dim//4,norm_fun = args.cnnnorm_type)  # 32*32*128 --> 32*32*256
        self.upsample_3 = upsample(conv_dim // 4, conv_dim // 8)
        self.dec4 = ConvBlock(in_channels=conv_dim//8, out_channels=conv_dim//8,norm_fun = args.cnnnorm_type)  # 64*64*128 --> 32*32*256
        # self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.embed_dim)

    def forward(self, x):
        features = []
        x = tf2cnn(x)

        #8x8
        x = self.dec1(x)
        features.append(x)


        #16x16
        x = self.upsample_1(x)
        x = self.dec2(x)
        features.append(x)


        #32x32
        x = self.upsample_2(x)
        x = self.dec3(x)
        features.append(x)

        #64x64
        x = self.upsample_3(x)
        x = self.dec4(x)
        features.append(x)

        return features


class Generator_nopos(nn.Module):
    def __init__(self, args,upsample=Upsample):
        super(Generator_nopos, self).__init__()
        self.args = args
        self.bottom_width = args.bottom_width
        self.embed_dim = conv_dim = args.gf_dim
        self.dec1 = ConvBlock(in_channels=conv_dim, out_channels=conv_dim,norm_fun = args.cnnnorm_type)  # 8*8*128 --> 32*32*256
        self.upsample_1 = upsample(conv_dim, conv_dim // 2)
        self.dec2 = ConvBlock(in_channels=conv_dim//2, out_channels=conv_dim//2,norm_fun = args.cnnnorm_type)  # 16*16*128 --> 32*32*256
        self.upsample_2 = upsample(conv_dim//2, conv_dim // 4)
        self.dec3 = ConvBlock(in_channels=conv_dim//4, out_channels=conv_dim//4,norm_fun = args.cnnnorm_type)  # 32*32*128 --> 32*32*256
        self.upsample_3 = upsample(conv_dim // 4, conv_dim // 8)
        self.dec4 = ConvBlock(in_channels=conv_dim//8, out_channels=conv_dim//8,norm_fun = args.cnnnorm_type)  # 64*64*128 --> 32*32*256
        # self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.embed_dim)

    def forward(self, x):
        features = []
        # x = tf2cnn(x)

        #8x8
        x = self.dec1(x)
        features.append(x)


        #16x16
        x = self.upsample_1(x)
        x = self.dec2(x)
        features.append(x)


        #32x32
        x = self.upsample_2(x)
        x = self.dec3(x)
        features.append(x)

        #64x64
        x = self.upsample_3(x)
        x = self.dec4(x)
        features.append(x)

        return features

def tf2cnn(x):
    B, L, C = x.shape
    H = int(math.sqrt(L))
    W = int(math.sqrt(L))
    x = x.transpose(1, 2).contiguous().view(B, C, H, W)
    return x

def cnn2tf(x):
    B,C,H,W = x.shape
    L = H*W
    x = x.flatten(2).transpose(1,2).contiguous()  # B H*W C
    return x,C,H,W

def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W


def bicubic_upsample(x,H,W,up_mode='bicubic'):
    B, N, C = x.size()
    assert N == H*W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = F.interpolate(x, scale_factor=2, mode=up_mode)
    B, C, H, W = x.size()
    x = x.view(-1, C, H*W)
    x = x.permute(0,2,1)
    return x, H, W

def get_norm_fun(norm_fun_type='none'):
    if norm_fun_type == 'BatchNorm':
        norm_fun = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'InstanceNorm':
        norm_fun = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'none':
        norm_fun = lambda x: Identity()
    else:
        raise NotImplementedError('normalization function [%s] is not found' % norm_fun_type)
    return norm_fun


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3,dilation = 1,norm_fun='none'):
        super(ConvBlock, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        norm_fun = get_norm_fun(norm_fun)
        self.conv = nn.Sequential(
            #1
            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(in_channels, out_channels, 3, 1, 0),
            norm_fun(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #2
            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(in_channels, out_channels, 3, 1, 0),
            norm_fun(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=3,dilation = 1,norm_fun='none'):
        super(ResBlock, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        norm_fun = get_norm_fun(norm_fun)
        self.conv = nn.Sequential(
            #1
            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(in_channels, out_channels, 3, 1, 0),
            norm_fun(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #2
            nn.ReflectionPad2d(self.padding),
            nn.Conv2d(in_channels, out_channels, 3, 1, 0),
            norm_fun(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        return self.conv(x) + x


def get_act_fun(act_fun_type='LeakyReLU'):
    if isinstance(act_fun_type, str):
        if act_fun_type == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun_type == 'ReLU':
            return nn.ReLU(inplace=True)
        elif act_fun_type == 'SELU':
            return nn.SELU(inplace=True)
        elif act_fun_type == 'none':
            return nn.Sequential()
        else:
            raise NotImplementedError('activation function [%s] is not found' % act_fun_type)
    else:
        return act_fun_type()

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_fun(norm_fun_type='none'):
    if norm_fun_type == 'BatchNorm':
        norm_fun = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'InstanceNorm':
        norm_fun = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'none':
        norm_fun = lambda x: Identity()
    else:
        raise NotImplementedError('normalization function [%s] is not found' % norm_fun_type)
    return norm_fun