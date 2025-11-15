import torch
import torch.nn as nn
import torch.nn.functional as F
import models.Transformer as Transformer
import models.CNN as CNN
import numpy as np



class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        assert data_format in ["channels_last", "channels_first"]
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            return self.weight[:, None, None] * x + self.bias[:, None, None]


class SGAC(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        self.Conv1 = nn.Conv2d(n_feats, n_feats * 2, 1, 1, 0)
        self.DWConv1 = nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats)
        self.Conv2 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        shortcut = x
        x = self.Conv1(self.norm(x))
        a, x = torch.chunk(x, 2, dim=1)
        x = x * self.DWConv1(a)
        x = self.Conv2(x)
        return x * self.scale + shortcut

class MRFC(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        if n_feats % 3 != 0:
            raise ValueError("n_feats must be divisible by 3 for MRFC.")
        self.norm = LayerNorm(n_feats, data_format='channels_first')
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        # 三路：3/5/7 大核 + 膨胀卷积
        self.LKA7 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 9, 1, (9 // 2) * 4, groups=n_feats // 3, dilation=4),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA5 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, (7 // 2) * 3, groups=n_feats // 3, dilation=3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))
        self.LKA3 = nn.Sequential(
            nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3),
            nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, (5 // 2) * 2, groups=n_feats // 3, dilation=2),
            nn.Conv2d(n_feats // 3, n_feats // 3, 1, 1, 0))

        self.X3 = nn.Conv2d(n_feats // 3, n_feats // 3, 3, 1, 1, groups=n_feats // 3)
        self.X5 = nn.Conv2d(n_feats // 3, n_feats // 3, 5, 1, 5 // 2, groups=n_feats // 3)
        self.X7 = nn.Conv2d(n_feats // 3, n_feats // 3, 7, 1, 7 // 2, groups=n_feats // 3)

        self.proj_first = nn.Conv2d(n_feats, 2 * n_feats, 1, 1, 0)
        self.proj_last  = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        x = self.proj_first(x)
        a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3 = torch.chunk(a, 3, dim=1)
        a = torch.cat([self.LKA3(a_1) * self.X3(a_1),
                       self.LKA5(a_2) * self.X5(a_2),
                       self.LKA7(a_3) * self.X7(a_3)], dim=1)
        x = self.proj_last(x * a) * self.scale + shortcut
        return x

class MKGA(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.LKA = MRFC(n_feats)
        self.LFE = SGAC(n_feats)
    def forward(self, x):
        return self.LFE(self.LKA(x))

class CMSF(nn.Module):
    def __init__(self, dim, reduction=16, use_spectral_norm=False):
        super().__init__()
        self.dim = dim
        self.ln_c = LayerNorm(dim, data_format='channels_first')
        self.ln_t = LayerNorm(dim, data_format='channels_first')
        # H/W 方向的自适应平均池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))   # -> (B,C,H,1)
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))   # -> (B,C,1,W)

        def conv1x1(cin, cout):
            m = nn.Conv2d(cin, cout, 1, 1, 0)
            return nn.utils.spectral_norm(m) if use_spectral_norm else m

        # Q/K: 对单通道相关图做通道扩展与压缩
        self.Q = nn.Sequential(conv1x1(1, dim // reduction), nn.GELU(), conv1x1(dim // reduction, dim), nn.Sigmoid())
        self.K = nn.Sequential(conv1x1(1, dim // reduction), nn.GELU(), conv1x1(dim // reduction, dim), nn.Sigmoid())

        self.V = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            conv1x1(dim, dim)
        )

    def forward(self, F_c, F_t, PhiTb_embed):
        # F_c/F_t: (B,C,H,W) ; PhiTb_embed: (B,C,H,W)（已插值到相同尺度）
        B, C, H, W = F_c.shape
        F_c = self.ln_c(F_c)
        F_t = self.ln_t(F_t)

        # 生成 H/W 方向的 query & key
        Hc = self.pool_h(F_c).squeeze(3)         # (B,C,H)
        Wc = self.pool_w(F_c).squeeze(2)         # (B,C,W)
        Ht = self.pool_h(F_t).squeeze(3)         # (B,C,H)
        Wt = self.pool_w(F_t).squeeze(2)         # (B,C,W)

        A_local  = torch.bmm(Hc.permute(0,2,1), Wc)          # (B,H,W)
        A_global = torch.bmm(Ht.permute(0,2,1), Wt)          # (B,H,W)
        A_local  = self.Q(A_local.unsqueeze(1))              # (B,C,H,W)
        A_global = self.K(A_global.unsqueeze(1))             # (B,C,H,W)

        V_phi = self.V(PhiTb_embed)                          # (B,C,H,W)
        return V_phi * A_local * A_global                    # (B,C,H,W)


class SamplingLayer(nn.Module):
    def __init__(self, n_input, n_output=256):
        super().__init__()
        self.depthwise = nn.Conv2d(n_input, n_input, 3, 2, 1, groups=n_input)
        self.pointwise = nn.Conv2d(n_input, n_output, 1)
        self.conv = nn.Conv2d(n_output, n_output, 3, 2, 1)
    def forward(self, x):
        return self.conv(self.pointwise(self.depthwise(x)))

# ==================== 主网络 CT-MSA ====================
class CT_MSA(nn.Module):
    def __init__(self, args, use_spectral_norm=False):
        super().__init__()
        self.args = args
        self.n_input = args.n_input
        self.bottom_width = args.bottom_width
        self.embed_dim = args.gf_dim  # 基础通道
        self.outdim = int(np.ceil((args.img_size ** 2) // (args.bottom_width ** 2)))


        self.iniconv = nn.Sequential(
            nn.Conv2d(self.n_input, 128, 1, 1, 0), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 1, 1, 0),         nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 1, 1, 0),         nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0),         nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, 1, 1, 0)
        )

        # 可学习测量矩阵 Φ 与 Φᵀ（论文 §3.2，式(1)(2)）
        self.Phi  = nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(self.n_input, 256)))
        self.PhiT = nn.Parameter(torch.nn.init.xavier_normal_(torch.Tensor(256, self.n_input)))

        # 双分支
        self.td = Transformer.Transformer(args)
        self.gs = CNN.Generator(args)

        # 四级 CMSF + MKGA
        self.cmsf_layers = nn.ModuleList([CMSF(dim=self.embed_dim // (2 ** i), reduction=16,
                                               use_spectral_norm=use_spectral_norm) for i in range(4)])
        self.mkga_layers = nn.ModuleList([
            MKGA(n_feats=self._adjust_channels_for_mkga(self.embed_dim // (2 ** i))) for i in range(4)
        ])

        # 融合权重 ψ_i（1×1 conv 输出单通道 logits）
        self.psi_layers = nn.ModuleList([
            nn.Conv2d(self.embed_dim // (2 ** i), 1, kernel_size=1) for i in range(4)
        ])

        # 重建头（输出 n_input，做与 Φᵀb 的残差）
        self.recon_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, 3, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.embed_dim, self.n_input, 1)
        )

    def _adjust_channels_for_mkga(self, channels):
        return channels if channels % 3 == 0 else ((channels // 3) + 1) * 3

    # 将块级结果拼回整图（论文 §3.5 BlockAssembly）
    def together(self, inputs, S, H, L):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0)
        inputs = inputs.squeeze(1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=H * S, dim=0), dim=2)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=S, dim=0), dim=1)
        inputs = inputs.unsqueeze(1)
        return inputs

    def forward(self, inputs):
        # ---------- 分块 ----------
        H_blocks = int(inputs.shape[2] / 64)
        L_blocks = int(inputs.shape[3] / 64)
        S_batch  = inputs.shape[0]

        x_blocks = torch.squeeze(inputs, dim=1)
        x_blocks = torch.cat(x_blocks.chunk(H_blocks, dim=1), dim=0)
        x_blocks = torch.cat(x_blocks.chunk(L_blocks, dim=2), dim=0)
        x_blocks = x_blocks.unsqueeze(1)

        # ---------- 采样 ----------
        PhiWeight = self.Phi.contiguous().view(self.n_input, 1, 16, 16)
        y = F.conv2d(x_blocks, PhiWeight, padding=0, stride=16, bias=None)

        PhiTWeight = self.PhiT.contiguous().view(256, self.n_input, 1, 1)
        PhiTb = F.conv2d(y, PhiTWeight, padding=0, bias=None)
        PhiTb = torch.nn.PixelShuffle(16)(PhiTb)  # (B, C= self.n_input, 64, 64)

        # ---------- 初始特征（进入双分支） ----------
        F_ini = self.iniconv(y)                     # (B,512,?,?)
        F_ini = torch.nn.PixelShuffle(2)(F_ini)     # 下采样对齐后提高分辨率
        # 展平为序列以兼容外部分支（如有需要）
        x_seq = F_ini.flatten(2).transpose(1, 2).contiguous()

        # 取四级特征（要求 gs/td 返回 list[4]）
        gsfeatures = self.gs(x_seq)    # list of 4
        tsfeatures = self.td(x_seq)    # list of 4

        # ---------- 逐级 CMSF + MKGA ----------
        enhanced_feats = []
        up_targets = None
        for i, (cmsf, mkga) in enumerate(zip(self.cmsf_layers, self.mkga_layers)):
            Fc = gsfeatures[i]
            Ft = tsfeatures[i]

            # 保证 4D
            def to_4d(feat):
                if feat.ndim == 3:
                    B, L, C = feat.shape
                    HW = int(np.sqrt(L))
                    feat = feat.transpose(1, 2).reshape(B, C, HW, HW)
                return feat
            Fc = to_4d(Fc)
            Ft = to_4d(Ft)

    
            if up_targets is None:
                up_targets = Fc.shape[-2:]


            PhiTb_scaled = F.interpolate(PhiTb, size=Fc.shape[-2:], mode='bilinear', align_corners=False)

            # CMSF（带先验注入）→ MKGA
            fused = cmsf(Fc, Ft, PhiTb_scaled)
            # 若通道非3整除，进入MKGA前用1×1对齐，出后再拉回
            C_orig = fused.shape[1]
            C_mkga = self._adjust_channels_for_mkga(C_orig)
            if C_orig != C_mkga:
                proj_in  = nn.Conv2d(C_orig, C_mkga, 1).to(fused.device)
                proj_out = nn.Conv2d(C_mkga, C_orig, 1).to(fused.device)
                fused = proj_out(mkga(proj_in(fused)))
            else:
                fused = mkga(fused)

            enhanced_feats.append(fused)

        # ---------- 多尺度 softmax 融合 ----------
        up_feats = [F.interpolate(f, size=up_targets, mode='bilinear', align_corners=False) for f in enhanced_feats]
        logits   = [self.psi_layers[i](up_feats[i]) for i in range(4)]       # [(B,1,H,W)]*4
        logits   = torch.cat(logits, dim=1)                                  # (B,4,H,W)
        weights  = torch.softmax(logits, dim=1)                               # 逐尺度 softmax

        F_agg = 0.
        for i in range(4):
            F_agg = F_agg + up_feats[i] * weights[:, i:i+1, ...]             # (B,C0,H0,W0)

-

        if F_agg.shape[1] != self.embed_dim:
            adj = nn.Conv2d(F_agg.shape[1], self.embed_dim, 1).to(F_agg.device)
            F_agg = adj(F_agg)
        X_patch = self.recon_head(F_agg) + F.interpolate(PhiTb, size=up_targets, mode='bilinear', align_corners=False)

        # ---------- 拼回整图 ----------
        merge_output = self.together(X_patch, S_batch, H_blocks, L_blocks)
        merge_PhiTb  = self.together(PhiTb,  S_batch, H_blocks, L_blocks)

        return merge_output, merge_PhiTb, up_feats, PhiTb

