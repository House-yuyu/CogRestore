# -*- coding: utf-8 -*-
from collections import OrderedDict
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, Normalize

from basicsr.archs.restormer_arch import (
    TransformerBlock,
    Downsample,
    Upsample,
    OverlapPatchEmbed
)

from einops import rearrange


# ==================== Atmospheric Scattering Prior Extraction ====================
class AtmosphericScatteringPriorExtraction(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.atmosphere_light_estimator = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, channels, 1),
            nn.Sigmoid()
        )
        self.transmission_estimator = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.physical_constraint = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input_image):
        B, C, H, W = input_image.shape
        atmosphere_light = self.atmosphere_light_estimator(input_image)
        atmosphere_light = atmosphere_light.expand(-1, -1, H, W)
        transmission = self.transmission_estimator(input_image)
        transmission = torch.clamp(transmission, 0.1, 1.0)

        dark_channel = torch.min(input_image, dim=1, keepdim=True)[0]
        depth_prior = 1.0 - dark_channel
        refined_depth = self.physical_constraint(depth_prior)
        beta = 0.2
        transmission_physical = torch.exp(-beta * refined_depth)
        transmission = 0.6 * transmission + 0.4 * transmission_physical

        clear_image_est = (input_image - atmosphere_light) / transmission + atmosphere_light
        residual_prior = input_image - clear_image_est
        return residual_prior


# ==================== SimpleGate & LayerNorm2d ====================
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=[0,2,3]), grad_output.sum(dim=[0,2,3]), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


# ==================== CHIMB_RS_Spectral ====================
class CHIMB_RS_Spectral(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        
        # Spatial Branch
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)
        self.dwconv3 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        self.dwconv5 = nn.Conv2d(dw_channel, dw_channel, 5, padding=2, groups=dw_channel, bias=True)
        self.dwconv7 = nn.Conv2d(dw_channel, dw_channel, 7, padding=3, groups=dw_channel, bias=True)
        self.fuse_scale = nn.Conv2d(dw_channel * 3, dw_channel, 1, bias=True)
        self.sg = SimpleGate()
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)

        # Spectral Prior Branch
        self.freq_proj = nn.Conv2d(c, c, 1, bias=True)
        self.freq_mlp = nn.Sequential(
            nn.Conv2d(c, c // 4, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 4, c, 1, bias=True)
        )

        # Adaptive Fusion
        self.fusion_gate = nn.Sequential(
            nn.Conv2d(c * 2, c, 1, bias=True),
            nn.Sigmoid()
        )

        # FFN
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel, ffn_channel, 3, padding=1, groups=ffn_channel, bias=True)
        self.conv6 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)

        # Norm & Residual
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def extract_spectral_prior(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(x, norm='ortho')
        mag = torch.abs(x_fft)
        pha = torch.angle(x_fft)

        h_grid = torch.arange(H, device=x.device).view(1, 1, H, 1).expand(1, 1, H, W//2+1)
        w_grid = torch.arange(W//2+1, device=x.device).view(1, 1, 1, W//2+1).expand(1, 1, H, W//2+1)
        center_h, center_w = H // 2, W // 4
        dist = ((h_grid - center_h) ** 2 + (w_grid - center_w) ** 2) ** 0.5
        lp_filter = torch.exp(-dist / (H * 0.5))

        mag_filtered = mag * lp_filter
        real = mag_filtered * torch.cos(pha)
        imag = mag_filtered * torch.sin(pha)
        x_filtered_fft = torch.complex(real, imag)
        x_recon = torch.fft.irfft2(x_filtered_fft, s=(H, W), norm='ortho')
        return x_recon

    def forward(self, inp):
        x = self.norm1(inp)

        x_spatial = self.conv1(x)
        x3 = self.dwconv3(x_spatial)
        x5 = self.dwconv5(x_spatial)
        x7 = self.dwconv7(x_spatial)
        x_multi = torch.cat([x3, x5, x7], dim=1)
        x_multi = self.fuse_scale(x_multi)
        x_spatial_out = self.conv3(self.sg(x_multi))

        x_freq = self.freq_proj(x)
        x_spectral = self.extract_spectral_prior(x_freq)
        x_spectral = self.freq_mlp(x_spectral)

        fuse_input = torch.cat([x_spatial_out, x_spectral], dim=1)
        weight = self.fusion_gate(fuse_input)
        x_fused = weight * x_spatial_out + (1 - weight) * x_spectral

        x_fused = self.dropout1(x_fused)
        y = inp + x_fused * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x1, x2 = self.conv5(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.conv6(x)
        x = self.dropout2(x)

        return y + x * self.gamma


# ==================== RIR with CHIMB_RS_Spectral ====================
class RIR_CHIMB(nn.Module):
    def __init__(self, n_feats, n_blocks):
        super(RIR_CHIMB, self).__init__()
        module_body = [CHIMB_RS_Spectral(n_feats) for _ in range(n_blocks)]
        module_body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True))
        self.module_body = nn.Sequential(*module_body)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.module_body(x)
        res += x
        return self.relu(res)


# ==================== Helper Conv Block ====================
class convd(nn.Module):
    def __init__(self, inputchannel, outchannel, kernel_size, stride):
        super(convd, self).__init__()
        self.padding = nn.ReflectionPad2d(kernel_size // 2)
        self.conv = nn.Conv2d(inputchannel, outchannel, kernel_size, stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(self.padding(x))
        return self.relu(x)


class res_ch_CHIMB(nn.Module):
    def __init__(self, n_feats, out_ch=None, blocks=2):
        super(res_ch_CHIMB, self).__init__()
        self.match_channel = out_ch
        self.conv_init1 = convd(3, n_feats // 2, 3, 1)
        self.conv_init2 = convd(n_feats // 2, n_feats, 3, 1)
        self.extra = RIR_CHIMB(n_feats, n_blocks=blocks)
        if out_ch is not None:
            self.conv_out = convd(n_feats, out_ch, 3, 1)

    def forward(self, x):
        x = self.conv_init2(self.conv_init1(x))
        x = self.extra(x)
        if self.match_channel is not None:
            x = self.conv_out(x)
        return x


# ==================== CrossAttention & Modulation ====================
class CrossAttention_LDM(nn.Module):
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x, context):
        h = self.heads
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class Modulation2D(nn.Module):
    def __init__(self, inChannels):
        super(Modulation2D, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, inChannels, 3, padding=1)
        self.conv2 = nn.Conv2d(inChannels, inChannels, 3, padding=1)
        self.conv_gama = nn.Conv2d(inChannels, inChannels, 1)
        self.conv_beta = nn.Conv2d(inChannels, inChannels, 1)
        self.act = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.act(self.conv1(x))
        out = self.act(self.conv2(out))
        gama = self.sigmoid(self.conv_gama(out))
        beta = self.conv_beta(out)
        return gama, beta


# ==================== Prompt Fuser (NEW) ====================
class PromptFuser(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.attn = CrossAttention_LDM(query_dim=dim, context_dim=dim, heads=8)
        self.norm = nn.LayerNorm(dim)

    def forward(self, text_emb, img_emb):
        # text_emb: [B, 1, D], img_emb: [B, 1, D]
        fused = self.attn(self.norm(text_emb), img_emb) + text_emb
        return fused


# ==================== Decoder Block ====================
class DecoderBlock(nn.Module):
    def __init__(self, in_dim, heads, ffn_expansion_factor, bias, LayerNorm_type, num_blocks,
                 prompt_dim=512, rcp_dim=48):
        super(DecoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.cross_att_text = CrossAttention_LDM(query_dim=in_dim, context_dim=prompt_dim)
        self.cross_att_img = CrossAttention_LDM(query_dim=in_dim, context_dim=prompt_dim)
        self.fuse_rcp = Modulation2D(inChannels=in_dim)
        self.RestormerBlock = nn.Sequential(*[
            TransformerBlock(dim=in_dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks)
        ])

    def forward(self, x, prompt, rcp_feature, iter):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if prompt.ndim == 2:
            prompt = prompt.unsqueeze(0)
        x = self.cross_att_text(self.norm1(x), prompt) + x
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if rcp_feature is not None:
            gamma, beta = self.fuse_rcp(rcp_feature)
            x = x + gamma * x + beta

        return self.RestormerBlock(x)


# ==================== Main Model: RS_AiOIR ====================
class RS_AiOIR0203(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 model_clip=None,
                 num_vector=8,
                 iter_times=2):
        super(RS_AiOIR0203, self).__init__()

        self.model_clip = model_clip
        self.model_clip.eval()
        self.iter_times = iter_times

        # 可学习的渐进混合权重（每轮一个）
        self.prompt_blend_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(iter_times)
        ])

        self.atmospheric_extractor = AtmosphericScatteringPriorExtraction(channels=3)

        self.num_vector = num_vector
        if self.num_vector != 0:
            learnable_vector = torch.empty(self.num_vector, 512)
            nn.init.normal_(learnable_vector, std=0.02)
            self.learnable_vector = nn.Parameter(learnable_vector)
            self.deg_aware = nn.Sequential(
                nn.Linear(512, 512 // 16),
                nn.LayerNorm(512 // 16),
                nn.ReLU(inplace=True),
                nn.Linear(512 // 16, 512),
            )

        self.clip_input_preprocess = Compose([
            Resize([224, 224]),
            Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                      std=(0.26862954, 0.26130258, 0.27577711))
        ])

        # Encoder
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim * 2), num_heads=heads[1],
            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim * 4), num_heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim * 4))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim * 8), num_heads=heads[3],
            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_blocks[3])])

        # Decoder
        self.up4_3 = Upsample(int(dim * 8))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 8), int(dim * 4), 1, bias=bias)
        self.decoder_level3 = DecoderBlock(in_dim=dim * 4, heads=heads[2],
            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
            LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[2] // 2)

        self.up3_2 = Upsample(int(dim * 4))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 4), int(dim * 2), 1, bias=bias)
        self.decoder_level2 = DecoderBlock(in_dim=dim * 2, heads=heads[1],
            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
            LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[1] // 2)

        self.up2_1 = Upsample(int(dim * 2))
        self.decoder_level1 = DecoderBlock(in_dim=dim * 2, heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor, bias=bias,
            LayerNorm_type=LayerNorm_type, num_blocks=num_blocks[0] // 2)

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim * 2), num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
            for _ in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim * 2), out_channels, 3, padding=1, bias=bias)

        # RCP Extractor
        self.rcp_extractor = res_ch_CHIMB(n_feats=dim)
        self.rcp_down1 = Downsample(dim)
        self.rcp_down2 = Downsample(dim * 2)
        self.rcp_ch = nn.Conv2d(dim, dim * 2, 3, padding=1)

        # Prompt Fusion
        self.prompt_fuser = PromptFuser(dim=512)

    @torch.no_grad()
    def get_text_feature(self, text):
        return self.model_clip.encode_text(text)

    @torch.no_grad()
    def get_img_feature(self, img):
        img_input_clip = self.clip_input_preprocess(img)
        return self.model_clip.encode_image(img_input_clip)

    def forward(self, imgs, texts):
        text_feat = self.get_text_feature(texts)  # [B, 512]
        lq_clip_feat = self.get_img_feature(imgs)  # [B, 512]

        # Encoder
        inp_enc_level1 = self.patch_embed(imgs)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_li = []
        prev_out = imgs
        rcp_level1 = rcp_level2 = rcp_level3 = None

        for i in range(self.iter_times):
            # 动态退化向量（基于当前输入）
            degware_vector = self.learnable_vector.repeat(lq_clip_feat.shape[0], 1, 1) + \
                             self.deg_aware(lq_clip_feat).unsqueeze(1).repeat(1, self.num_vector, 1)

            if i == 0:
                # 第一轮：仅用文本
                blended_prompt = text_feat.unsqueeze(1)  # [B, 1, 512]
            else:
                # 后续轮：融合文本 + 上一轮图像特征
                hq_feat = self.get_img_feature(prev_out.detach())  # [B, 512]
                hq_feat = hq_feat.unsqueeze(1)  # [B, 1, 512]
                text_emb = text_feat.unsqueeze(1)  # [B, 1, 512]
                fused_text = self.prompt_fuser(text_emb, hq_feat)  # [B, 1, 512]

                alpha = torch.sigmoid(self.prompt_blend_weights[i])
                blended_prompt = alpha * fused_text + (1 - alpha) * text_emb

            prompt = torch.cat([degware_vector, blended_prompt], dim=1)  # [B, N+1, 512]

            # Decode
            out_dec_level3 = self.decoder_level3(inp_dec_level3, prompt, rcp_level3, iter=i)
            inp_dec_level2 = self.up3_2(out_dec_level3)
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
            inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
            out_dec_level2 = self.decoder_level2(inp_dec_level2, prompt, rcp_level2, iter=i)

            inp_dec_level1 = self.up2_1(out_dec_level2)
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
            out_dec_level1 = self.decoder_level1(inp_dec_level1, prompt, rcp_level1, iter=i)

            out_dec_level1 = self.refinement(out_dec_level1)
            out_img = self.output(out_dec_level1) + imgs

            # 更新 RCP（仅第0轮）
            if i == 0:
                ch_res_out = self.atmospheric_extractor(out_img)
                rcp_feature = self.rcp_extractor(ch_res_out)
                rcp_level1 = self.rcp_ch(rcp_feature)
                rcp_level2 = self.rcp_down1(rcp_feature)
                rcp_level3 = self.rcp_down2(rcp_level2)

            out_li.append(out_img)
            prev_out = out_img

        out = torch.stack(out_li, dim=0).mean(dim=0)
        return out, out_li


# ==================== Test ====================
if __name__ == '__main__':
    from clip import clip
    model, preprocess = clip.load("ViT-B/32", device='cpu', jit=False)
    model.float()
    for p in model.parameters():
        p.requires_grad = False

    net = RS_AiOIR0203(model_clip=model, num_vector=8, num_blocks=[4,6,6,8], iter_times=2)
    input_img = torch.randn(1, 3, 256, 256)
    tokenized_text = torch.randint(0, 3000, (1, 77), dtype=torch.int32)

    from thop import profile
    print("---- model complexity evaluate by thop profile----")
    flops, params = profile(net, inputs=(input_img, tokenized_text), report_missing=True)
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))            # 
    print("params=", str(params / 1e6) + '{}'.format("M"))          # 
    print('\n')

    print("---- model complexity evaluate by thop profile----")
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"), end='\t')      # 230.42 G
    print("params=", str(params / 1e6) + '{}'.format("M"))              # 


    print("---- model complexity evaluate by customized code ----")
    n_param = sum([p.nelement() for p in net.parameters()])             # 所有参数数量
    n_param_train = sum([p.nelement() for p in net.parameters() if p.requires_grad])  # 只计算参与更新的参数数量
    print('Total params:', str(n_param / 1e6) + '{}'.format("M"))
    print('Tranable params:', str(n_param_train / 1e6) + '{}'.format("M"))
    print('\n')

    out, out_li = net(input_img, tokenized_text)
    print("Output shape:", out.shape)  # [1, 3, 256, 256]