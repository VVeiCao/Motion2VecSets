import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath
import numpy as np
from .condition_encoder import Scan2latent, Scan2latent_deform_timeatt
from util.console import print

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        if context_dim is None:
            context_dim = query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)

        if context is None:
            context = x

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class CrossAttentionT(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        if context_dim is None:
            context_dim = query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)

        if context is None:
            context = x

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(
            t, 'b t n (h d) -> (b h) t n d', h=h), (q, k, v))

        sim = torch.einsum('b t i d, b t j d -> b t i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b t i j, b t j d -> b t i d', attn, v)
        out = rearrange(out, '(b h) t n d -> b t n (h d)', h=h)
        return self.to_out(out)

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd, dim):
        super().__init__()
        self.dim = dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(timestep)
        scale, shift = torch.chunk(emb, 2, dim=self.dim)
        x = self.layernorm(x) * (1 + scale) + shift
        return x

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True, withT=False, is_deform=False, is_corr=False):
        super().__init__()
        self.withT = withT
        self.is_deform = is_deform
        self.is_corr = is_corr
        

        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
  
        self.norm1 = AdaLayerNorm(dim, 3 if self.is_deform else 2)
        self.norm2 = AdaLayerNorm(dim, 3 if self.is_deform else 2)
        self.norm4 = AdaLayerNorm(dim, 3 if self.is_deform else 2)
        
        self.checkpoint = checkpoint

        init_values = 0
        drop_path = 0.1


        self.ls1 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.ls2 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.ls4 = LayerScale(
            dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path4 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        if self.is_deform:
            self.attn1 = CrossAttentionT(
                query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
            self.attn2 = CrossAttentionT(query_dim=dim, context_dim=context_dim,
                heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none

            
            self.attn5 = CrossAttentionT(query_dim=dim, context_dim=context_dim,
                heads=n_heads, dim_head=d_head, dropout=dropout)
            self.norm5 = AdaLayerNorm(dim, 3 if self.is_deform else 2)

            self.ls5 = LayerScale(
                dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path5 = DropPath(
                drop_path) if drop_path > 0. else nn.Identity()
        
            if not self.is_corr:
                self.attn6 = CrossAttentionT(query_dim=dim, context_dim=context_dim,
                    heads=n_heads, dim_head=d_head, dropout=dropout)
                self.norm6 = AdaLayerNorm(dim, 3 if self.is_deform else 2)
                self.ls6 = LayerScale(
                    dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path6 = DropPath(
                    drop_path) if drop_path > 0. else nn.Identity()
            
            if self.withT:
                self.attn3 = CrossAttentionT(query_dim=dim, context_dim=context_dim,
                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
                self.norm3 = AdaLayerNorm(dim, 3 if self.is_deform else 2)
                self.ls3 = LayerScale(
                    dim, init_values=init_values) if init_values else nn.Identity()
                self.drop_path3 = DropPath(
                    drop_path) if drop_path > 0. else nn.Identity()
        else:
            self.attn1 = CrossAttention(
                query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
            self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none

    def forward(self, x, t, cond=None, shape_cond=None, cond_src_emb=None, cond_tgt_emb=None): # torch.Size([34, 64, 512])
        x = self.drop_path1(self.ls1(self.attn1(self.norm1(x, t)))) + x # torch.Size([34, 64, 512])
        if self.is_deform:
            if not self.is_corr: # wo_corr
                x = self.drop_path2(self.ls2(self.attn2(self.norm2(x, t), context=shape_cond))) + x # torch.Size([34, 64, 512])
                x = self.drop_path5(self.ls5(self.attn5(self.norm5(x, t), context=cond_src_emb))) + x
                x = self.drop_path6(self.ls6(self.attn6(self.norm6(x, t), context=cond_tgt_emb))) + x
            else: # w_corr
                x = self.drop_path2(self.ls2(self.attn2(self.norm2(x, t), context=shape_cond))) + x # torch.Size([34, 64, 512])
                x = self.drop_path5(self.ls5(self.attn5(self.norm5(x, t), context=cond))) + x
        
            if self.withT:
                x = x.permute(0, 2, 1, 3).contiguous()
                t = t.permute(0, 2, 1, 3).contiguous()
                x = self.drop_path3(self.ls3(self.attn3(self.norm3(x, t)))) + x
                x = x.permute(0, 2, 1, 3).contiguous()
                t = t.permute(0, 2, 1, 3).contiguous()
        else:
            x = self.drop_path2(self.ls2(self.attn2(self.norm2(x, t), context=cond))) + x # torch.Size([34, 64, 512])
            
        x = self.drop_path4(self.ls4(self.ff(self.norm4(x, t)))) + x
        return x

class LatentArrayTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(self, in_channels, t_channels, n_heads, d_head,
                 depth=1, latent_dim=8, dropout=0., context_dim=None, out_channels=None, withT=False, is_deform=False, is_corr=False):
        super().__init__()
        
        self.withT = withT
        self.is_deform = is_deform
        self.is_corr = is_corr
        
        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self.t_channels = t_channels

        self.proj_in = nn.Linear(in_channels, inner_dim, bias=False)
        if self.is_deform:
            self.shape_cond_proj = nn.Linear(latent_dim, inner_dim, bias=False)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, withT=withT, is_deform=is_deform, is_corr=is_corr)
                for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(inner_dim)

        if out_channels is None:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels, bias=False))
        else:
            self.num_cls = out_channels
            self.proj_out = zero_module(nn.Linear(inner_dim, out_channels, bias=False))

        self.context_dim = context_dim

        self.map_noise = PositionalEmbedding(t_channels)

        self.map_layer0 = nn.Linear(in_features=t_channels, out_features=inner_dim)
        self.map_layer1 = nn.Linear(in_features=inner_dim, out_features=inner_dim)


    def forward(self, x, t, cond=None, shape_cond=None, cond_src_emb=None, cond_tgt_emb=None):
        if self.is_deform:
            t_emb = self.map_noise(t)[:, None, None]
        else:
            t_emb = self.map_noise(t)[:, None]
        t_emb = F.silu(self.map_layer0(t_emb))
        t_emb = F.silu(self.map_layer1(t_emb))

        x = self.proj_in(x)
        if self.is_deform:
            shape_cond = self.shape_cond_proj(shape_cond)
        for block in self.transformer_blocks:
            x = block(x, t_emb, cond=cond, shape_cond=shape_cond, cond_src_emb=cond_src_emb, cond_tgt_emb=cond_tgt_emb)
        
        x = self.norm(x)

        x = self.proj_out(x)
        return x

def edm_sampler(
    net, latents, cond = None, shape_cond=None, cond_src_emb=None, cond_tgt_emb=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    #Get cond embed
    #cond_emb = net.get_cond_embed(cond, latents.device)

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, cond=cond, shape_cond=shape_cond, cond_src_emb=cond_src_emb, cond_tgt_emb=cond_tgt_emb).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, cond= cond, shape_cond=shape_cond, cond_src_emb=cond_src_emb, cond_tgt_emb=cond_tgt_emb).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

def random_masking(self, x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=1):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, inputs, cond=None, shape_cond=None, cond_src_emb=None, cond_tgt_emb=None, augment_pipe=None):
        if inputs.dim() == 3:
            n_shape = [inputs.shape[0], 1, 1]
        elif inputs.dim() == 4:
            n_shape = [inputs.shape[0], 1, 1, 1]
        # print(n_shape)
        rnd_normal = torch.randn(n_shape, device=inputs.device) # torch.Size([34, 64, 32])
        # rnd_normal = torch.randn([1, 1, 1], device=inputs.device).repeat(inputs.shape[0], 1, 1)
        # print(rnd_normal.shape)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(inputs) if augment_pipe is not None else (inputs, None)
        # print(y.shape)
        # raise
        n = torch.randn_like(y) * sigma

        D_yn = net(y + n, sigma, cond =cond, shape_cond=shape_cond, cond_src_emb=cond_src_emb, cond_tgt_emb=cond_tgt_emb)
        loss = weight * ((D_yn - y) ** 2)
        return loss.mean()

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

class EDMPrecond(torch.nn.Module):
    def __init__(self,
        n_latents = 512,
        shape_dim= 8,
        channels = 8, 
        use_fp16 = False,
        sigma_min = 0,
        sigma_max = float('inf'),
        sigma_data  = 1,
        n_heads = 8,
        d_head = 64,
        depth = 12,
        cond_type='image',
        cond_num_inputs=2048,
        withT=False,
        is_deform=False,
        is_corr=False
    ):
        super().__init__()
        self.n_latents = n_latents
        self.channels = channels
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        
        self.withT = withT
        self.is_deform = is_deform
        self.is_corr = is_corr
        

        self.model = LatentArrayTransformer(in_channels=channels, t_channels=256, n_heads=n_heads, d_head=d_head, depth=depth, latent_dim=shape_dim, withT=withT, is_deform=is_deform, is_corr=is_corr)

        if self.is_deform and self.is_corr:
            self.cond_enc = Scan2latent_deform_timeatt(dim=n_heads*d_head, num_inputs=cond_num_inputs, num_latents=n_latents)
        else:
            self.cond_enc = Scan2latent(dim=n_heads*d_head, num_inputs=cond_num_inputs, num_latents=n_latents)
    
    def get_cond_embed(self, cond, device):
        if isinstance(cond, torch.Tensor):
            cond = cond.to(torch.float32)
            cond_emb = self.cond_enc(cond)
        else:
            cond_emb = self.cond_enc(cond, device)
            cond_emb = cond_emb.to(torch.float32)
        if len(cond_emb.shape) == 2:
            cond_emb = cond_emb[:, None, :]
        return cond_emb
    
    def get_cond_embed_cond(self, cond1, cond2, device):
        if cond2 is not None:
            if isinstance(cond1, torch.Tensor):
                cond1 = cond1.to(torch.float32)
                cond2 = cond2.to(torch.float32)
                cond_emb = self.cond_enc(cond1, cond2)
            else:
                cond_emb = self.cond_enc(cond1, cond2, device)
                cond_emb = cond_emb.to(torch.float32)
        else:
            if isinstance(cond1, torch.Tensor):
                cond1 = cond1.to(torch.float32)
                cond_emb = self.cond_enc(cond1)
            else:
                cond_emb = self.cond_enc(cond1, device)
                cond_emb = cond_emb.to(torch.float32)
            if len(cond_emb.shape) == 2:
                cond_emb = cond_emb[:, None, :]
        return cond_emb

    def forward(self, x, sigma, cond=None, shape_cond=None, cond_src_emb=None, cond_tgt_emb=None, force_fp32=False, **model_kwargs):
        cond_emb = None
        if self.is_deform:
            if not self.is_corr:
                B,T,N,C = cond_src_emb.shape
                cond_src_emb = cond_src_emb.view(-1,N,C)
                cond_tgt_emb = cond_tgt_emb.view(-1,N,C)
                
                cond_src_emb = self.get_cond_embed(cond_src_emb, x.device)
                cond_tgt_emb = self.get_cond_embed(cond_tgt_emb, x.device)
            
                _,M,L = cond_src_emb.shape
                cond_src_emb = cond_src_emb.view(B,T,M,L)
                cond_tgt_emb = cond_tgt_emb.view(B,T,M,L)
            else:
                cond_emb = self.get_cond_embed_cond(cond_src_emb, cond_tgt_emb, x.device)
        else:
            cond_emb = self.get_cond_embed(cond, x.device)

        
        x = x.to(torch.float32)
        
        if x.dim() == 3:
            sigma = sigma.to(torch.float32).reshape(-1, 1, 1)
        elif x.dim() == 4:
            sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        else:
            raise ValueError(f'Unsupported input shape: {x.shape}')
        
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        if self.is_deform:
            F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), cond=cond_emb, cond_src_emb=cond_src_emb, cond_tgt_emb=cond_tgt_emb, shape_cond=shape_cond, **model_kwargs)
        else:
            F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), cond=cond_emb, **model_kwargs)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
    @torch.no_grad()
    def sample(self, device, cond=None, shape_cond=None, cond_src_emb=None, cond_tgt_emb=None, batch_seeds=None, random=False):
        if self.is_deform:
            if not isinstance(cond_src_emb, list):
                batch_size = len(cond_src_emb)
                time_steps = cond_src_emb.shape[1]
                cond_src_emb = cond_src_emb.to(device)
                cond_tgt_emb = cond_tgt_emb.to(device)
            else:
                batch_size = len(cond_src_emb)
                time_steps = cond_src_emb.shape[1]
        else:
            if not isinstance(cond, list):
                batch_size = len(cond)
                cond = cond.to(device)
            else:
                batch_size = len(cond)
        
        if batch_seeds is None: 
            if random:
                batch_seeds = torch.randint(1<<32, size=(batch_size,))
            else:
                batch_seeds = torch.arange(batch_size)
        
        rnd = StackedRandomGenerator(device, batch_seeds)
        
        if self.is_deform:
            latents = rnd.randn([batch_size, time_steps, self.n_latents, self.channels], device=device)
        else:
            latents = rnd.randn([batch_size, self.n_latents, self.channels], device=device)

        return edm_sampler(self, latents, 
                           cond=cond,
                           shape_cond=shape_cond,
                            cond_src_emb=cond_src_emb,
                            cond_tgt_emb=cond_tgt_emb,
                            randn_like=rnd.randn_like)
        
def kl_d512_m512_l8_surf300_edm():
    model = EDMPrecond(n_latents=512, channels=8, cond_type='surf', cond_num_inputs=300)
    return model

def kl_d512_m512_l8_surf512_edm():
    model = EDMPrecond(n_latents=512, channels=8, cond_type='surf', cond_num_inputs=512)
    return model

def de_kl_d512_m512_l32_surf300_edm(withT= False, is_corr=True):
    model = EDMPrecond(n_latents=512, channels=32, cond_type='surf', cond_num_inputs=300, is_deform=True, is_corr=is_corr, withT=withT)
    return model

def de_kl_d512_m512_l32_surf512_edm(withT= False, is_corr=True):
    model = EDMPrecond(n_latents=512, channels=32, cond_type='surf', cond_num_inputs=512, is_deform=True, is_corr=is_corr, withT=withT)
    return model