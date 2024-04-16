from functools import wraps
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from torch_cluster import fps
from timm.models.layers import DropPath
import os
from util.console import print

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        return embed


class DiagonalGaussianDistribution(object):
    def __init__(self, mean, logvar, deterministic=False):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.mean.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.mean.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class AutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim = 1,
        num_inputs = 2048,
        num_latents = 512,
        heads = 8,
        dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False
    ):
        super().__init__()

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

    def encode(self, pc):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs
        
        ###### fps
        flattened = pc.view(B*N, D)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened

        ratio = 1.0 * self.num_latents / self.num_inputs

        idx = fps(pos, batch, ratio=ratio)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        ######

        sampled_pc_embeddings = self.point_embed(sampled_pc) # positional embedding

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None) + sampled_pc_embeddings
        x = cross_ff(x) + x

        return x


    def decode(self, x, queries):

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        return self.to_outputs(latents)

    def forward(self, pc, queries):
        x = self.encode(pc)

        o = self.decode(x, queries).squeeze(-1)

        return {'logits': o}


class AutoEncoder_deform(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim = 1,
        num_inputs = 2048,
        num_latents = 512,
        heads = 8,
        dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False
    ):
        super().__init__()

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=int(dim//2))

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

    def encode(self, pc, pc2):
        # pc: B x N x 3
        B, N, D = pc.shape
        assert N == self.num_inputs
        _, N2, D2 = pc2.shape
        assert N2 == self.num_inputs
        
        ###### fps
        flattened = pc.view(B*N, D)
        flattened2 = pc2.view(B*N, D)
        pos = flattened
        pos2 = flattened2
        
        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)
        ratio = 1.0 * self.num_latents / self.num_inputs
        idx = fps(pos, batch, ratio=ratio)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)
        sampled_pc2 = pos2[idx]
        sampled_pc2 = sampled_pc2.view(B, -1, 3)
        ######

        sampled_pc_embeddings = self.point_embed(sampled_pc)
        sampled_pc_embeddings2 = self.point_embed(sampled_pc2)
        pc_embeddings = self.point_embed(pc)
        pc_embeddings2 = self.point_embed(pc2)
        sampled_pc_embeddings_concat = torch.cat([sampled_pc_embeddings, sampled_pc_embeddings2], dim=-1).contiguous()
        pc_embeddings_concat = torch.cat([pc_embeddings, pc_embeddings2], dim=-1).contiguous()

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(sampled_pc_embeddings_concat, context = pc_embeddings_concat, mask = None) + sampled_pc_embeddings_concat
        x = cross_ff(x) + x

        return x


    def decode(self, x, queries):

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        return self.to_outputs(latents)

    def forward(self, pc, pc2, queries):
        x = self.encode(pc, pc2)

        d = self.decode(x, queries)

        return {'deformed': d}


class KLAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim = 1,
        num_inputs = 2048,
        num_latents = 512,
        latent_dim = 64,
        heads = 8,
        dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False
    ):
        super().__init__()

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=dim)

        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    def encode(self, pc):
        B, N, D = pc.shape
        self.num_inputs = N
        flattened = pc.view(B*N, D)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened

        ratio = 1.0 * self.num_latents / self.num_inputs

        idx = fps(pos, batch, ratio=ratio)

        sampled_pc = pos[idx]
        sampled_pc = sampled_pc.view(B, -1, 3)

        sampled_pc_embeddings = self.point_embed(sampled_pc)

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None) + sampled_pc_embeddings
        x = cross_ff(x) + x

        mean = self.mean_fc(x)
        logvar = self.logvar_fc(x)

        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        kl = posterior.kl()

        return kl, x


    def decode(self, x, queries):

        x = self.proj(x)

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        return self.to_outputs(latents)

    def forward(self, pc, queries):
        kl, x = self.encode(pc)

        o = self.decode(x, queries).squeeze(-1)

        # return o.squeeze(-1), kl
        return {'logits': o, 'kl': kl}

class KLAutoEncoder_deform(nn.Module):
    def __init__(
        self,
        *,
        depth=24,
        dim=512,
        queries_dim=512,
        output_dim = 1,
        num_inputs = 2048,
        num_latents = 512,
        latent_dim = 64,
        heads = 8,
        dim_head = 64,
        weight_tie_layers = False,
        decoder_ff = False
    ):
        super().__init__()

        self.depth = depth

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.point_embed = PointEmbed(dim=int(dim // 2))

        
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])
        
        get_latent_attn = lambda: PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, drop_path_rate=0.1))
        get_latent_ff = lambda: PreNorm(dim, FeedForward(dim, drop_path_rate=0.1))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))
            
        self.decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, dim, heads = 1, dim_head = dim), context_dim = dim)
        self.decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self.to_outputs = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

        self.proj = nn.Linear(latent_dim, dim)

        self.mean_fc = nn.Linear(dim, latent_dim)
        self.logvar_fc = nn.Linear(dim, latent_dim)

    def encode(self, pc, pc2):

        if pc.ndim == 4:
            B, T, N, D = pc.shape
            assert N == self.num_inputs
            assert pc2.shape[2] == self.num_inputs
            
            flattened = pc[:, 0].reshape(B, N, D)  # Reshape and only take the first T-frame
            batch = torch.arange(B).to(pc.device).repeat_interleave(N)
            ratio = 1.0 * self.num_latents / self.num_inputs
            idx = fps(flattened.reshape(-1, D), batch, ratio=ratio)  # Compute indices on the first frame of each sample

            sampled_pc_ls = [pc[:, i].reshape(B * N, D)[idx].reshape(B, -1, 3).unsqueeze(1) for i in range(T)]
            sampled_pc2_ls = [pc2[:, i].reshape(B * N, D)[idx].reshape(B, -1, 3).unsqueeze(1) for i in range(T)]

            sampled_pc = torch.cat(sampled_pc_ls, dim=1)
            sampled_pc2 = torch.cat(sampled_pc2_ls, dim=1)

            # change to three diemension
            pc = pc.view(B*T, -1, D)
            pc2 = pc2.view(B*T, -1, D)
            sampled_pc = sampled_pc.view(B*T, -1, D)
            sampled_pc2 = sampled_pc2.view(B*T, -1, D)

            sampled_pc_embeddings = self.point_embed(sampled_pc)
            sampled_pc_embeddings2 = self.point_embed(sampled_pc2)
            pc_embeddings = self.point_embed(pc)
            pc_embeddings2 = self.point_embed(pc2)
            
            cross_attn, cross_ff = self.cross_attend_blocks
            
            sampled_pc_embeddings_concat = torch.cat([sampled_pc_embeddings, sampled_pc_embeddings2], dim=-1).contiguous()
            pc_embeddings_concat = torch.cat([pc_embeddings, pc_embeddings2], dim=-1).contiguous()


            x = cross_attn(sampled_pc_embeddings_concat, context = pc_embeddings_concat, mask = None) + sampled_pc_embeddings_concat
            x = cross_ff(x) + x

            mean = self.mean_fc(x)
            logvar = self.logvar_fc(x)

            posterior = DiagonalGaussianDistribution(mean, logvar)
            x = posterior.sample()
            _, M, L = x.shape
            x = x.view(B, T, M, L)
            kl = posterior.kl()
        else:
            B, N, D = pc.shape
            _, N2, D2 = pc2.shape
            self.num_inputs = N           
            flattened = pc.view(B*N, D)
            flattened2 = pc2.view(B*N, D)
            pos = flattened
            pos2 = flattened2
            
            batch = torch.arange(B).to(pc.device)
            batch = torch.repeat_interleave(batch, N)
            ratio = 1.0 * self.num_latents / self.num_inputs
            idx = fps(pos, batch, ratio=ratio)

            sampled_pc = pos[idx]
            sampled_pc = sampled_pc.view(B, -1, 3)   
            sampled_pc2 = pos2[idx]                  
            sampled_pc2 = sampled_pc2.view(B, -1, 3) 

            sampled_pc_embeddings = self.point_embed(sampled_pc)   
            sampled_pc_embeddings2 = self.point_embed(sampled_pc2)
            pc_embeddings = self.point_embed(pc)                   
            pc_embeddings2 = self.point_embed(pc2)                
            
            sampled_pc_embeddings_concat = torch.cat([sampled_pc_embeddings, sampled_pc_embeddings2], dim=-1).contiguous()
            pc_embeddings_concat = torch.cat([pc_embeddings, pc_embeddings2], dim=-1).contiguous()

            cross_attn, cross_ff = self.cross_attend_blocks

            x = cross_attn(sampled_pc_embeddings_concat, context = pc_embeddings_concat, mask = None) + sampled_pc_embeddings_concat
            x = cross_ff(x) + x
                
            mean = self.mean_fc(x)
            logvar = self.logvar_fc(x)

            posterior = DiagonalGaussianDistribution(mean, logvar)
            x = posterior.sample()
            kl = posterior.kl()

        return kl, x


    def decode(self, x, queries):

        x = self.proj(x)

        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        # cross attend from decoder queries to latents
        queries_embeddings = self.point_embed(queries)
        latents = self.decoder_cross_attn(queries_embeddings, context = x)

        # optional decoder feedforward
        if exists(self.decoder_ff):
            latents = latents + self.decoder_ff(latents)
        
        return self.to_outputs(latents)

    def forward(self, pc, pc2, queries):
        kl, x = self.encode(pc, pc2)

        d = self.decode(x, queries)

        return {'deformed': d, 'kl': kl, 'latent': x}



def create_autoencoder(dim=512, M=512, latent_dim=64, N=2048, determinisitc=False, depth=24, deform=False):
    if determinisitc:
        if not deform:
            model = AutoEncoder(
                depth=depth,
                dim=dim,
                queries_dim=dim,
                output_dim = 1,
                num_inputs = N,
                num_latents = M,
                heads = 8,
                dim_head = 64,
            )
        else:
            model = AutoEncoder_deform(
                depth=depth,
                dim=dim,
                queries_dim=dim//2,
                output_dim = 3,
                num_inputs = N,
                num_latents = M,
                heads = 8,
                dim_head = 64,
            )
    else:
        if not deform:
            model = KLAutoEncoder(
                depth=depth,
                dim=dim,
                queries_dim=dim,
                output_dim = 1,
                num_inputs = N,
                num_latents = M,
                latent_dim = latent_dim,
                heads = 8,
                dim_head = 64,
            )
        else:
            model = KLAutoEncoder_deform(
                depth=depth,
                dim=dim,
                queries_dim=dim // 2,
                output_dim = 3,
                num_inputs = N,
                num_latents = M,
                latent_dim = latent_dim,
                heads = 8,
                dim_head = 64,
            )
    return model

# Get diy shape ae

def get_model(N=2048, model=None):
    if model == None:
        raise ValueError("Please specify the model name")
    else:
        use_kl = 'kl' in model
        use_deform = 'de' in model

        d = int(model.split('_')[1][1:]) if not use_deform else int(model.split('_')[2][1:])
        m = int(model.split('_')[2][1:]) if not use_deform else int(model.split('_')[3][1:])
        l = int(model.split('_')[3][1:]) if not use_deform else int(model.split('_')[4][1:])
        
        print(f"Creating model with dim={d}, M={m}, latent_dim={l}, kl={use_kl}, deform={use_deform}")
        
        return create_autoencoder(dim=d, M=m, latent_dim=l, determinisitc=not use_kl, deform=use_deform, N=N)
        
def de_kl_d512_m512_l32(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=32, N=N, determinisitc=False, deform=True)

def kl_d512_m512_l8(N=2048):
    return create_autoencoder(dim=512, M=512, latent_dim=8, N=N, determinisitc=False)