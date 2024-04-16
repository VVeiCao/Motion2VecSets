import torch
import torch.nn as nn
from .models_ae import PreNorm, Attention, FeedForward, PointEmbed, fps

class Scan2latent(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        num_inputs = 2048,
        num_latents = 512,
    ):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=dim)


    def forward(self, pc):
        B, N, D = pc.shape
        assert N == self.num_inputs
        
        flattened = pc.reshape(B*N, D)

        batch = torch.arange(B).to(pc.device)
        batch = torch.repeat_interleave(batch, N)

        pos = flattened
        if self.num_inputs > self.num_latents:
            ratio = 1.0 * self.num_latents / self.num_inputs
            idx = fps(pos, batch, ratio=ratio)
            sampled_pc = pos[idx]
        else:
            sampled_pc = pos
        sampled_pc = sampled_pc.view(B, -1, 3)

        sampled_pc_embeddings = self.point_embed(sampled_pc)

        pc_embeddings = self.point_embed(pc)

        cross_attn, cross_ff = self.cross_attend_blocks

        x = cross_attn(sampled_pc_embeddings, context = pc_embeddings, mask = None) + sampled_pc_embeddings
        x = cross_ff(x) + x

        return x
    
class Scan2latent_deform_timeatt(nn.Module):
    def __init__(
        self,
        *,
        dim=512,
        n_cond_frames=3,
        num_inputs = 2048,
        num_latents = 512,
        dim_cond_latent = 32,
    ):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_latents = num_latents

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(dim, Attention(dim, dim, heads = 1, dim_head = dim), context_dim = dim),
            PreNorm(dim, FeedForward(dim))
        ])

        self.point_embed = PointEmbed(dim=int(dim//2))

    def forward(self, pc, pc2):

        if pc.ndim == 3:
            B, N, D = pc.shape
            assert N == self.num_inputs
            _, N2, D2 = pc2.shape
            assert N2 == self.num_inputs
            
            flattened = pc.view(B*N, D)
            flattened2 = pc2.view(B*N, D)
            pos = flattened
            pos2 = flattened2
            
            batch = torch.arange(B).to(pc.device)
            batch = torch.repeat_interleave(batch, N)


            if self.num_inputs > self.num_latents:
                ratio = 1.0 * self.num_latents / self.num_inputs
                idx = fps(pos, batch, ratio=ratio)
                sampled_pc = pos[idx]
                sampled_pc2 = pos2[idx]
            else:
                sampled_pc = pos
                sampled_pc2 = pos2
                
            sampled_pc = sampled_pc.view(B, -1, 3)
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


        else:
            B, T, N, D = pc.shape
            assert N == self.num_inputs

            _, _, N2, D2 = pc2.shape
            assert N2 == self.num_inputs
            
            flattened = pc[:, 0].contiguous().view(B, N, D) 
            flattened2 = pc2[:, 0].contiguous().view(B, N, D) 

            batch = torch.arange(B).to(pc.device)
            batch = torch.repeat_interleave(batch, N)


            if self.num_inputs > self.num_latents:
                ratio = 1.0 * self.num_latents / self.num_inputs
                idx = fps(flattened.view(-1, D), batch, ratio=ratio) 

                idx = idx.view(B, -1) # Reshape idx to [B, new_N], where new_N is the number of points after FPS
                idx = idx.unsqueeze(1).expand(B, T, -1) # Expand along the time dimension

                sampled_pc = torch.gather(pc, 2, idx.unsqueeze(-1).expand(*idx.shape, D))
                sampled_pc2 = torch.gather(pc2, 2, idx.unsqueeze(-1).expand(*idx.shape, D))

                pc = pc.view(B*T, -1, D)
                pc2 = pc2.view(B*T, -1, D)
                sampled_pc = sampled_pc.view(B*T, -1, D)
                sampled_pc2 = sampled_pc2.view(B*T, -1, D)
            else:
                pc = pc.view(B*T, -1, D)
                pc2 = pc2.view(B*T, -1, D)
                sampled_pc = pc
                sampled_pc2 = pc2

            sampled_pc_embeddings = self.point_embed(sampled_pc)
            sampled_pc_embeddings2 = self.point_embed(sampled_pc2)
            pc_embeddings = self.point_embed(pc)
            pc_embeddings2 = self.point_embed(pc2)
            sampled_pc_embeddings_concat = torch.cat([sampled_pc_embeddings, sampled_pc_embeddings2], dim=-1).contiguous()
            pc_embeddings_concat = torch.cat([pc_embeddings, pc_embeddings2], dim=-1).contiguous()
            
            cross_attn, cross_ff = self.cross_attend_blocks

            x = cross_attn(sampled_pc_embeddings_concat, context = pc_embeddings_concat, mask = None) + sampled_pc_embeddings_concat
            x = cross_ff(x) + x
            _, M, L = x.shape
            x = x.view(B, T, M, L)

        return x