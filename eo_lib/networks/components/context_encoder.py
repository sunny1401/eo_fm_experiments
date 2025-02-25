import torch
from torch import nn
from typing import Callable


class ChannelSpecificContext(nn.Module):

    def __init__(self, embedding_dim, activation: Callable[..., "nn.Module"] = nn.GELU):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ctx_proj = nn.Sequential(
            nn.Linear(2, embedding_dim),
            activation(),
            nn.LayerNorm(embedding_dim),
        )

    def forward(
        self, 
        ctx: torch.Tensor, 
    ):
        channel_ctx = ctx[0, :, :2]
        channel_ctx_proj = self.ctx_proj(channel_ctx)
        channel_ctx_proj = channel_ctx_proj * 0.01
        return channel_ctx_proj
    

class ImageSpecificContext(nn.Module):

    def __init__(self, embedding_dim, activation: Callable[..., "nn.Module"] = nn.GELU):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ctx_proj = nn.Sequential(
            nn.Linear(6, embedding_dim),
            activation(),
            nn.LayerNorm(embedding_dim)
        )

    def forward(
        self, 
        ctx: torch.Tensor, 
        sequence_length: int = 1, 
    ):
        if ctx.shape[-1] == 8:
            image_ctx = ctx[:, 0, 2:]
        elif ctx.shape[-1] == 6:
            image_ctx = ctx[:, 0, :]
        else:
            raise ValueError("Incongruous Image Context Shape Shape")
        image_ctx_proj = self.ctx_proj(image_ctx)
        image_ctx_proj = image_ctx_proj.unsqueeze(1).repeat(1, sequence_length, 1)
        return image_ctx_proj
    

class FixedContext(nn.Module):

    def __init__(self, embedding_dim, activation: Callable[..., "nn.Module"] = nn.GELU):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ctx_proj = nn.Sequential(
            nn.Linear(8, embedding_dim),
            activation(),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(
        self, 
        ctx: torch.Tensor, 
        sequence_length: int, 
    ):
        ctx = self.ctx_proj(ctx).mean(dim=1)
        ctx = ctx.unsqueeze(1).repeat(1, sequence_length, 1)
        return ctx

class CrossAttentionContext(nn.Module):
    """
    We basically have two things in our context -> 
    image specific features and channel specific features.
    In a batch, the channel wavelengths and resolution - which are both channel specific are the same. 
    However, for an individual image, these values differ
    INn a batch, image specific features such as LatLon, Seasons and Year of capture are same for all channels,
    but they differ from each individual image.

    It is plausible to learn these in a cross attention manner to create a global context token
    Use channel specific features as key and value and use image specifc feature to query these.
    """
    def __init__(self, embedding_dim, num_heads=4, activation=nn.GELU):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.channel_proj = ChannelSpecificContext(embedding_dim=embedding_dim)
        self.image_proj = ImageSpecificContext(embedding_dim=embedding_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)
        self.activation = activation()

    def forward(
        self, 
        ctx: torch.Tensor, 
        sequence_length: int | None = None, 
    ):
        
        channel_ctx = ctx[0, :, :2]
        image_ctx = ctx[:, 0, 2:]
        B = image_ctx.shape[0]
        C = ctx.shape[1]
        channel_ctx_proj = self.channel_proj.ctx_proj(channel_ctx)
        image_ctx_proj = self.image_proj.ctx_proj(image_ctx)
        image_ctx_proj = image_ctx_proj.unsqueeze(1).repeat(1, C, 1)
        channel_ctx_proj = channel_ctx_proj.unsqueeze(0).repeat(B, 1, 1)
        attended_ctx, _ = self.cross_attention(
            image_ctx_proj, channel_ctx_proj, channel_ctx_proj)
        
        if sequence_length:
            global_context_token = attended_ctx.mean(dim=1)
            global_context_token = self.activation(global_context_token)
            global_context_token = global_context_token.unsqueeze(1).repeat(1, sequence_length, 1)
            return global_context_token
        return attended_ctx