from abc import abstractmethod
from functools import partial

import torch
from torch import nn
from timm.models.vision_transformer import Block

from eo_lib.networks.utils import random_masking, get_2d_sincos_pos_embed


class MAEBlock(nn.Module):

    def __init__(
        self,
        depth: int,
        embed_dim: int,
        num_heads: int,
        num_patches: int,
        activation_layer: nn.Module = nn.GELU,
        mlp_ratio: int = 4,
        norm_layer: nn.Module =partial(nn.LayerNorm, eps=1e-6), 
        attn_drop: float = 0.,
        drop_path_rate: float = 0.1,
        qkv_bias: bool = True,
        qkv_norm: bool = None,
        proj_drop: float = 0.,
    ):
        
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                act_layer=activation_layer,
                dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_norm=qkv_norm,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path_rate,
                norm_layer=norm_layer
            )
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.num_patches = num_patches

    @abstractmethod
    def init_block_component(self):
        raise NotImplementedError

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.init_block_component()
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @abstractmethod
    def forward(self, x, mask_ratio: float = 0.75, ids_restore: torch.tensor | None = None):
        raise NotImplementedError



class MAEEncoder(MAEBlock):

    def __init__(
        self, 
        depth, 
        embed_dim,
        num_heads, 
        num_patches, 
        activation_layer = nn.GELU, 
        mlp_ratio = 4, 
        norm_layer = partial(nn.LayerNorm, eps=0.000001), 
        attn_drop = 0, 
        drop_path_rate = 0.1, 
        qkv_bias = True, 
        qkv_norm = None, 
        proj_drop = 0
    ):
        super().__init__(
            depth=depth, 
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            num_patches=num_patches, 
            activation_layer=activation_layer, 
            mlp_ratio=mlp_ratio, 
            norm_layer=norm_layer, 
            attn_drop=attn_drop, 
            drop_path_rate=drop_path_rate, 
            qkv_bias=qkv_bias, 
            qkv_norm=qkv_norm, 
            proj_drop=proj_drop
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def init_block_component(self):
        torch.nn.init.normal_(self.cls_token, std=.02)

    def forward(self, x, mask_ratio: float = 0.75):

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore
    

class MAEDecoder(MAEBlock):

    def __init__(
        self, 
        depth, 
        embed_dim,
        encoder_dim,
        num_heads, 
        num_patches, 
        output_dim, 
        activation_layer = nn.GELU, 
        mlp_ratio = 4, 
        norm_layer = partial(nn.LayerNorm, eps=0.000001), 
    ):
        super().__init__(
            depth=depth, 
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            num_patches=num_patches, 
            activation_layer=activation_layer, 
            mlp_ratio=mlp_ratio, 
            norm_layer=norm_layer, 
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder_embed = nn.Linear(encoder_dim, embed_dim, bias=True)
        # self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * in_chans, bias=True)
        self.decoder_pred = nn.Linear(embed_dim, output_dim, bias=True)

    def init_block_component(self):
        torch.nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x, ids_restore):
        
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x