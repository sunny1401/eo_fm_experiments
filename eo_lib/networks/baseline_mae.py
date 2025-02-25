from functools import partial
from typing import Dict

import torch
from torch import nn
from timm.models.vision_transformer import PatchEmbed

from eo_lib.networks.components.mae import MAEDecoder, MAEEncoder


class MaskedAutoencoderViT(nn.Module):

    def __init__(
        self, 
        patch_args: Dict,
        patch_size=16, 
        in_chans=3,
        embed_dim=1024, 
        depth=24, 
        num_heads=16,
        decoder_embed_dim=512, 
        decoder_depth=8, 
        decoder_num_heads=16,
        mlp_ratio=4., 
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6), 
        qkv_bias: bool = True,
        qkv_norm: bool = None,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        drop_path_rate: float = 0.1,
        norm_pix_loss=False,
        patch_module: nn.Module = PatchEmbed,
    ):
        super().__init__()
        self.patch_embed = patch_module(**patch_args)
        self._in_channels = in_chans
        num_patches = self.patch_embed.num_patches

        self.encoder = MAEEncoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_patches=num_patches,
            activation_layer=act_layer,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            attn_drop=attn_drop,
            drop_path_rate=drop_path_rate,
            qkv_bias=qkv_bias,
            qkv_norm=qkv_norm,
            proj_drop=proj_drop
        )

        self.decoder = MAEDecoder(
            depth=decoder_depth,
            embed_dim=decoder_embed_dim,
            num_heads=decoder_num_heads,
            num_patches=num_patches,
            activation_layer=act_layer,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            output_dim=patch_size**2 * in_chans
        )

        self.norm_pix_loss = norm_pix_loss

    def initialize_weights(self):

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.encoder.initialize_weights()
        self.decoder.initialize_weights()

    def forward(self, imgs, mask_ratio: float = 0.75):
        x = self.patch_embed(imgs)
        latent, mask, ids_restore = self.encoder(x, mask_ratio)
        pred = self.decoder(latent, ids_restore)
        loss = self.loss(imgs, pred, mask)
        return loss, pred, mask
    
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        c = imgs.shape[1]
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self._in_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self._in_channels, h * p, h * p))
        return imgs
    
    def loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def downstream_forward_features(self, x, return_for_segmentation: bool = True):
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.encoder.pos_embed[:, 1:, :]
        # append cls token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        if return_for_segmentation:
            return x
        else:
            return x[:, 0]
        

def mae_vit_base_patch(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch  # decoder: 512 dim, 8 blocks
