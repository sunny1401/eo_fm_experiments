import torch.nn.functional as F
from torch import nn

class GroupedPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = (patch_size, patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, groups=in_chans
        )
        

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DynamicChannelSpecificPatches(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = (patch_size, patch_size)
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

    def initialize_weights(self):
        pass

    def forward(self, x, channel_weights):
        channel_weights = channel_weights.permute(1, 0).unsqueeze(-1).repeat(
            1, 1, self.patch_size[0]).unsqueeze(-1).repeat(
                1, 1, 1, self.patch_size[0]
        )
        assert channel_weights.shape[0] == self.embed_dim
        assert channel_weights.shape[1] == x.shape[1]
        output = F.conv2d(
            x, 
            weight=channel_weights, 
            stride=self.patch_size, 
            padding=0, 
            groups=1
        )
        B, E, N, N = output.shape
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(B, N*N, E)
        return output

