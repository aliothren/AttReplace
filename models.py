import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

param_dict_qkv = {
    "deit_tiny_patch16_224":{
        "num_patches": 197,
        "token_hid_dim": 384,
    },
   
    "deit_base_patch16_224":{ # to be determined
        "num_patches": 197,
        "token_hid_dim": 3072,
    }
}

param_dict_all = {
    "deit_tiny_patch16_224":{
        "num_patches": 197,
        "token_hid_dim": 512,
        "channels_dim": 192,
        "channels_hid_dim": 512,
    },
   
    "deit_base_patch16_224":{ # to be determined
        "num_patches": 197,
        "token_hid_dim": 3072,
        "channels_dim": 768,
        "channels_hid_dim": 3072,
    }
}

weight_schedule = {
    5: 0.8, 
    10: 0.7,
    15: 0.6,
    20: 0.5,
    25: 0.4,
    30: 0.3,
    35: 0.2,
    40: 0.1,
    45: 0,
}


class ParallelBlock(nn.Module):
    def __init__(self, attn_block, mixer_block, initial_weight_attention=0.9):
        super(ParallelBlock, self).__init__()
        self.attn_block = attn_block
        self.mixer_block = mixer_block
        self.attn_weight = nn.Parameter(torch.tensor(initial_weight_attention), requires_grad=False)
        self.mixer_weight = nn.Parameter(torch.tensor(1.0 - initial_weight_attention), requires_grad=False)

    def forward(self, x):
        # Forward pass through both the attention and mixer block
        attn_out = self.attn_block(x)
        mixer_out = self.mixer_block(x)
        # Weighted sum of both outputs
        y = self.attn_weight * attn_out + self.mixer_weight * mixer_out
        return y


class MlpBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim, dropout = 0.):
        super(MlpBlock, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        y = self.fc1(x)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.fc2(y)
        y = self.dropout(y)
        return y


class MixerBlock(nn.Module):
    """This is the mixer block to replace attention module"""
    def __init__(self, mode, model_name, dropout = 0. ):
        super(MixerBlock, self).__init__()
        self.mode = mode
        if mode == "qkv":
            self.param_dict = param_dict_qkv
        else:
            self.param_dict = param_dict_all
            
        self.token_mixing = MlpBlock(self.param_dict[model_name]["num_patches"],
                                     self.param_dict[model_name]["token_hid_dim"],
                                     dropout)
        if mode == "all":
            self.channel_mixing = MlpBlock(self.param_dict[model_name]["channels_dim"],
                                           self.param_dict[model_name]["channels_hid_dim"],
                                           dropout)
            self.norm1 = nn.LayerNorm(self.param_dict[model_name]["channels_dim"])
            self.norm2 = nn.LayerNorm(self.param_dict[model_name]["channels_dim"])
        
    def forward(self, x):
        # Token mixing
        if self.mode == "all":
            y = self.norm1(x)
        else:
            y = x
            
        y = y.transpose(1, 2)
        y = self.token_mixing(y)
        y = y.transpose(1, 2)
        x = x + y
        
        if self.mode == "all":
            y = self.norm2(x)
            y = self.channel_mixing(y)
            x = x + y
        
        return x
            

class MlpMixer(nn.Module):
    """Mixer architecture."""
    """
    Note that this class cannot be directly use as a MLP Mixer network,
    because the extra token cause a change in parameter num, but doesn't change the input dim.
    """
    def __init__(self, 
                 in_channels = 3,
                 patch_size = 16,
                 img_size = 224,
                 num_classes = 100,
                 num_blocks = 8,
                 channel_dim = 768,
                 token_hid_dim = 3072,
                 channel_hid_dim = 3072,
                 extra_token = True
                 ):
        super(MlpMixer, self).__init__()
        
        self.num_patches = (img_size // patch_size) ** 2
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=channel_dim, kernel_size=patch_size, stride=patch_size)

        if extra_token:
            self.num_tokens = self.num_patches + 1 
        else:
            self.extra_token = None
            self.num_tokens = self.num_patches
        
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            block = MixerBlock(self.num_tokens, token_hid_dim, channel_dim, channel_hid_dim)
            self.blocks.append(block)
            
        self.layer_norm = nn.LayerNorm(self.num_tokens)
        self.mlp_head = nn.Linear(self.num_tokens, num_classes)
        

    def forward(self, inputs):
        x = self.conv(inputs)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        for block in self.blocks:
            x = block(x)

        x = self.layer_norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.mlp_head(x)  # Final classification layer
        
        return x


class EmptyBlock(nn.Module):
    def __init__(self):
        super(EmptyBlock, self).__init__()
    
    def forward(self, x):
        return x 
    

# original_model = MixerBlock(197, 512, 768, 3072)
# original_model.to(device)
# print(original_model)
# summary(original_model, (1, 197, 768))


