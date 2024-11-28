import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            

def load_weight(model, weight):
    if weight.startswith("https"):
        checkpoint = torch.hub.load_state_dict_from_url(
            weight, map_location="cpu", check_hash=True)
    else:
        checkpoint = torch.load(weight, map_location="cpu")
    checkpoint_model = checkpoint["model"]
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
            
    # interpolate position embedding
    pos_embed_checkpoint = checkpoint_model['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model['pos_embed'] = new_pos_embed
    
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_model, strict=False)
    # print("Missing keys:", missing_keys)
    # print("Unexpected keys:", unexpected_keys)
    return model

 
def replace_attention(model, repl_blocks, target = "mixer", model_name = "", mode = "all", 
                      grad_train = False):
    for blk_index in repl_blocks:
        
        if target == "mixer":
            mixer_block = MixerBlock(mode, model_name)
            
        if mode == "all":
            repl_block = mixer_block
        elif mode == "qkv":
            mlp_block = mixer_block
            repl_block = copy.deepcopy(model.blocks[blk_index])
            repl_block.attn = mlp_block
        else:
            raise NotImplementedError("Not available replace method")  

        if grad_train:
            teacher_block = copy.deepcopy(model.blocks[blk_index])
            repl_block = ParallelBlock(teacher_block, repl_block)
                
        repl_block.to("cuda")
        model.blocks[blk_index] = repl_block
    
    return model


def recomplete_model(trained_model, origin_model, repl_blocks, grad_train = False):
    for blk_index in repl_blocks:
        if grad_train:
            origin_model.blocks[blk_index] = trained_model.blocks[blk_index].mixer_block
        else:
            origin_model.blocks[blk_index] = trained_model.blocks[blk_index]
    return origin_model


def cut_extra_layers(model, max_index):
    del model.blocks[max_index + 1 :]
    # del model.norm 
    # model.norm = nn.Identity()
    del model.fc_norm
    del model.head_drop
    del model.head
    return model


def set_requires_grad(model, mode, target_blocks = [], target_layers = "qkv", trainable=True):
    target_names = [f"blocks.{block}" for block in target_blocks]
    
    if mode == "finetune":
        for name, param in model.named_parameters():
            param.requires_grad = trainable
    
    if mode == "train":
        # turn the whole block to trainable
        if target_layers == "all" or target_layers == "block":
            for name, param in model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    param.requires_grad = trainable
        # turn the replaced part to trainable
        elif target_layers == "qkv":
            for name, param in model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    if "mlp" not in name:
                        param.requires_grad = trainable
        # turn the FC layers in replaced block to trainable      
        elif target_layers == "FC":
            for name, param in model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    if "mlp" in name:
                        param.requires_grad = trainable

