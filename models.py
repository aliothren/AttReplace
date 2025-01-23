import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

param_dict_mixer = {
    "deit_tiny_patch16_224":{
        "num_patches": 197,
        "token_hid_dim": 384,
    },
   
    "deit_base_patch16_224":{ # to be determined
        "num_patches": 197,
        "token_hid_dim": 3072,
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
    def __init__(self, model_name, dropout = 0. ):
        super(MixerBlock, self).__init__()
        
        self.param_dict = param_dict_mixer
        self.token_mixing = MlpBlock(self.param_dict[model_name]["num_patches"],
                                     self.param_dict[model_name]["token_hid_dim"],
                                     dropout)
        
    def forward(self, x):
        # Token mixing
        y = x
        y = y.transpose(1, 2)
        y = self.token_mixing(y)
        y = y.transpose(1, 2)
        
        return y


class AttnBlockNoSC(nn.Module):
    """DeiT Block without shortcut"""
    def __init__(self, original_block):
        super(AttnBlockNoSC, self).__init__()
        self.attn = original_block.attn
        self.mlp = original_block.mlp
        self.norm1 = original_block.norm1
        self.norm2 = original_block.norm2

    def forward(self, x):
        x = self.norm1(x)
        x = self.attn(x)
        x = self.norm2(x)
        x = self.mlp(x)
        return x


class AttnBlockWithSC(nn.Module):
    """DeiT Block with shortcut"""
    def __init__(self, original_block):
        super(AttnBlockNoSC, self).__init__()
        self.attn = original_block.attn
        self.mlp = original_block.mlp
        self.norm1 = original_block.norm1
        self.norm2 = original_block.norm2

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.attn(y)
        y = self.norm2(x)
        x = x + self.mlp(y)
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


# Remove shortcut to avoid gradiant vanish in training
def remove_shortcut(block):
    block = AttnBlockNoSC(block)
    return block
    

# Add shortcut to recover network structure for inference
def recover_shortcut(block):
    block = AttnBlockWithSC(block)
    return  block

    
def replace_attention(model, repl_blocks, target = None, remove_sc = False,
                      model_name = "", grad_train = False):
    for blk_index in repl_blocks:
        
        if remove_sc:
            model.blocks[blk_index] = remove_shortcut(model.blocks[blk_index])
        
        if target == "attn":
            continue
        elif target == "mixer":
            mlp_block = MixerBlock(model_name)
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


def recomplete_model(trained_model, origin_model, repl_blocks, 
                     grad_train = False, remove_sc = False):
    for blk_index in repl_blocks:
        if grad_train:
            origin_model.blocks[blk_index] = trained_model.blocks[blk_index].mixer_block
        else:
            origin_model.blocks[blk_index] = trained_model.blocks[blk_index]
        
        # recover removed shortcuts
        if remove_sc:
            origin_model.blocks[blk_index] = recover_shortcut(origin_model.blocks[blk_index])
    return origin_model


def cut_extra_layers(model, max_index):
    del model.blocks[max_index + 1 :]
    # del model.norm 
    # model.norm = nn.Identity()
    del model.fc_norm
    del model.head_drop
    del model.head
    return model


def set_requires_grad(model, mode, target_blocks = [], target_layers = "mixer", trainable=True):
    target_names = [f"blocks.{block}" for block in target_blocks]
    
    if mode == "finetune":
        for name, param in model.named_parameters():
            param.requires_grad = trainable
    
    if mode == "train":
        # turn the whole block to trainable
        if target_layers == "block":
            for name, param in model.named_parameters():
                param.requires_grad = not trainable
                if any(target in name for target in target_names):
                    param.requires_grad = trainable
        # turn the replaced MLP-Mixer to trainable
        elif target_layers == "mixer":
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

