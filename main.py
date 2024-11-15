import copy
import torch
import datetime
import argparse

import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn

from datasets import build_dataset
from train import train_model, evaluate
from models import MixerBlock, EmptyBlock
from loss import CosineSimilarityLoss, CombinedLoss

from pathlib import Path
from torchsummary import summary
from timm.utils import NativeScaler
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler


def get_args_parser():
    parser = argparse.ArgumentParser("DeiT -> MLP Mixer", add_help=False)
    
    # deit model parameters
    parser.add_argument("--d-model", default="deit_tiny_patch16_224", type=str, metavar="MODEL")
    parser.add_argument("--input-size", default=224, type=int, help="expected images size for model input")
    parser.add_argument("--d-weight", default="", help="path of DeiT model checkpoint")
    
    # mixer model parameters
    parser.add_argument("--replace", nargs="+", type=int, help="list for index of blocks to be replaced")
    parser.add_argument("--rep-mode", default="all", choices=["qkv", "all"], 
                        help="Choose to relace whole attention block or only qkv part")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--eval-model", default="", help="Path of model to be evaluated")
    parser.add_argument('--train', action='store_true', help='Train replaced Mixer blockes')
    parser.add_argument('--finetune', action='store_true', help='Finetuning the whole model')
    parser.add_argument("--ft-model", default="", help="Path of model to be finetuned")
    
    # data parameters
    parser.add_argument("--data-path", default="/home/u17/yuxinr/datasets/", type=str, help="dataset path")
    parser.add_argument("--data-set", default="CIFAR", choices=["CIFAR", "IMNET", "INAT", "INAT19"],
                        type=str, help="Image Net dataset path")
    parser.add_argument("--nb-classes", default=100, type=int, help="number of classes (default:100)")
    
    # data augment parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    
    # training parameters
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--output-dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--batch-size', default=512, type=int)
    parser.add_argument("--drop", type=float, default=0.0, metavar="PCT",
                        help="Dropout rate (default: 0.)")
    parser.add_argument("--drop-path", type=float, default=0.1, metavar="PCT",
                        help="Drop path rate (default: 0.1)")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--train-mode', action='store_true')
    parser.set_defaults(train_mode=True)
    
    # Learning rate schedule parameters
    parser.add_argument('--unscale-lr', action='store_true')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    
    # Finetune parameters
    parser.add_argument("--ft-unscale-lr", action="store_true")
    parser.add_argument("--ft-lr", default=5e-5, type=float)
    parser.add_argument("--ft-batch-size", default=512, type=int)
    parser.add_argument("--ft-epochs", default=50, type=int)
    parser.add_argument("--ft-start-epoch", default=0, type=int)
    parser.add_argument('--ft-clip-grad', type=float, default=None, metavar='NORM') 
    parser.add_argument("--ft-mode", default="cosine", choices=["cosine", "class", "combine"],
                        type=str, help="criterion of finetune")
    return parser
    

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

    
def replace_att2mixer(model, repl_blocks, model_name = "",
                      mode = "all", weighted_model = None):
    for blk_index in repl_blocks:
        
        if weighted_model == None:
            if mode == "all":
                mlp_block = MixerBlock(mode, model_name)
                mlp_block.to("cuda")
                model.blocks[blk_index] = mlp_block
            elif mode == "qkv":
                mlp_block = MixerBlock(mode, model_name)
                mlp_block.to("cuda")
                model.blocks[blk_index].attn = mlp_block
            else:
                raise NotImplementedError("Not available replace method")
            
        else:
            model.blocks[blk_index] = weighted_model.blocks[blk_index]

    return model


def cut_extra_layers(model, max_index, depth = 12):
    for index in range(max_index + 1, depth):
        model.blocks[index] = EmptyBlock()
    model.norm = nn.Identity()
    model.fc_norm = nn.Identity()
    model.head_drop = nn.Identity()
    model.head = nn.Identity()
    return model


def set_requires_grad(model, targets, mode, trainable=True):
    target_names = [f"blocks.{target}" for target in targets]
    for name, param in model.named_parameters():
        # print(name)
        if mode == "finetune":
            param.requires_grad = trainable
        elif any(target in name for target in target_names):
            if mode == "qkv":
                if "mlp" in name:
                    param.requires_grad = not trainable
                else:
                    param.requires_grad = trainable
            elif mode == "all":
                param.requires_grad = trainable
            else:
                raise NotImplementedError("Not available replace method")
        else:
            param.requires_grad = not trainable


def load_dataset(args, mode):
    if mode == "train":
        print(f"Loading training dataset {args.data_set}")
        dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        return data_loader_train
    
    elif mode == "val":
        print(f"Loading validation dataset {args.data_set}")
        dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        return data_loader_val, dataset_val


def main(args):
    print(args)

    device = torch.device(args.device)
    
    ########################################
    ### distributed training not implemented
    ########################################
    
    # fix the seed for reproducibility
    seed = args.seed
    # seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    if args.eval and not args.train and not args.finetune:
        data_loader_val, dataset_val = load_dataset(args, "val")
        print(f"Evaluation model: {args.eval_model}")
        model = torch.load(args.eval_model)
        # Using when wrong head only
        model_deit = create_model(
            args.d_model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size
        )
        model_head = load_weight(model_deit, args.d_weight)
        # model_head = torch.load("/home/u17/yuxinr/block_distill/model/2024-11-01-18-22/deit_model_cifar_head.pth")
        model.head = model_head.head
        
        model.to(device)
        set_requires_grad(model, [], "all")
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
    elif args.train and not args.eval and not args.finetune:
        data_loader_train = load_dataset(args, "train")
        # create structure of DeiT
        print(f"Creating DeiT model: {args.d_model}")
        args.nb_classes = 1000 # if using CIFAR train imnet model
        model_deit = create_model(
            args.d_model,
            pretrained=False,
            num_classes=args.nb_classes,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            img_size=args.input_size
        )
        model_ori = copy.deepcopy(model_deit)
        print(f"Replacing blocks: {args.replace}")
        model_repl = replace_att2mixer(model=model_deit, repl_blocks=args.replace, 
                                       mode=args.rep_mode, model_name = args.d_model)
    
        print(f"Train model: {args.d_model}, target blocks:{args.replace}")
        model = load_weight(model_repl, args.d_weight)
        model_ori = load_weight(model_ori, args.d_weight)
        weighted_model_ori = copy.deepcopy(model_ori)
        partial_model =  cut_extra_layers(model, max(args.replace))
        partial_model_ori = cut_extra_layers(model_ori, max(args.replace))
        partial_model.to(device)
        partial_model_ori.to(device)
        
        set_requires_grad(partial_model, args.replace, args.rep_mode)
        set_requires_grad(partial_model_ori, [], args.rep_mode)
        partial_model.to(device)
        partial_model_ori.to(device)
        
        ### EMA augmentation in training not implemented
        n_parameters = sum(p.numel() for p in partial_model.parameters() if p.requires_grad)
        print('number of trainable params:', n_parameters)
        
        if not args.unscale_lr:
            linear_scaled_lr = args.lr * args.batch_size / 512.0
            args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, model)
        loss_scaler = NativeScaler()
        lr_scheduler, _ = create_scheduler(args, optimizer)
        criterion = CosineSimilarityLoss()
        
        current_time = datetime.datetime.now()
        output_dir = "/home/u17/yuxinr/block_distill/model/" + current_time.strftime("%Y-%m-%d-%H-%M") + "/"
        args.output_dir = Path(output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        trained_model, trained_model_dict = train_model(
            args=args, mode="train", model=partial_model, teacher_model=partial_model_ori,
            criterion=criterion, optimizer=optimizer, loss_scaler=loss_scaler,
            lr_scheduler=lr_scheduler, train_data=data_loader_train, device=device,
            n_parameters=n_parameters)
        trained_model = replace_att2mixer(model=weighted_model_ori, repl_blocks=args.replace, 
                                          weighted_model= trained_model)
        # print(trained_model)
        save_path = args.output_dir / "replaced_model.pth"
        trained_model_dict["model"] = trained_model.state_dict()
        # torch.save(trained_model_dict, save_path)
        torch.save(trained_model, save_path)
        
    elif args.finetune and not args.train and not args.eval:
        data_loader_train = load_dataset(args, "train")
        print(f"Finetuning model: {args.ft_model}")
        model = torch.load(args.ft_model)
        
        # when changing head:
        change2cifar_head = True
        if change2cifar_head:
            model_head = torch.load("/home/u17/yuxinr/block_distill/model/2024-11-01-18-22/deit_model_cifar_head.pth")
            model.head = model_head.head
        
        model.to(device)
        set_requires_grad(model, list(range(len(model.blocks))), "finetune")
        
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of finetunable params:', n_parameters)
        
        args.unscale_lr = args.ft_unscale_lr
        args.lr = args.ft_lr
        args.batch_size = args.ft_batch_size
        args.output_dir = Path(args.ft_model[:-18])
        args.epochs = args.ft_epochs
        args.start_epoch = args.ft_start_epoch
        args.clip_grad = args.ft_clip_grad
        # args.sched = "constant"
        
        if not args.unscale_lr:
            linear_scaled_lr = args.lr * args.batch_size / 512.0
            args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, model)
        loss_scaler = NativeScaler()
        lr_scheduler, _ = create_scheduler(args, optimizer)
        
        if args.ft_mode == "class":
            criterion = torch.nn.CrossEntropyLoss()
            teacher = None
            
        else:
            model_deit = create_model(
                args.d_model,
                pretrained=False,
                num_classes=args.nb_classes,
                drop_rate=args.drop,
                drop_path_rate=args.drop_path,
                drop_block_rate=None,
                img_size=args.input_size
            )
            teacher = load_weight(model_deit, args.d_weight)
            teacher.to(device)
            if change2cifar_head:
                teacher.head = model_head.head
        
            if args.ft_mode == "cosine":
                criterion = CosineSimilarityLoss()
            elif args.ft_mode == "combine":
                criterion == CombinedLoss()
            else:
                raise ValueError("Wrong finetune mode.") 
                
        finetuned_model, finetuned_model_dict = train_model(
            args=args, mode=args.ft_mode, model=model, teacher_model=teacher,
            criterion=criterion, optimizer=optimizer, loss_scaler=loss_scaler,
            lr_scheduler=lr_scheduler, train_data=data_loader_train, device=device,
            n_parameters=n_parameters)
        
        # print(trained_model)
        save_path = args.output_dir / f"finetuned_model_{args.ft_mode}.pth"
        finetuned_model_dict["model"] = finetuned_model.state_dict()
        # torch.save(trained_model_dict, save_path)
        torch.save(finetuned_model, save_path)
        
    else:
        raise ValueError("Please specify running mode (eval/train/finetune).") 


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DeiT -> MLP Mixer", parents=[get_args_parser()])
    args = parser.parse_args()
    
    deit_model = "deit_tiny_patch16_224"
    deit_weight = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
    repl_index = [10]
    
    args.d_model = deit_model
    args.d_weight = deit_weight
    args.replace = repl_index
    args.rep_mode = "all"
    args.epochs = 30
    args.lr = 5e-4
    args.batch_size = 2048
    # args.opt = "sgd"
    
    # train
    args.data_set = "IMNET"
    args.data_path = "/contrib/datasets/ILSVRC2012/"
    args.train = True
    
    # eval
    # args.data_set = "IMNET"
    # args.train = False
    # args.eval = True
    # args.eval_model = "/home/u17/yuxinr/block_distill/model/2024-10-31-17-29/finetuned_model_cosine.pth"
    # args.sched = "constant"
    
    # finetune
    # args.data_set = "CIFAR"
    # args.train = False
    # args.finetune = True
    # args.ft_model = "/home/u17/yuxinr/block_distill/model/2024-10-31-17-29/replaced_model.pth"
    # args.ft_lr = 5e-5
    # args.ft_mode = "cosine"
    # # args.data_path = "/contrib/datasets/ILSVRC2012/"
    # args.ft_epochs = 50
        
    main(args)
    # train_head(args)
 
