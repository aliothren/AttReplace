import os
import gc
import copy
import torch
import fcntl
import models
import datetime
import argparse

import numpy as np
import torch.backends.cudnn as cudnn

from AttReplace.data import load_dataset
from train import train_model, evaluate_model
from loss import CosineSimilarityLoss, CombinedLoss

from pathlib import Path
from torchsummary import summary
from timm.utils import NativeScaler
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

DEVICE = torch.device("cuda")
BASE_DIR = "/home/yuxinr/AttnDistill/models/"
BASE_MODEL = "deit_tiny_patch16_224"
D_WEIGHT = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"


def get_args_parser():
    parser = argparse.ArgumentParser("DeiT -> MLP Mixer", add_help=False)
    
    # deit model parameters
    parser.add_argument("--d-model", default="deit_tiny_patch16_224", type=str, metavar="MODEL")
    parser.add_argument("--input-size", default=224, type=int, help="expected images size for model input")
    parser.add_argument("--d-weight", default="", help="path of DeiT model checkpoint")
    
    # mixer model parameters
    parser.add_argument("--replace", nargs="+", type=int, help="list for index of blocks to be replaced")
    parser.add_argument("--rep-mode", default="mixer", choices=["mixer", "lstm"], 
                        help="Choose to relace whole attention block or only qkv part")
    parser.add_argument("--qkv-ft-mode", nargs="+", type=str, help="list for qkv finetune strategies")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--eval-model", default="", help="Path of model to be evaluated")
    parser.add_argument('--train', action='store_true', help='Train replaced Mixer blockes')
    parser.add_argument("--gradually-train", action="store_true", help="Gradually distill weight from teacher to student")
    parser.add_argument("--grad-mode", default="linear", choices=["linear", "step", "exp", "cosine", "inverse"], 
                        help="The scheme of weight decay in gradually training")
    parser.add_argument('--train-in-eval', action='store_true', help='Set not training blocks into eval mode')
    parser.add_argument('--finetune', action='store_true', help='Finetuning the whole model')
    parser.add_argument("--ft-model", default="", help="Path of model to be finetuned")
    parser.add_argument("--ft-mode", default="cosine", choices=["cosine", "class", "combine"],
                        type=str, help="criterion of finetune, only used in global finetune")
    
    # data parameters
    parser.add_argument("--data-path", default="/home/yuxinr/datasets/CIFAR/", type=str, help="dataset path")
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
    parser.add_argument('--batch-size', default=4096, type=int)
    parser.add_argument("--drop", type=float, default=0.0, metavar="PCT",
                        help="Dropout rate (default: 0.)")
    parser.add_argument("--drop-path", type=float, default=0.1, metavar="PCT",
                        help="Drop path rate (default: 0.1)")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--rm-shortcut', action='store_true')
    
    # Learning rate schedule parameters
    parser.add_argument('--unscale-lr', action='store_true')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--qkv-ft-lr', type=float, default=5e-5, 
                        help='qkv finetune learning rate (default: 5e-5)')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--warmup-epochs', type=int, default=0, help='Number of warmup epochs')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, help='Warm-up initial learning rate')

    
    # Finetune parameters
    parser.add_argument("--ft-unscale-lr", action="store_true")
    parser.add_argument("--ft-lr", default=5e-5, type=float)
    parser.add_argument("--ft-batch-size", default=512, type=int)
    parser.add_argument("--ft-epochs", default=50, type=int)
    parser.add_argument("--ft-start-epoch", default=0, type=int)
    parser.add_argument('--ft-clip-grad', type=float, default=None, metavar='NORM') 
    return parser


def get_unique_output_dir(base_dir):
    lock_file = Path(base_dir) / ".output_dir_lock"
    lock_file.touch(exist_ok=True)

    with open(lock_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = Path(base_dir) / timestamp
        while output_dir.exists():
            current_time += datetime.timedelta(seconds=1)
            timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
            output_dir = Path(base_dir) / timestamp
        
        output_dir.mkdir(parents=True, exist_ok=False)
        print(f"Created directory: {output_dir}")
        fcntl.flock(f, fcntl.LOCK_UN)

    return output_dir
    

def concat_model(base_model, name_list, idx_list):
    for idx in range(len(name_list)):
        name = name_list[idx]
        block_idx = idx_list[idx]
        
        model_path = Path(BASE_DIR) / name / "model_block.pth"
        model = torch.load(model_path)
        block = model.blocks[block_idx]
        block.to(DEVICE)
        base_model.blocks[block_idx] = block
    return base_model


def train_head(args, replaced_model):
    # Finetune head
    for param in replaced_model.parameters():
        param.requires_grad = False
    for name, param in replaced_model.head.named_parameters():
        param.requires_grad = True
        print(name)
        
    n_parameters = sum(p.numel() for p in replaced_model.parameters() if p.requires_grad)
    print('number of finetunable params:', n_parameters)
    
    optimizer = create_optimizer(args, replaced_model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    finetuned_model, finetuned_model_dict = train_model(
        args=args, mode="class", model=replaced_model, teacher_model=None, criterion=criterion, 
        optimizer=optimizer, loss_scaler=loss_scaler, lr_scheduler=lr_scheduler, 
        train_data=data_loader_train, device=DEVICE, n_parameters=n_parameters
        )
    save_path = args.output_dir / "fted_model.pth"
    torch.save(finetuned_model, save_path)
    
    del optimizer, data_loader_train
    gc.collect()
    torch.cuda.empty_cache()
    return save_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DeiT -> MLP Mixer", parents=[get_args_parser()])
    args = parser.parse_args()
    
    output_dir = Path(BASE_DIR) / "concat"
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = get_unique_output_dir(output_dir)
    
    blocks = [
        "2025-02-11-00-17-13", 
        "2025-02-11-00-17-43",
        "2025-02-11-00-18-14",
        # "2025-02-11-00-18-47", 
        # "2025-02-11-00-19-20", 
        # "2025-02-11-00-19-52",
        # "2025-02-11-05-16-00", 
        # "2025-02-11-05-16-32", 
        # "2025-02-11-10-12-42",
        # "2025-02-11-10-13-40", 
    ]
    block_idx = [
        0,
        1, 
        2,
        # 3,
        # 4,
        # 5,
        # 6,
        # 7,
        # 8,
        # 9, 
        ]
    args.replace = block_idx
    model_deit = create_model(
        BASE_MODEL, pretrained=False, num_classes=1000, drop_rate=args.drop,
        drop_path_rate=args.drop_path, drop_block_rate=None, img_size=args.input_size
        )
    model_deit = models.load_weight(model_deit, D_WEIGHT)
    ori_model = copy.deepcopy(model_deit)
    teacher = copy.deepcopy(model_deit)
    model_deit.to(DEVICE)
    teacher.to(DEVICE)
    replaced_model = concat_model(model_deit, blocks, block_idx)
    
    data_loader_train = load_dataset(args, "train")
    data_loader_val, dataset_val = load_dataset(args, "val")
    
    
    # Test without finetune
    print(f"Evaluating model concated by: {blocks}")
    models.set_requires_grad(replaced_model, "train", target_blocks=[], target_layers="block")
    test_stats = evaluate_model(data_loader_val, replaced_model, DEVICE)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
   
   
    # Finetune with cosine similarity
    print(f"Finetuning concated model with cosine loss:") 
    replaced_model = models.cut_extra_layers(replaced_model, max(block_idx))
    teacher = models.cut_extra_layers(teacher, max(block_idx))
    
    models.set_requires_grad(replaced_model, "train", block_idx, "block")
    models.set_requires_grad(teacher, "train", [], "block")
    
    n_parameters = sum(p.numel() for p in replaced_model.parameters() if p.requires_grad)
    print('number of finetunable params:', n_parameters)
    
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, replaced_model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = CosineSimilarityLoss()
    finetuned_model, finetuned_model_dict = train_model(
        args=args, mode="train", model=replaced_model, teacher_model=teacher, criterion=criterion, 
        optimizer=optimizer, loss_scaler=loss_scaler, lr_scheduler=lr_scheduler, 
        train_data=data_loader_train, device=DEVICE, n_parameters=n_parameters
        )
    
    finetuned_model = models.recomplete_model(
        trained_model=finetuned_model, origin_model=ori_model, repl_blocks=block_idx, 
        grad_train=args.gradually_train, remove_sc=args.rm_shortcut
        )
    
    save_path = args.output_dir / "fted_model.pth"
    torch.save(finetuned_model, save_path)
    
    del optimizer
    
    
    # Test after finetune
    print(f"Evaluating fintuned model:")
    finetuned_model = torch.load(save_path)
    finetuned_model.to(DEVICE)
    models.set_requires_grad(finetuned_model, "train", target_blocks=[], target_layers="block")
    test_stats = evaluate_model(data_loader_val, finetuned_model, DEVICE)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    

    # Train classification head
    save_path = train_head(args, finetuned_model)
    
     
    # Test after train head
    print(f"Evaluating head-trained model:")
    finetuned_model = torch.load(save_path)
    finetuned_model.to(DEVICE)
    models.set_requires_grad(finetuned_model, "train", target_blocks=[], target_layers="block")
    test_stats = evaluate_model(data_loader_val, finetuned_model, DEVICE)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    exit(0)
       
    
    
    
    # Test after 1st stage finetune
    print(f"Evaluating fintuned model:")
    finetuned_model = torch.load(save_path)
    finetuned_model.to(DEVICE)
    models.set_requires_grad(finetuned_model, "train", target_blocks=[], target_layers="block")
    test_stats = evaluate_model(data_loader_val, finetuned_model, DEVICE)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    
    
    blocks = [
        # "2025-02-11-05-16-00", 
        # "2025-02-11-05-16-32", 
        # "2025-02-11-10-12-42",
        # "2025-02-11-10-13-40", 
        "2025-02-11-10-14-39", 
        "2025-02-11-10-15-11",
    ]
    block_idx = [
        # 6,
        # 7,
        # 8,
        # 9, 
        10, 
        11
        ]
    args.replace = block_idx
    replaced_model = concat_model(finetuned_model, blocks, block_idx)
    
    # Global finetune
    print(f"Finetuning concated model on classification loss:")
    models.set_requires_grad(replaced_model, "train", block_idx, "block")
    
    n_parameters = sum(p.numel() for p in replaced_model.parameters() if p.requires_grad)
    print('number of finetunable params:', n_parameters)
    
    optimizer = create_optimizer(args, replaced_model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    teacher = None
    finetuned_model, finetuned_model_dict = train_model(
        args=args, mode="class", model=replaced_model, teacher_model=teacher, criterion=criterion, 
        optimizer=optimizer, loss_scaler=loss_scaler, lr_scheduler=lr_scheduler, 
        train_data=data_loader_train, device=DEVICE, n_parameters=n_parameters
        )
    save_path = args.output_dir / "fted_model.pth"
    torch.save(finetuned_model, save_path)
    
    del optimizer, data_loader_train
    gc.collect()
    torch.cuda.empty_cache()
    
    # Test after finetune
    print(f"Evaluating fintuned model:")
    finetuned_model = torch.load(save_path)
    finetuned_model.to(DEVICE)
    models.set_requires_grad(finetuned_model, "train", target_blocks=[], target_layers="block")
    test_stats = evaluate_model(data_loader_val, finetuned_model, DEVICE)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    