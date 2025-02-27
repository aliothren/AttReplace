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

from data import load_dataset
from train import train_model, evaluate_model

from pathlib import Path
from torchsummary import summary
from timm.utils import NativeScaler
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler

DATA_PATH = {
    "CIFAR": "/home/yuxinr/datasets/CIFAR/",
    "IMNET": "/srv/datasets/imagenet/",
    }
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = "/home/yuxinr/AttnDistill/"


def parse_replace(value):
    """Parse --replace parameter, support single numbers and ranges"""
    parts = value.split()
    numbers = []
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            numbers.extend(range(start, end + 1))
        else:
            numbers.append(int(part))
    return numbers


def get_args_parser():
    parser = argparse.ArgumentParser("DeiT -> MLP Mixer", add_help=False)
    
    # Environment setups
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--base-dir", default=BASE_DIR, help="Base output path")
    parser.add_argument("--output-dir", default='', help="Output path")
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.set_defaults(pin_mem=True)
    
    # Transformer (DeiT) model setups
    parser.add_argument("--d-model", default="deit_tiny_patch16_224", type=str, metavar="MODEL")
    parser.add_argument("--input-size", default=224, type=int, help="expected images size for model input")
    parser.add_argument("--d-weight", default="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                        help="path of DeiT model checkpoint")
    
    # data parameters
    parser.add_argument("--dataset", default="IMNET", choices=["CIFAR", "IMNET"], type=str)
    parser.add_argument("--data-path", type=str, help="Path of dataset")
    parser.add_argument("--nb-classes", default=1000, type=int, 
                        help="Number of classes in dataset (default:1000)")
    parser.add_argument("--train-subset", default=1.0, type=float, help="Sampling rate from dataset")
    
    # Running mode
    parser.add_argument("--mode", default="train", choices=["train", "eval", "finetune"], 
                        help="Runing mode")
    
    # Training setups
    parser.add_argument("--train-mode", default="parallel", choices=["parallel", "sequential"])
    parser.add_argument("--step", default=3, type=int, help="Step length when sequentially replace blocks and training")
    parser.add_argument("--interm-model", default="", type=str, help="Path of intermediate model in sequential training")
    parser.add_argument("--replace", type=parse_replace, help="List of indices or range of blocks to replace")
    parser.add_argument("--rep-by", default="lstm", choices=["mixer", "lstm"], 
                        help="Structure used to replace attention")
    parser.add_argument("--block-ft", action='store_true', 
                        help="Block-level finetune the replaced blocks after training attention")
    parser.set_defaults(block_ft=True)
    parser.add_argument("--train-loss", default="similarity", choices=["similarity", "classification", "combine"],
                        type=str, help="Criterion using in training")
    
    # Training parameters
    parser.add_argument('--seed', default=42, type=int, help="Random seed")
    parser.add_argument("--drop", type=float, default=0.0, metavar="PCT",help="Dropout rate (default: 0.)")
    parser.add_argument("--drop-path", type=float, default=0.1, metavar="PCT", help="Drop path rate (default: 0.1)")
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--epochs", default=30, type=int, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, metavar='LR', help='Replaced attention learning rate')
    parser.add_argument('--unscale-lr', action='store_true', help="Not scale lr according to batch size")
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--warmup-lr', type=float, default=1e-5, help='Warm-up initial learning rate')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')    
    
    parser.add_argument("--block-ft-train-loss", default="similarity", choices=["similarity", "classification", "combine"],
                        type=str, help="Criterion using in training")
    parser.add_argument("--block-ft-batch-size", default=256, type=int, help="Batch size when block-level finetuning")
    parser.add_argument("--block-ft-epochs", default=30, type=int, help="Training epochs when block-level finetuning")
    parser.add_argument("--block-ft-lr", type=float, default=5e-4, metavar='LR', 
                        help='Learning rate when block-level finetuning')
    parser.add_argument('--block-ft-unscale-lr', action='store_true',
                        help="Not scale lr according to batch size when block-level finetuning")
    parser.add_argument('--block-ft-warmup-epochs', type=int, default=5, 
                        help='Number of warmup epochs when block-level finetuning')
    parser.add_argument('--block-ft-warmup-lr', type=float, default=1e-5,
                        help='Warm-up initial learning rate when block-level finetuning')
    parser.add_argument('--block-ft-sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler when block-level finetuning (default: "cosine")')
    parser.add_argument("--finetune-head", action='store_true', 
                        help="Only use for sequential train/finetune: finetune classification head after sequential train")
    parser.set_defaults(finetune_head=True)
    
    # Evaluation setups
    parser.add_argument("--eval-model", default="", help="Path of model to be evaluated")
    
    # Finetuning setups
    parser.add_argument("--ft-mode", default="head", choices=["head", "sequential"])
    parser.add_argument("--ft-model", default="", help="Path of model to be finetuned")
    parser.add_argument("--ft-loss", default="classification", 
                        choices=["similarity", "classification", "combine"],
                        type=str, help="Criterion using in global finetune")
    parser.add_argument("--ft-batch-size", default=256, type=int, help="Batch size when global finetuning")
    parser.add_argument("--ft-epochs", default=30, type=int, help="Training epochs when global finetuning")
    parser.add_argument("--ft-lr", type=float, default=5e-4, metavar='LR', 
                        help='Learning rate when global finetuning')
    parser.add_argument('--ft-unscale-lr', action='store_true',
                        help="Not scale lr according to batch size when global finetuning")
    parser.add_argument('--ft-warmup-epochs', type=int, default=5, 
                        help='Number of warmup epochs when global finetuning')
    parser.add_argument('--ft-warmup-lr', type=float, default=1e-5,
                        help='Warm-up initial learning rate when global finetuning')
    parser.add_argument('--ft-sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler when global finetuning (default: "cosine")')
    
    # data augment parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, help="Crop ratio for evaluation")
    
    # Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT', help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel', help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1, help='Random erase count (default: 1)')

    return parser
    

def get_unique_output_dir(base_dir):
    model_dir = Path(base_dir) / "models"
    lock_file = Path(model_dir) / ".output_dir_lock"
    lock_file.touch(exist_ok=True)

    with open(lock_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = model_dir / timestamp
        while output_dir.exists():
            current_time += datetime.timedelta(seconds=1)
            timestamp = current_time.strftime("%Y-%m-%d-%H-%M-%S")
            output_dir = model_dir / timestamp
        
        output_dir.mkdir(parents=True, exist_ok=False)
        print(f"Created directory: {output_dir}")
        fcntl.flock(f, fcntl.LOCK_UN)

    return output_dir


def concat_model(base_model, name_list, idx_list):
    for idx in range(len(idx_list)):
        name = name_list[idx]
        # Notice that this name only works for parallel trained models
        model_path = Path(BASE_DIR) / "models" / name / "model_block.pth"
        model = torch.load(model_path)
        block = model.blocks[idx_list[idx]]
        block.to(DEVICE)
        base_model.blocks[idx_list[idx]] = block
    return base_model


def evaluate(args):
    print(args)
    print(f"Running in eval mode, args.mode: {args.mode}")
    print(f"Using device: {args.device}")

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True
    # Load dataset
    data_loader_val, dataset_val = load_dataset(args, "val")
    # Load model
    print(f"Evaluation model: {args.eval_model}")
    model = torch.load(args.eval_model)
    model.to(args.device)
    
    models.set_requires_grad(model, target_blocks=[])
    test_stats = evaluate_model(data_loader_val, args.device, model)
    print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
    
    del model, dataset_val
    gc.collect()
    torch.cuda.empty_cache()


def train(args, seq=0):
    print(args)
    print(f"Running in train mode, args.mode: {args.mode}")
    print(f"Using device: {args.device}")

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    data_loader_train = load_dataset(args, "train")
    data_loader_val, _ = load_dataset(args, "val")
        
    # Load base models
    print(f"Creating DeiT model: {args.d_model}")
    args.nb_classes = 1000 # if using CIFAR train imnet model
    model_deit = create_model(
        args.d_model, pretrained=False, num_classes=args.nb_classes, drop_rate=args.drop,
        drop_path_rate=args.drop_path, drop_block_rate=None, img_size=args.input_size
        )
    model_deit = models.load_weight(model_deit, args.d_weight)
    model_deit.to(args.device)
    # Load and modify teacher model
    teacher_model = copy.deepcopy(model_deit)
    teacher_model.to(args.device)
    teacher_model = models.replace_attention(
        model=teacher_model, repl_blocks=args.replace, target="attn", 
        model_name=args.d_model
    )
    # Load and modify student model
    if seq == 0:
        student_model = models.replace_attention(
            model=model_deit, repl_blocks=args.replace, target=args.rep_by, 
            model_name=args.d_model
        )
    else:
        student_model = torch.load(args.interm_model)
        student_model = models.replace_attention(
            model=student_model, repl_blocks=args.replace, target=args.rep_by, 
            model_name=args.d_model
        )
    
    # Set trainable parameters
    models.set_requires_grad(teacher_model, "train", [], "attn") # No trainable param in teacher
    models.set_requires_grad(student_model, "train", args.replace, "attn") # Target attn part trainable
    n_parameters = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    print(f"number of trainable params: {n_parameters}")
    
    # Set unique output directory
    if seq == 0:
        args.output_dir = get_unique_output_dir(args.base_dir)
    
    # Set training configurations
    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size / 512.0
        args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, student_model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    
    # Train model
    trained_model, trained_model_dict = train_model(
        args=args, stage="attn", loss_mode=args.train_loss,
        model=student_model, teacher_model=teacher_model,
        train_data=data_loader_train, test_data=data_loader_val,
        optimizer=optimizer, loss_scaler=loss_scaler, 
        lr_scheduler=lr_scheduler, n_parameters=n_parameters
        )
    
    # Save model
    save_path = args.output_dir / f"model_seq{seq}.pth"
    torch.save(trained_model, save_path)
    args.interm_model = save_path
    
    if args.block_ft:
        print(f"Doing block-level finetuning of attention-trained model...")
        # Continue training on trained_model
        models.set_requires_grad(trained_model, "train", args.replace, "block") # Target blocks trainbale
        models.set_requires_grad(teacher_model, "train", [], "block") # No parameter trainable in teacher
        n_parameters = sum(p.numel() for p in trained_model.parameters() if p.requires_grad)
        print(f"number of trainable params: {n_parameters}")
        
        # Set block-level parameters as training params
        args.batch_size = args.block_ft_batch_size
        args.epochs = args.block_ft_epochs
        args.lr = args.block_ft_lr
        if not args.block_ft_unscale_lr:
            linear_scaled_lr = args.lr * args.batch_size / 512.0
            args.lr = linear_scaled_lr
        args.warmup_epochs = args.block_ft_warmup_epochs
        args.warmup_lr = args.block_ft_warmup_lr
        args.sched = args.block_ft_sched
        # Set training configurations
        optimizer = create_optimizer(args, trained_model)
        loss_scaler = NativeScaler()
        lr_scheduler, _ = create_scheduler(args, optimizer)
        
        # Finetune model
        fted_model, fted_model_dict = train_model(
            args=args, stage="block", loss_mode=args.block_ft_train_loss,
            model=trained_model, teacher_model=teacher_model,
            train_data=data_loader_train, test_data=data_loader_val,
            optimizer=optimizer, loss_scaler=loss_scaler, lr_scheduler=lr_scheduler, 
            n_parameters=n_parameters
            )
        
        # Save finetuned model
        save_path = args.output_dir / f"model_block_seq{seq}.pth"
        torch.save(fted_model, save_path)
        args.interm_model = save_path
        
        del optimizer, fted_model, teacher_model, data_loader_train, data_loader_val
        gc.collect()
        torch.cuda.empty_cache()


def finetune(args, seq=0, ft_mode="head", name_list=[], target_blocks = []):
    args.replace = target_blocks
    print(args)
    print(f"Running in finetune mode, args.mode: {args.mode}")
    print(f"Using device: {args.device}")

    # fix the seed for reproducibility
    seed = args.seed
    # seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)
    cudnn.benchmark = True

    data_loader_train = load_dataset(args, "train")
    data_loader_val, _ = load_dataset(args, "val")
    
    # Load replaced model
    if ft_mode == "head":
        # If name_list is empty, suppose that replaced model already exists
        if len(name_list) == 0:
            print("Name_list is empty")
            print(f"Loading intermediate model: {args.interm_model}")
            model = torch.load(args.interm_model)
            model.to(args.device)
        # If name_list is not empty, replace the parallel trained blocks into DeiT
        else:
            print(f"Name_list: {name_list}")
            print(f"Creating DeiT model: {args.d_model}")
            args.nb_classes = 1000 # if using CIFAR train imnet model
            model_deit = create_model(
                args.d_model, pretrained=False, num_classes=args.nb_classes, drop_rate=args.drop,
                drop_path_rate=args.drop_path, drop_block_rate=None, img_size=args.input_size
                )
            model_deit = models.load_weight(model_deit, args.d_weight)
            model_deit.to(args.device)
            model = concat_model(model_deit, name_list, args.replace)
        teacher_model = None
        if args.ft_loss != "classification":
            raise ValueError("You can only use classification loss as train_loss.") 
            
    elif ft_mode == "sequential":
        print(f"Creating DeiT model: {args.d_model}")
        args.nb_classes = 1000 # if using CIFAR train imnet model
        model_deit = create_model(
            args.d_model, pretrained=False, num_classes=args.nb_classes, drop_rate=args.drop,
            drop_path_rate=args.drop_path, drop_block_rate=None, img_size=args.input_size
            )
        model_deit = models.load_weight(model_deit, args.d_weight)
        teacher_model = copy.deepcopy(model_deit)
        teacher_model.to(args.device)
        if seq == 0:
            model = concat_model(model_deit, name_list, args.replace)
            model.to(args.device)
        else:
            model = torch.load(args.interm_model)
            model.to(args.device)
            model = concat_model(model, name_list, args.replace)
        
    else:
        raise ValueError("Invalid ft_mode (sequential/head).") 
    
    models.set_requires_grad(model, "finetune", args.replace, ft_mode) # head trainable
    if teacher_model != None:
        models.set_requires_grad(teacher_model, "finetune", [], ft_mode) # No trainable param in teacher
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable params: {n_parameters}")
    
    # Set unique output directory
    if seq == 0 and args.output_dir == '':
        args.output_dir = get_unique_output_dir(args.base_dir)
    
    # Set block-level parameters as training params
    args.batch_size = args.ft_batch_size
    args.epochs = args.ft_epochs
    args.lr = args.ft_lr
    if not args.ft_unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size / 512.0
        args.lr = linear_scaled_lr
    args.warmup_epochs = args.ft_warmup_epochs
    args.warmup_lr = args.ft_warmup_lr
    args.sched = args.ft_sched
    # Set training configurations
    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    
    fted_model, fted_model_dict = train_model(
        args=args, stage="global", loss_mode=args.ft_loss,
        model=model, teacher_model=teacher_model,
        train_data=data_loader_train, test_data=data_loader_val,
        optimizer=optimizer, loss_scaler=loss_scaler, 
        lr_scheduler=lr_scheduler, n_parameters=n_parameters
        )
    
    if ft_mode == "head":
        save_path = args.output_dir / f"model_ft_head.pth"
    else:
        save_path = args.output_dir / f"model_ft_seq{seq}.pth"
    torch.save(fted_model, save_path)
    args.interm_model = save_path
    

def eval_trained_models(args, seq=0):
    args.mode = "eval"
    model_dir = Path(args.output_dir)
    args.eval_model = model_dir / f"model_seq{seq}.pth"
    print(f"Evaluating Seq{seq} trained model after attn stage.")
    print(f"(Replaced blocks in Seq{seq}: {args.replace}")
    evaluate(args)
    if args.block_ft:
        args.eval_model = model_dir / f"model_block_seq{seq}.pth"
        print(f"Evaluating Seq{seq} trained model after block-level finetune stage.")
        print(f"(Replaced blocks in Seq{seq}: {args.replace}")
        evaluate(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DeiT -> MLP Mixer", parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.data_path is None:
        args.data_path = DATA_PATH[args.dataset]
    
    if args.mode == "eval":
        evaluate(args)
        
    elif args.mode == "train":
        # Sequentially replace blocks and train, args.step blocks each time
        if args.train_mode == "sequential":
            seq = 0
            print(f"Sequentially training blocks {args.replace}, {args.step} blocks once...")
            blocks = args.replace.copy()
            while len(blocks) > 0:
                brgs = copy.deepcopy(args) # Copy args to avoid change on args 
                brgs.replace = blocks[: brgs.step] # Modify brgs for this seq training
                
                train(args=brgs, seq=seq)
                
                # Deliver params for next seq
                args.output_dir = brgs.output_dir
                args.interm_model = brgs.interm_model
                blocks = blocks[args.step :]
                seq += 1
                
            if args.finetune_head:
                finetune(args, ft_mode="head")
        
        # Replace and train all blocks in args.replace in once 
        elif args.train_mode == "parallel":
            print(f"Parallel training blocks {args.replace}...")
            train(args)
        
        else:
            raise ValueError("Invalid train_mode (sequential/parallel).") 
            
    elif args.mode == "finetune": 
        # Name_list and block_list should correspond exactly
        block_list = [ # each item indicates blocks to be finetuned in one step
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
        ]
        name_list = [ # the name of corresponding block
            ["2025-02-13-20-36-16", "2025-02-13-20-36-45", "2025-02-13-20-37-18"],
            ["2025-02-13-20-58-53", "2025-02-13-23-03-53", "2025-02-13-23-04-53"],
            ["2025-02-13-23-05-54", "2025-02-13-23-07-24", "2025-02-13-23-08-52"],
            ["2025-02-13-23-09-23", "2025-02-13-23-09-54", "2025-02-13-23-29-29"],
        ]
        # Sequentially finetune target blocks
        if args.ft_mode == "sequential":
            if len(block_list) != len(name_list):
                raise ValueError("Unmatch length between block_list and name_list.")
            for idx in range(len(block_list)):
                target_block = block_list[idx]
                target_name = name_list[idx]
                if len(target_block) != len(target_name):
                    raise ValueError(f"Unmatch length between blocks and names in seq{idx}.")
                
                brgs = copy.deepcopy(args) # Copy args to avoid change on args 
                finetune(
                    args=brgs, seq=idx, ft_mode=args.ft_mode, 
                    name_list=target_name, target_blocks=target_block
                    )
                
                # Deliver params for next seq
                args.output_dir = brgs.output_dir
                args.interm_model = brgs.interm_model
            
            if args.finetune_head:
                finetune(args,ft_mode="head")
                
        # Only finetune classification head
        elif args.ft_mode == "head":
            flattened_block = [x for sublist in block_list for x in sublist]
            flattened_name = [x for sublist in name_list for x in sublist]
            finetune(
                args, ft_mode=args.ft_mode, 
                name_list=flattened_name, target_blocks=flattened_block
                )
           
        else:
            raise ValueError("Invalid ft_mode (sequential/head).")  
            
    else:
        raise ValueError("Invalid mode (eval/train/finetune).") 
        