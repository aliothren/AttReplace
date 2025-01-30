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

from datasets import load_dataset
from train import train_model, evaluate
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
    parser.add_argument('--rm-shortcut', action='store_true')
    
    # Learning rate schedule parameters
    parser.add_argument('--unscale-lr', action='store_true')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--qkv-ft-lr', type=float, default=5e-5, 
                        help='qkv finetune learning rate (default: 5e-5)')
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs')
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
        model.to(device)
        models.set_requires_grad(model, "train", target_blocks=[], target_layers="all")
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        del model, data_loader_val, dataset_val
        gc.collect()
        torch.cuda.empty_cache()
        
    elif args.train and not args.eval and not args.finetune:
        data_loader_train = load_dataset(args, "train")
        # create structure of DeiT
        print(f"Creating DeiT model: {args.d_model}")
        args.nb_classes = 1000 # if using CIFAR train imnet model
        model_deit = create_model(
            args.d_model, pretrained=False, num_classes=args.nb_classes, drop_rate=args.drop,
            drop_path_rate=args.drop_path, drop_block_rate=None, img_size=args.input_size
            )
        model_deit = models.load_weight(model_deit, args.d_weight)
        model_ori = copy.deepcopy(model_deit)
        model = copy.deepcopy(model_deit)
        print(f"Replacing blocks: {args.replace}; Replace by: {args.rep_mode}")
        model_ori = models.replace_attention(
            model=model_ori, repl_blocks=args.replace, target="attn", remove_sc=args.rm_shortcut)
        model_repl = models.replace_attention(
            model=model, repl_blocks=args.replace, target=args.rep_mode, remove_sc=args.rm_shortcut,
            model_name=args.d_model, grad_train=args.gradually_train
            )

        partial_model = models.cut_extra_layers(model_repl, max(args.replace))
        partial_model_ori = models.cut_extra_layers(model_ori, max(args.replace))
        models.set_requires_grad(partial_model, "train", args.replace, args.rep_mode)
        models.set_requires_grad(partial_model_ori, "train", [], args.rep_mode)
        partial_model.to(device)
        partial_model_ori.to(device) 
        
        base_dir = "/home/u17/yuxinr/block_distill/model/"
        args.output_dir = get_unique_output_dir(base_dir)
              
        ### EMA augmentation in training not implemented
        n_parameters = sum(p.numel() for p in partial_model.parameters() if p.requires_grad)
        print(f"number of trainable params: {n_parameters}")

        if not args.unscale_lr:
            linear_scaled_lr = args.lr * args.batch_size / 512.0
            args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, partial_model)
        loss_scaler = NativeScaler()
        lr_scheduler, _ = create_scheduler(args, optimizer)
        criterion = CosineSimilarityLoss()
        
        trained_partial_model, trained_model_dict = train_model(
            args=args, mode="train", model=partial_model, teacher_model=partial_model_ori,
            criterion=criterion, optimizer=optimizer, loss_scaler=loss_scaler, lr_scheduler=lr_scheduler, 
            train_data=data_loader_train, device=device, n_parameters=n_parameters
            )
        
        complete_model = copy.deepcopy(model_deit)
        trained_model = models.recomplete_model(
            trained_model=trained_partial_model, origin_model=complete_model, repl_blocks=args.replace, 
            grad_train=args.gradually_train, remove_sc=args.rm_shortcut
            )
        save_path = args.output_dir / "model.pth"
        trained_model_dict["model"] = trained_model.state_dict()
        # torch.save(trained_model_dict, save_path)
        torch.save(trained_model, save_path)
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        
        for ft_mode in args.qkv_ft_mode:
            args.gradually_train = False
            
            qkv_ft_model = models.cut_extra_layers(trained_model, max(args.replace))
            model_ori = copy.deepcopy(model_deit)
            partial_model_ori = models.cut_extra_layers(model_ori, max(args.replace))
            models.set_requires_grad(qkv_ft_model, "train", args.replace, ft_mode)
            models.set_requires_grad(partial_model_ori, "train", [], args.rep_mode)
            qkv_ft_model.to(device)
            partial_model_ori.to(device)
            
            print(f"Training {ft_mode} of QKV-trained model")
            args.lr = args.qkv_ft_lr
            optimizer = create_optimizer(args, qkv_ft_model)
            loss_scaler = NativeScaler()
            lr_scheduler, _ = create_scheduler(args, optimizer)
            args.epochs = args.ft_epochs
            fted_partial_model, fted_model_dict = train_model(
                args=args, mode="train", model=qkv_ft_model, teacher_model=partial_model_ori,
                criterion=criterion, optimizer=optimizer, loss_scaler=loss_scaler, lr_scheduler=lr_scheduler, 
                train_data=data_loader_train, device=device, n_parameters=n_parameters
                ) 
            
            complete_model = copy.deepcopy(model_deit)
            trained_model = models.recomplete_model(
                trained_model=fted_partial_model, origin_model=complete_model, repl_blocks=args.replace,
                grad_train=args.gradually_train, remove_sc=False
                )
            save_path = args.output_dir / f"model_{ft_mode}.pth"
            fted_model_dict["model"] = trained_model.state_dict()
            # torch.save(fted_model_dict, save_path)
            torch.save(trained_model, save_path)   
            del optimizer, trained_model, model_deit, partial_model_ori, data_loader_train
            gc.collect()
            torch.cuda.empty_cache()
        
    elif args.finetune and not args.train and not args.eval:
        data_loader_train = load_dataset(args, "train")
        print(f"Finetuning model: {args.ft_model}")
        model = torch.load(args.ft_model)
        model.to(device)
        
        models.set_requires_grad(model, "finetune", list(range(len(model.blocks))))
        if args.ft_mode == "class":
            criterion = torch.nn.CrossEntropyLoss()
            teacher = None
        else:
            model_deit = create_model(
                args.d_model, pretrained=False, num_classes=args.nb_classes, drop_rate=args.drop,
                drop_path_rate=args.drop_path, drop_block_rate=None, img_size=args.input_size
                )
            teacher = models.load_weight(model_deit, args.d_weight)
            teacher.to(device)
            if args.ft_mode == "cosine":
                criterion = CosineSimilarityLoss()
            elif args.ft_mode == "combine":
                criterion == CombinedLoss()
            else:
                raise ValueError("Wrong finetune mode.") 
     
        args.output_dir = Path(args.ft_model[:-18])  
        save_path = args.output_dir / f"model_{args.ft_mode}_ft.pth"
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of finetunable params:', n_parameters)
        
        args.unscale_lr = args.ft_unscale_lr
        args.lr = args.ft_lr
        args.batch_size = args.ft_batch_size
        args.epochs = args.ft_epochs
        args.start_epoch = args.ft_start_epoch
        args.clip_grad = args.ft_clip_grad
        
        if not args.unscale_lr:
            linear_scaled_lr = args.lr * args.batch_size / 512.0
            args.lr = linear_scaled_lr
        optimizer = create_optimizer(args, model)
        loss_scaler = NativeScaler()
        lr_scheduler, _ = create_scheduler(args, optimizer)
        finetuned_model, finetuned_model_dict = train_model(
            args=args, mode=args.ft_mode, model=model, teacher_model=teacher, criterion=criterion, 
            optimizer=optimizer, loss_scaler=loss_scaler, lr_scheduler=lr_scheduler, 
            train_data=data_loader_train, device=device, n_parameters=n_parameters
            )
        
        finetuned_model_dict["model"] = finetuned_model.state_dict()
        # torch.save(trained_model_dict, save_path)
        torch.save(finetuned_model, save_path)
        
    else:
        raise ValueError("Please specify running mode (eval/train/finetune).") 
    

def eval_trained_models(args):
    args.data_set = "IMNET"
    args.data_path = "/home/u17/yuxinr/datasets/"
    args.train = False
    args.eval = True
    model_path = Path(args.output_dir)
    
    args.eval_model = model_path / "model.pth"
    if os.path.exists(args.eval_model):
        print("evaluating NO FT")
        main(args)
        
    args.eval_model = model_path / "model_FC.pth"
    if os.path.exists(args.eval_model):
        print("evaluating FC FT")
        main(args)
        
    args.eval_model = model_path / "model_block.pth"
    if os.path.exists(args.eval_model):
        print("evaluating BLOCK FT")
        main(args)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DeiT -> MLP Mixer", parents=[get_args_parser()])
    args = parser.parse_args()
    
    deit_model = "deit_tiny_patch16_224"
    deit_weight = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
    
    args.d_model = deit_model
    args.d_weight = deit_weight
    args.qkv_ft_mode = ["block"]
    args.lr = 5e-4
    args.batch_size = 2048
    
    # train
    # args.data_set = "IMNET"
    # args.data_path = "/contrib/datasets/ILSVRC2012/"
    # args.train = True
    # args.gradually_train = True
    # args.rm_shortcut = True
    
    # eval
    # args.data_set = "IMNET"
    # args.train = False
    # args.eval = True
    # args.eval_model = "/home/u17/yuxinr/block_distill/model/2024-11-20-16-12/replaced_model_qkvFC_ft.pth"
    # args.sched = "constant"
    
    # finetune
    # args.data_set = "CIFAR"
    # args.train = False
    # args.finetune = True
    # # args.ft_model = "/home/u17/yuxinr/block_distill/model/2024-11-19-19-11/replaced_model.pth"
    # args.ft_lr = 5e-5
    # args.ft_mode = "cosine"
    # # # args.data_path = "/contrib/datasets/ILSVRC2012/"
    # args.ft_epochs = 50
        
    main(args)
    eval_trained_models(args)
