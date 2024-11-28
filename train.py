import sys
import math
import time
import json
import torch
import utils
import datetime

from typing import Iterable
from timm.utils import accuracy


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


def adjust_weights(epoch, parallel_block, weight_schedule):
    if epoch in weight_schedule:
        new_weight = weight_schedule[epoch]
        parallel_block.attn_weight.data.fill_(new_weight)
        parallel_block.mixer_weight.data.fill_(1.0 - new_weight)


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    # switch to evaluation mode
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    
def train_one_epoch(mode, model: torch.nn.Module, teacher_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0):
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
         
        with torch.cuda.amp.autocast():
            if mode == "train" or mode == "cosine":
                outputs = model.forward_features(samples)
                teacher_output = teacher_model.forward_features(samples)
                loss = criterion(outputs, teacher_output)
            elif mode == "class":
                outputs = model(samples)
                loss = criterion(outputs, targets)
            elif mode == "combine":
                cos_outputs = model.forward_features(samples)
                cos_teacher_output = teacher_model.forward_features(samples)
                cls_outputs = model(samples)
                loss = criterion(cos_outputs, cos_teacher_output, cls_outputs, targets)
                
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    
def train_model(args, mode, model, teacher_model, 
                criterion, optimizer, loss_scaler, lr_scheduler,
                train_data, device, n_parameters):
    
    with (args.output_dir / "log.txt").open("a") as f:
        f.write("Args: " + str(args) + "\n")
        
    if mode == "train":
        checkpoint_path = args.output_dir / "checkpoint.pth"
        print(f"Start training for {args.epochs} epochs")
        
    else:
        checkpoint_path = args.output_dir / "ft_checkpoint.pth"
        print(f"Start finetuning for {args.epochs} epochs")
    
    teacher_model.eval()
    if args.train_in_eval:
        model.eval()
        for blk in args.replace:
            model.blocks[blk].train()
    else:
        model.train()    
        
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.gradually_train:
            for blk in args.replace:
                adjust_weights(epoch, model.blocks[blk], weight_schedule)
            s_weight = float(model.blocks[args.replace[0]].mixer_weight)
            t_weight = float(model.blocks[args.replace[0]].attn_weight)
            print(f"Teacher weight: {t_weight}; student weight: {s_weight}")
        train_stats = train_one_epoch(
            mode, model, teacher_model, criterion, train_data, optimizer,
            device, epoch, loss_scaler,args.clip_grad)

        lr_scheduler.step(epoch)
        
        model_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
            }
        if epoch%10 == 0:
            torch.save(model_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    #  **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        with (args.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    
    if mode == "train":
        print('Training time {}'.format(total_time_str))
        with (args.output_dir / "log.txt").open("a") as f:
            f.write("Training time:" + json.dumps(total_time_str) + "\n")
    else:
        print('Finetuning time {}'.format(total_time_str))
        with (args.output_dir / "log.txt").open("a") as f:
            f.write("Finetuning time:" + json.dumps(total_time_str) + "\n")
            
    return model, model_dict

