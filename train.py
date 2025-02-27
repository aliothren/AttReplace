import sys
import math
import time
import json
import torch
import utils
import datetime

from typing import Iterable
from timm.utils import accuracy
from loss import CosineSimilarityLoss, CombinedLoss


@torch.no_grad()
def evaluate_model(data_loader, device, model, teacher_model = None, loss_type="classification"):
    autocast_ctx = torch.amp.autocast(device_type="cuda") if device.type == "cuda" else torch.cpu.amp.autocast(device_type="cpu")
    if loss_type == "classification":
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_type == "similarity":
        criterion = CosineSimilarityLoss()
    else:
        raise ValueError("Invalid evaluation loss type (classification/similarity).") 

    # switch to evaluation mode
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for images, target in metric_logger.log_every(data_loader, 100, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if loss_type == "classification":
            with autocast_ctx:
                output = model(images)
                loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = images.shape[0]
            metric_logger.update(test_class_loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        elif loss_type == "similarity":
            with autocast_ctx:
                output = model.forward_features(images)
                teacher_output = teacher_model.forward_features(images)
                loss = criterion(output, teacher_output)
            metric_logger.update(test_cosine_loss=loss.item())
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    if loss_type == "classification":
        print("Classification loss on test set:")
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.test_class_loss))
    elif loss_type == "similarity":
        print("Similarity loss on test set:")
        print(metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    
def train_one_epoch(mode, model: torch.nn.Module, teacher_model: torch.nn.Module, 
                    replace: list, data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0):
    autocast_ctx = torch.autocast(device_type=device.type)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
         
        with autocast_ctx:
            if mode == "similarity":
                criterion = CosineSimilarityLoss()
                output_student = model(samples)
                output_teacher = teacher_model(samples)
                loss = 0
                for blk in replace:
                    output_block_s = model.blocks[blk].block_output
                    output_block_t = teacher_model.blocks[blk].block_output
                    loss += criterion(output_block_s, output_block_t)
                
            elif mode == "classification":
                criterion = torch.nn.CrossEntropyLoss()
                output = model(samples)
                loss = criterion(output, targets)
                
            elif mode == "combine":
                ce_criterion = torch.nn.CrossEntropyLoss()
                cos_criterion = CosineSimilarityLoss()
                output_student = model(samples)
                output_teacher = teacher_model(samples)
                loss = ce_criterion(output_student, targets)
                for blk in replace:
                    output_block_s = model.blocks[blk].block_output
                    output_block_t = teacher_model.blocks[blk].block_output
                    loss += cos_criterion(output_block_s, output_block_t)
                
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

    
def train_model(args, stage, loss_mode, model, teacher_model, train_data, test_data,
                optimizer, loss_scaler, lr_scheduler, n_parameters):
    
    with (args.output_dir / "log.txt").open("a") as f:
        f.write("Args: " + str(args) + "\n")
    checkpoint_path = args.output_dir / f"{stage}_checkpoint.pth"
    best_checkpoint_path = args.output_dir / f"{stage}_best_checkpoint.pth"
    print(f"Start {stage} training for {args.epochs} epochs")      
    
    if teacher_model is not None:
        teacher_model.eval()
         
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.epochs):
        # Set only target part in training mode
        model.eval()
        for blk in args.replace:
            if stage == "attn":
                model.blocks[blk].attn.train()
            elif stage in ["block", "global"]:
                model.blocks[blk].train()
            else:
                raise ValueError("Invalid training stage (attn/block/global).") 
            
        train_stats = train_one_epoch(
            mode=loss_mode, model=model, teacher_model=teacher_model, 
            replace=args.replace, data_loader=train_data, 
            optimizer=optimizer, device=args.device, epoch=epoch, 
            loss_scaler=loss_scaler, max_norm=args.clip_grad)

        lr_scheduler.step(epoch)
        
        # Save checkpoint model
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
        
        # Evaluate training
        test_stats_class = evaluate_model(test_data, args.device, model, None, "classification") # Classification result on test set
        if loss_mode in ["similarity", "combine"]:
            test_stats_cosine = evaluate_model(test_data, args.device, model, teacher_model, "similarity") # Similarity loss on test set

        # Save best checkpoint
        if max_accuracy < test_stats_class["acc1"]:
            max_accuracy = test_stats_class["acc1"]
            torch.save(model_dict, best_checkpoint_path)
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats_class.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if loss_mode in ["similarity", "combine"]:
            log_stats.update({f'test_cosine_{k}': v for k, v in test_stats_cosine.items()})
        with (args.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"{stage} finished.")
    print('Training time {}'.format(total_time_str))
    with (args.output_dir / "log.txt").open("a") as f:
        f.write("Training time:" + json.dumps(total_time_str) + "\n")
            
    return model, model_dict

