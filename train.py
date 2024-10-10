import sys
import math
import time
import json
import torch
import datetime

import utils

from typing import Iterable, Optional


def evaluate():
    ...
    
    
def train_one_epoch(model: torch.nn.Module, teacher_model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True, args = None):
    
    model.train(set_training_mode)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    for samples, _ in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
         
        with torch.cuda.amp.autocast():
            outputs = model.forward_features(samples)
            targets = teacher_model.forward_features(samples)
            loss = criterion(outputs, targets)
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

    
def train_model(args, model, teacher_model, 
                criterion, optimizer, loss_scaler, lr_scheduler,
                train_data, test_data, test_dataset,
                device, n_parameters):
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, teacher_model, criterion, train_data,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
        )

        lr_scheduler.step(epoch)
        
        if epoch%10 == 0:
            checkpoint_path = args.output_dir / 'checkpoint.pth'
            checkpoint_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
            }
            torch.save(checkpoint_dict, checkpoint_path)

        # test_stats = evaluate(test_data, model, device)
        # print(f"Accuracy of the network on the {len(test_dataset)} test images: {test_stats['acc1']:.1f}%")
        
        # if max_accuracy < test_stats["acc1"]:
        #     max_accuracy = test_stats["acc1"]
        #     checkpoint_path = args.output_dir / 'best_checkpoint.pth'
        #     checkpoint_dict = {
        #     'model': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     'lr_scheduler': lr_scheduler.state_dict(),
        #     'epoch': epoch,
        #     'scaler': loss_scaler.state_dict(),
        #     'args': args,
        #     }
        #     torch.save(checkpoint_dict, checkpoint_path)
            
        # print(f'Max accuracy: {max_accuracy}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    #  **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        with (args.output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    with (args.output_dir / "log.txt").open("a") as f:
        f.write("Training time:" + json.dumps(total_time_str) + "\n")
    
