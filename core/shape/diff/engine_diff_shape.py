# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch

sys.path.append('../../../..')
sys.path.append('')
import util.misc as misc
import util.lr_sched as lr_sched
from util.console import print

def train_one_epoch(model: torch.nn.Module, ae: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (model_data, meta_info) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        surface = model_data['surface']
        cond = model_data['inputs']
        surface = surface.to(device, non_blocking=True)
        cond = cond.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():

                ae_out = ae.encode(surface)
                _, x = ae_out
 
            loss = criterion(model, x, cond = cond)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, ae, criterion, device, args):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    batch_cnt = 0
    for (model_data, meta_info) in metric_logger.log_every(data_loader, 10, header):
        batch_cnt += 1
        points = model_data['points']
        labels = model_data['labels']
        surface = model_data['surface']
        cond = model_data['inputs']
        if batch_cnt % 1 == 0:
            surface = surface.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # compute output

            with torch.cuda.amp.autocast(enabled=False):
                with torch.no_grad():

                    ae_out = ae.encode(surface)
                    _, x = ae_out
                    shape_sampled_array = model.sample(device, cond=cond).float() #  torch.Size([17, 512, 8])
                    output = ae.decode(shape_sampled_array,points)

                loss = criterion(model, x, cond = cond)
                
            batch_size = surface.shape[0]
            
            metric_logger.update(loss=loss.item())

            pred = torch.zeros_like(output.squeeze(-1))
            pred[output.squeeze(-1) >= 0] = 1
            intersection = (pred * labels).sum(dim=1)
            union = (pred + labels).gt(0).sum(dim=1)
            iou = intersection * 1.0 / union + 1e-5
            iou = iou.mean()
            metric_logger.update(iou=iou.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(metric_logger)
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}