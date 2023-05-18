import datetime
import os
import torch
import yaml

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils

from dota import gt_cls_target
from losses import build_loss


def train(cfg, model, traindata_loader, optimizer,
          lr_scheduler, begin_epoch, index_guess, index_loss):
    # Tensorboard
    writer = SummaryWriter(cfg.output + '/tensorboard/train_{}'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    # backup the config file
    with open(os.path.join(cfg.output, 'cfg.yml'), 'w') as bkfile:
        yaml.dump(cfg, bkfile, default_flow_style=False)

    # debug config
    debug_train_grad = cfg.get('debug_train_grad', False)
    debug_train_weight = cfg.get('debug_train_weight', False)
    debug_train_grad_level = cfg.get('debug_train_grad_level', 0)
    debug_train_weight_level = cfg.get('debug_train_weight_level', 0)
    debug_loss = cfg.get('debug_loss', False)

    criterion = build_loss(cfg)
    rnn_type = cfg.get('rnn_type', 'lstm')
    rnn_cell_num = cfg.get('rnn_cell_num', 1)

    index = index_guess
    index_l = index_loss
    fb = cfg.NF

    model.train(True)
    for e in range(begin_epoch, cfg.epochs):
        for j, (video_data, data_info) in tqdm(
                enumerate(traindata_loader), total=len(traindata_loader),
                desc='Epoch: %d / %d' % (e + 1, cfg.epochs)):
            video_data = video_data.to(cfg.device, non_blocking=True)
            data_info = data_info.to(cfg.device, non_blocking=True)

            # [B, F, C, W, H] -> [B, C, F, W, H]
            video_data = torch.swapaxes(video_data, 1, 2)

            t_shape = (video_data.shape[0], video_data.shape[2] - fb)
            targets = torch.full(t_shape, -100).to(video_data.device)
            outputs = torch.full(
                t_shape, -100, dtype=float).to(video_data.device)

            video_len_orig = data_info[:, 0]
            toa_batch = data_info[:, 2]
            tea_batch = data_info[:, 3]
            v_len = video_data.shape[2]

            rnn_state = None
            if rnn_type == 'gru':
                rnn_state = torch.randn(
                    rnn_cell_num, cfg.batch_size,
                    cfg.rnn_state_size).to(cfg.device)

            # FIXME start from zero!
            for i in range(fb, v_len):
                target = gt_cls_target(i, toa_batch, tea_batch).long()
                x = video_data[:, :, i-fb:i]

                output, rnn_state = model(x, rnn_state)

                # filter frame fillers
                flt = i >= video_len_orig
                target[flt] = -100
                output[flt] = -100

                if cfg.get('apply_softmax', True):
                    output = output.softmax(dim=1)

                optimizer.zero_grad()
                loss = criterion(output, target)
                loss.backward()

                optimizer.step()

                utils.debug_weights(
                    writer, debug_train_weight, debug_train_grad, debug_loss,
                    debug_train_grad_level, debug_train_weight_level,
                    model, optimizer, loss, index_l)

                index_l += 1

                targets[:, i-fb] = target.clone()
                out = output.max(1)[1]
                out[target == -100] = -100
                outputs[:, i-fb] = out

            # filter not selected frames
            outputs = outputs[outputs != -100]
            targets = targets[targets != -100]

            # debug info
            utils.debug_guess(writer, outputs, targets, index)

            index += 1

        # update for scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

        # save checkpoint
        if (e+1) % cfg.snapshot_interval == 0:
            # lr scheduler state
            lr_scheduler_state_dict = None
            if lr_scheduler is not None:
                lr_scheduler_state_dict = lr_scheduler.state_dict()

            dir_chk = os.path.join(cfg.output, 'checkpoints')
            os.makedirs(dir_chk, exist_ok=True)
            path = os.path.join(dir_chk, 'model-{:02d}.pt'.format(e+1))
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler_state_dict,
                'index_guess': index,
                'index_loss': index_l,
            }, path)
