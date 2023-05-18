import datetime
import torch
import numpy as np
import pickle

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from losses import build_loss
from dota import gt_cls_target


def test(cfg, model, testdata_loader, epoch,
         filename):
    # Tensorboard
    writer = SummaryWriter(cfg.output + '/tensorboard/eval_{}'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    targets_all = []
    outputs_all = []
    toas_all = []
    teas_all = []
    idxs_all = []
    info_all = []
    frames_counter = []

    rnn_type = cfg.get('rnn_type', 'lstm')
    rnn_cell_num = cfg.get('rnn_cell_num', 1)

    index_l = 0
    criterion = build_loss(cfg)
    fb = cfg.NF
    model.eval()
    for j, (video_data, data_info) in tqdm(
            enumerate(testdata_loader), total=len(testdata_loader),
            desc='Epoch: %d / %d' % (epoch, cfg.epochs)):
        video_data = video_data.to(cfg.device, non_blocking=True)
        data_info = data_info.to(cfg.device, non_blocking=True)

        # [B, F, C, W, H] -> [B, C, F, W, H]
        video_data = torch.swapaxes(video_data, 1, 2)

        t_shape = (video_data.shape[0], video_data.shape[2] - fb)
        targets = torch.full(t_shape, -100).to(video_data.device)
        outputs = torch.full(t_shape, -100, dtype=float).to(video_data.device)

        idx_batch = data_info[:, 1]
        toa_batch = data_info[:, 2]
        tea_batch = data_info[:, 3]
        info_batch = data_info[:, 7:11]
        rnn_state = None

        if rnn_type == 'gru':
            rnn_state = torch.randn(
                rnn_cell_num, 1, cfg.rnn_state_size).to(cfg.device)

        # FIXME start from zero!
        for i in range(fb, video_data.shape[2]):
            target = gt_cls_target(i, toa_batch, tea_batch).long()
            x = video_data[:, :, i-fb:i]

            output, rnn_state = model(x, rnn_state)

            if cfg.get('apply_softmax', True):
                output = output.softmax(dim=1)

            loss = criterion(output, target)
            writer.add_scalar('losses/test', loss.item(), index_l)
            index_l += 1

            if not cfg.get('apply_softmax', True):
                output = output.softmax(dim=1)

            targets[:, i-fb] = target.clone()
            outputs[:, i-fb] = output[:, 1].clone()

        # collect results for each video
        targets_all.append(targets.view(-1).tolist())
        outputs_all.append(outputs.view(-1).tolist())
        toas_all.append(toa_batch.tolist())
        teas_all.append(tea_batch.tolist())
        idxs_all.append(idx_batch.tolist())
        info_all.append(info_batch.tolist())
        frames_counter.append(video_data.shape[2])

    # collect results for all dataset
    toas_all = np.array(toas_all).reshape(-1)
    teas_all = np.array(teas_all).reshape(-1)
    idxs_all = np.array(idxs_all).reshape(-1)
    info_all = np.array(info_all).reshape(-1, 4)
    frames_counter = np.array(frames_counter).reshape(-1)

    print('save file {}'.format(filename))
    with open(filename, 'wb') as f:
        pickle.dump({
            'targets': targets_all,
            'outputs': outputs_all,
            'toas': toas_all,
            'teas': teas_all,
            'idxs': idxs_all,
            'info': info_all,
            'frames_counter': frames_counter,
        }, f)
