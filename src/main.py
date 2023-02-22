import torch
import argparse
import yaml
import numpy as np
import random
import os
import utils

from easydict import EasyDict

from dota import setup_dota, Dota
from metrics import evaluation, print_results
from models import build_cls, build_model_cfg
from optim import build_optimizer
from play import play
from test import test
from train import train


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def parse_configs():
    parser = argparse.ArgumentParser(description='MOVAD implementation')
    # For training and testing
    parser.add_argument('--config',
                        default="cfgs/v1.yml",
                        help='Configuration file.')
    parser.add_argument('--phase',
                        default='train',
                        choices=['train', 'test', 'play'],
                        help='Training or testing or play phase.')
    help_num_workers = 'The number of workers to load dataset. Default: 0'
    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        metavar='N',
                        help=help_num_workers)
    parser.add_argument('--seed',
                        type=int,
                        default=123,
                        metavar='N',
                        help='random seed (default: 123)')
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        metavar='N',
                        help='number of epoches (default: 50)')
    help_snapshot = 'The epoch interval of model snapshot (default: 10)'
    parser.add_argument('--snapshot_interval',
                        type=int,
                        default=10,
                        metavar='N',
                        help=help_snapshot)
    help_epoch = 'The epoch to restart from (training) or to eval (testing).'
    parser.add_argument('--epoch',
                        type=int,
                        default=-1,
                        help=help_epoch)
    parser.add_argument('--output',
                        default='./output/v1',
                        help='Directory where save the output.')
    parser.add_argument('--num_videos',
                        type=int,
                        default=20,
                        metavar='N',
                        help='Number of video to play (phase = play)')
    parser.add_argument('--no_make_video',
                        action='store_true',
                        default=False)
    parser.add_argument('--machine_reading',
                        '-mr',
                        action='store_true',
                        default=False)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.update(vars(args))
    device = torch.device('cuda') if torch.cuda.is_available() \
        else torch.device('cpu')
    cfg.update(device=device)

    return cfg


if __name__ == "__main__":
    # parse input arguments
    cfg = parse_configs()

    # fix random seed
    set_deterministic(cfg.seed)

    traindata_loader, testdata_loader = setup_dota(
            Dota, cfg, num_workers=cfg.num_workers,
            VCL=cfg.get('VCL', None),
            phase=cfg.phase)

    checkpoint = None
    epoch = 0

    if cfg.phase != 'play':
        t_model, mod_kwargs, shape_input = build_model_cfg(cfg)
        model = build_cls(
            cfg, t_model(**mod_kwargs),
            shape_input,
            1 if cfg.phase == 'test' else None
        )

        try:
            checkpoint = utils.load_checkpoint(cfg)
            if cfg.phase != 'play':
                model.load_state_dict(checkpoint['model_state_dict'])

            epoch = checkpoint['epoch'] + 1
        except FileNotFoundError:
            print('no checkpoint found')
            # save info about no checkpoint has been loaded
            cfg._no_checkpoint = True
            if cfg.epoch != -1:
                epoch = cfg.epoch
            # load pretrained if available
            utils.load_pretrained(model, cfg)

    if cfg.phase == 'train':
        optimizer, lr_scheduler = build_optimizer(cfg, model, checkpoint)
        index_loss = 0
        index_guess = 0
        if checkpoint is not None:
            index_guess = checkpoint.get('index_guess', 0)
            index_loss = checkpoint.get('index_loss', 0)
        train(cfg, model, traindata_loader,
              optimizer, lr_scheduler, epoch, index_guess, index_loss)

    elif cfg.phase == 'test':
        filename = utils.get_result_filename(cfg, epoch)
        if not os.path.exists(filename):
            if cfg.get('_no_checkpoint', False):
                # in case you don't have a checkpoint to test
                raise Exception('no checkpoint to test')
            with torch.no_grad():
                test(cfg, model, testdata_loader,
                     epoch, filename)

        content = utils.load_results(filename)

        outputs = content['outputs']
        targets = content['targets']
        toas = content['toas']
        teas = content['teas']

        print_results(cfg, *evaluation(FPS=cfg.FPS, **content))

    elif cfg.phase == 'play':
        play(cfg, testdata_loader)
