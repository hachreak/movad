import itertools
import math
import numpy as np
import os
import pickle
import torch

from collections import OrderedDict


def get_visual_directory(cfg, epoch):
    output_dir = os.path.join(cfg.output, 'vis-{:02d}'.format(epoch))
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_result_filename(cfg, epoch):
    output_dir = os.path.join(cfg.output, 'eval')
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, 'results-{:02d}.pkl'.format(epoch))


def load_results(filename):
    print('load file {}'.format(filename))
    with open(filename, 'rb') as f:
        content = pickle.load(f)
    return content


def get_last_epoch(filenames):
    epochs = [int(name.split('-')[1].split('.')[0]) for name in filenames]
    return filenames[np.array(epochs).argsort()[-1]]


def load_checkpoint(cfg):
    dir_chk = os.path.join(cfg.output, 'checkpoints')
    # build file path
    if cfg.epoch != -1:
        path = os.path.join(dir_chk, 'model-{:02d}.pt'.format(cfg.epoch))
    else:
        try:
            fnames = os.listdir(dir_chk)
            path = get_last_epoch(fnames)
            path = os.path.join(dir_chk, path)
        except IndexError:
            raise FileNotFoundError()
    # load checkpoint
    if not os.path.exists(path):
        raise FileNotFoundError()
    print('load file {}'.format(path))
    return torch.load(path, map_location=cfg.device)


def load_pretrained(model, cfg):
    pretrained = cfg.get('pretrained', None)
    if pretrained is not None:
        print('loading pretrained {}'.format(pretrained))
        t_type = cfg.get('transformer_model', 'SwinTransformer3D')
        if t_type == 'SwinTransformer3D':
            if cfg.get('pretrained2d', False):
                print('load 2D -> 3D pretrained')
                model.model.init_weights(pretrained=pretrained)
            else:
                print('load 3D pretrained')
                # load file
                checkpoint = torch.load(pretrained)
                # remove backbone key
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    if 'backbone' in k:
                        name = k[9:]
                        new_state_dict[name] = v
                strict = cfg.get('pretrained_strict', True)

                # check if you want only relative_position_index
                if cfg.get('pretrained_only_relative_position_index', False):
                    print('load only relative_position_index keys')
                    strict = False
                    rel_keys = [key for key in new_state_dict.keys()
                                if key.endswith('relative_position_bias_table')
                                ]
                    new_state_dict = {k: v for k, v in new_state_dict.items()
                                      if k in rel_keys}

                # load swin pretrained
                model.model.load_state_dict(new_state_dict, strict=strict)
        else:
            # load 2d -> 3d pretrained
            print('load 2D -> 3D pretrained')
            model.model.init_weights(pretrained=pretrained)


def w_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flat_list(list_):
    if isinstance(list_, (np.ndarray, np.generic)):
        # to be retrocompatible
        return list_
    return list(itertools.chain(*list_))


def filter_by_class(outputs, info, cls):
    return [out for out, inf in zip(outputs, info.tolist()) if inf[0] == cls]


def filter_by_class_ego(outputs, info, cls, ego):
    return [out for out, inf in zip(outputs, info.tolist())
            if all([inf[0] == cls, inf[1] == ego])]


def split_by_class(outputs, targets, info):
    clss = np.unique(info[:, 0]).tolist()
    return {
        cls: {
            'outputs': np.array(
                flat_list(filter_by_class(outputs, info, cls))),
            'targets': np.array(
                flat_list(filter_by_class(targets, info, cls))),
        } for cls in clss
    }


def merge_oo_class(splitted):
    def _cat(v1, v2):
        return np.concatenate([v1, v2], axis=0)

    oo0_outputs = _cat(splitted[(8.0, 0.0)]['outputs'],
                       splitted[(9.0, 0.0)]['outputs'])
    oo1_outputs = _cat(splitted[(8.0, 1.0)]['outputs'],
                       splitted[(9.0, 1.0)]['outputs'])
    oo0_targets = _cat(splitted[(8.0, 0.0)]['targets'],
                       splitted[(9.0, 0.0)]['targets'])
    oo1_targets = _cat(splitted[(8.0, 1.0)]['targets'],
                       splitted[(9.0, 1.0)]['targets'])
    # OO (11)  = OO-r (8) + OO-l (9)
    splitted[(11.0, 0.0)] = {
        'outputs': oo0_outputs,
        'targets': oo0_targets,
    }
    splitted[(11.0, 1.0)] = {
        'outputs': oo1_outputs,
        'targets': oo1_targets,
    }
    return splitted


def split_by_class_ego(outputs, targets, info):
    clss = np.unique(info[:, 0]).tolist()
    egos = np.unique(info[:, 1]).tolist()
    pairs = [itertools.chain(*li.tolist()) for li in np.meshgrid(clss, egos)]
    pairs = zip(*pairs)
    return merge_oo_class({
        (cls, ego): {
            'outputs': np.array(
                flat_list(filter_by_class_ego(outputs, info, cls, ego))),
            'targets': np.array(
                flat_list(filter_by_class_ego(targets, info, cls, ego))),
        } for cls, ego in pairs
    })


def get_abs_weights_grads(model):
    return torch.cat([
            p.grad.detach().view(-1) for p in model.parameters()
            if p.requires_grad
        ]).abs()


def get_abs_weights(model):
    return torch.cat([
        p.detach().view(-1) for p in model.parameters()
        if p.requires_grad
    ]).abs()


def log_vals(writer, model, global_key, key, fun, index_l):
    if w_count(model):
        vals = fun(model)
        writer.add_scalar(
            '{}_mean/{}'.format(global_key, key), vals.mean().item(), index_l)
        writer.add_scalar(
            '{}_std/{}'.format(global_key, key), vals.std().item(), index_l)
        writer.add_scalar(
            '{}_max/{}'.format(global_key, key), vals.max().item(), index_l)


def scan_internal(writer, model, global_key, fun, level, index_l):
    # level 0
    log_vals(writer, model, global_key, 'model', fun, index_l)
    if level > 0:
        for name, param in model.named_children():
            # level 1
            log_vals(
                writer, param, global_key, 'model/{}'.format(name), fun,
                index_l)
            # level 2
            if level > 1:
                for name, param in model.model.named_children():
                    log_vals(
                        writer, param, global_key,
                        'model/model/{}'.format(name), fun, index_l)
            # level 3
            if level > 2:
                for name, param in model.model.layers.named_children():
                    log_vals(
                        writer, param, global_key,
                        'model/model/layers/{}'.format(name), fun, index_l)
                    # level 4
                    if level > 3:
                        for b_name, b_param in param.named_children():
                            full_name = 'model/model/layers/{}/{}'.format(
                                name, b_name)
                            log_vals(writer, b_param, global_key, full_name,
                                     fun, index_l)
                        # level 5
                        if level > 4:
                            for b_name, b_param in param.blocks.named_children():
                                full_name = 'model/model/layers/{}/blocks/{}'.format(
                                    name, b_name)
                                log_vals(writer, b_param, global_key, full_name,
                                         fun, index_l)
                                # level 6
                                if level > 5:
                                    for s_name, s_param in b_param.named_children():
                                        full_name = 'model/model/layers/{}/blocks/{}/{}'.format(
                                            name, b_name, s_name)
                                        log_vals(writer, s_param, global_key, full_name,
                                                 fun, index_l)
                                    # level 7
                                    if level > 6:
                                        for a_name, a_param in b_param.attn.named_children():
                                            full_name = 'model/model/layers/{}/blocks/{}/attn/{}'.format(
                                                name, b_name, a_name)
                                            log_vals(writer, a_param,
                                                     global_key, full_name,
                                                     fun, index_l)


def get_params(model, keys):
    return [param for name, param in model.named_parameters()
            if any([key in name for key in keys])]


def get_params_rest(model, keys):
    return [param for name, param in model.named_parameters()
            if all([key not in name for key in keys])]


def debug_weights(writer, debug_train_weight, debug_train_grad, debug_loss,
                  debug_train_grad_level, debug_train_weight_level,
                  model, optimizer, loss, index_l):
    if debug_train_grad:
        scan_internal(
            writer, model, 'grads', get_abs_weights_grads,
            debug_train_grad_level, index_l)

    if debug_train_weight:
        scan_internal(
            writer, model, 'weights', get_abs_weights,
            debug_train_weight_level, index_l)

    if debug_loss:
        writer.add_scalar(
            'learning_rate', optimizer.param_groups[0]["lr"], index_l)
        writer.add_scalar('losses/all', loss.item(), index_l)


def debug_guess(writer, outputs, targets, index):
    """Debug guess info."""
    f_tp = outputs == targets
    f_t_1 = targets == 1
    f_t_0 = targets == 0

    ok = len(outputs[f_tp])
    tpos = len(outputs[f_t_1])  # totally pos
    tneg = len(outputs[f_t_0])  # totally neg
    cpos = len(outputs[f_tp & f_t_1])  # true pos
    cneg = len(outputs[f_tp & f_t_0])  # true neg
    tot = math.prod(outputs.shape)

    g_all = ok / tot
    g_pos = (cpos / tpos) if tpos > 0 else 0
    g_neg = (cneg / tneg) if tneg > 0 else 0
    writer.add_scalar('guess/all', g_all, index)
    writer.add_scalar('guess/pos', g_pos, index)
    writer.add_scalar('guess/neg', g_neg, index)
