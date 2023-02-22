import torch


def build_loss(cfg):
    loss_type = cfg.get('loss_type', 'CrossEntropyLoss')
    if loss_type != 'CrossEntropyLoss':
        raise Exception('Not supported loss {}'.format(loss_type))
    smoothing = cfg.get('smoothing', 0.0)
    return torch.nn.CrossEntropyLoss(
        weight=torch.tensor(
            eval(cfg.class_weights)
        ).to(cfg.device), label_smoothing=smoothing)
