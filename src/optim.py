import torch


def build_optimizer(cfg, model, checkpoint):
    lr_scheduler = None
    lr = cfg.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return optimizer, lr_scheduler
