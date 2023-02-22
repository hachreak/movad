import torch
import math

from torch import nn
from torch.nn import functional as F

from video_swin_transformer import SwinTransformer3D


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    if isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class Cls(nn.Module):
    def __init__(
            self, model, shape_input, shape_output, dim_latent,
            dropout=0.5,
            rnn_state_size=None,
            batch_size=None,
            rnn_cell_num=1,
            rnn_type='lstm',
            avg_pool_shape=(1, 6, 6)):
        super(Cls, self).__init__()
        self.shape_input = shape_input
        self.shape_output = shape_output
        self.rnn_state_size = rnn_state_size
        self.rnn_type = rnn_type

        self.has_rnn_state = rnn_state_size is not None
        if self.has_rnn_state:
            rnn_cls = nn.LSTM
            if rnn_type == 'gru':
                rnn_cls = nn.GRU

            self.rnn = rnn_cls(dim_latent, rnn_state_size, rnn_cell_num)
            self.rnn_bn = nn.LayerNorm(dim_latent)

        self.dim_state = self.shape_input[0] * math.prod(avg_pool_shape)

        self.lin1 = nn.Linear(self.dim_state, dim_latent)
        self.lin2 = nn.Linear(rnn_state_size or dim_latent, dim_latent)
        self.lin3 = nn.Linear(dim_latent, self.shape_output)
        self.downsize = nn.AdaptiveAvgPool3d(avg_pool_shape)
        self.bn = nn.LayerNorm(self.dim_state)
        self.drop = nn.Dropout(dropout)

        # init weights bfore attach the vision transformer
        self.apply(weights_init_)

        self.model = model

    def forward(self, x, rnn_state=None):
        x = self.model(x)
        x = self.downsize(x)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.bn(x)
        x = F.relu(self.lin1(x))
        x = self.drop(x)

        if self.has_rnn_state:
            x = self.rnn_bn(x)
            # trick for pytorch v1.10
            x = x.unsqueeze(0)
            x, rnn_state = self.rnn(x, rnn_state)
            # trick for pytorch v1.10
            x = x.squeeze(0)

            if self.rnn_type == 'lstm':
                hx, cx = rnn_state
                # default behaviour
                rnn_state = (hx.detach(), cx.detach())
            else:
                # GRU
                rnn_state = rnn_state.detach()

        x = F.relu(self.lin2(x))
        x = self.drop(x)
        x = self.lin3(x)

        return x, rnn_state


def build_cls(cfg, model, shape_input, batch_size=None):
    shape_output = 2
    dim_latent = cfg.get('dim_latent', 1024)
    dropout = cfg.get('dropout', 0.5)
    rnn_state_size = cfg.get('rnn_state_size', None)
    batch_size = batch_size or cfg.batch_size
    rnn_cell_num = cfg.get('rnn_cell_num', 1)
    rnn_type = cfg.get('rnn_type', 'lstm')
    avg_pool_shape = (1, 6, 6)

    return Cls(
        model, shape_input, shape_output,
        dim_latent,
        dropout=dropout,
        rnn_state_size=rnn_state_size,
        batch_size=batch_size,
        rnn_cell_num=rnn_cell_num,
        rnn_type=rnn_type,
        avg_pool_shape=avg_pool_shape,
    ).to(cfg.device)


def build_model_cfg(cfg):
    # t_outshape should be incremented each time NF is
    # over patch depth
    t_type = cfg.get('transformer_type', 'swin-t')
    t_outshape = ((cfg.NF-1) // 4) + 1
    # hw_outshape
    hw_outshape = [15, 20]
    # depths
    depths = cfg.get('depths', None)

    if t_type == 'swin_base_patch4_window7_224_22k':
        shape_input = [1024, t_outshape] + hw_outshape
        mod_kwargs = {
            'depths': depths or [2, 2, 18, 2],
            'embed_dim': 128,
            'num_heads': [4, 8, 16, 32],
            'drop_path_rate': 0.3,
            'patch_size': (4, 4, 4),
            'window_size': (7, 7, 7),
            'patch_norm': True,
        }
    elif t_type == 'swin_base_patch244_window1677_sthv2':
        shape_input = [1024, t_outshape] + hw_outshape
        mod_kwargs = {
            'depths': depths or [2, 2, 18, 2],
            'embed_dim': 128,
            'num_heads': [4, 8, 16, 32],
            'drop_path_rate': 0.3,
            'patch_size': (2, 4, 4),
            'window_size': (16, 7, 7),
            'patch_norm': True,
        }
    elif t_type == 'swin-s':
        shape_input = [768, t_outshape] + hw_outshape
        mod_kwargs = {
            'embed_dim': 96,
            'depths': depths or [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
            'drop_path_rate': 0.2,
        }
    elif t_type == 'swin-b':
        shape_input = [1024, t_outshape] + hw_outshape
        mod_kwargs = {
            'embed_dim': 128,
            'depths': depths or [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'drop_path_rate': 0.3,
        }
    elif t_type == 'swin-l':
        shape_input = [1536, t_outshape] + hw_outshape
        mod_kwargs = {
            'embed_dim': 192,
            'depths': depths or [2,  2, 18, 2],
            'num_heads': [6, 12, 24, 48],
            'drop_path_rate': 0.3,
        }
    else:  # swin-t
        shape_input = [768, t_outshape] + hw_outshape
        mod_kwargs = {
            'embed_dim': 96,
            'depths': depths or [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'drop_path_rate': 0.1,
        }

    mod_kwargs['in_chans'] = 3

    t_model = SwinTransformer3D
    c_model = cfg.get('transformer_model', 'SwinTransformer3D')
    if c_model != 'SwinTransformer3D':
        raise Exception('Model {} not supported!'.format(c_model))

    return t_model, mod_kwargs, shape_input
