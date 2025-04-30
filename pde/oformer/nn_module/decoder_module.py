import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import numpy as np
from .attention_module import PreNorm, PostNorm, LinearAttention, CrossLinearAttention, FeedForward
from .cnn_module import UpBlock, PeriodicConv2d
from torch.nn.init import xavier_uniform_, orthogonal_
from copy import deepcopy
from .kinet import rand_vec, KINET_DSMC

########################### Transformer start ###########################
class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):

        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

class CrossFormer(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,
                 heads,
                 dim_head,
                 mlp_dim,
                 residual=True,
                 use_ffn=True,
                 use_ln=False,
                 relative_emb=False,
                 scale=1.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 dropout=0.,
                 cat_pos=False,
                 model_name='oformer',
                 ):
        super().__init__()

        self.cross_attn_module = CrossLinearAttention(dim, attn_type,
                                                       heads=heads, dim_head=dim_head, dropout=dropout,
                                                       relative_emb=relative_emb,
                                                       scale=scale,
                                                       relative_emb_dim=relative_emb_dim,
                                                       min_freq=min_freq,
                                                       init_method='orthogonal',
                                                       cat_pos=cat_pos,
                                                       pos_dim=relative_emb_dim,
                                                       use_ln=False
                                                  )
        self.use_ln = use_ln
        self.residual = residual
        self.use_ffn = use_ffn
        self.model_name = model_name

        if self.use_ln:
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

        if self.use_ffn:
            self.ffn = FeedForward(dim, mlp_dim, dropout)
        
        if self.model_name == 'oformer_kinet':
            n_divide = 128
            n = int(128/n_divide)
            self.randvec = rand_vec(16*5233, n_divide, n)

    def forward(self, x, z, x_pos=None, z_pos=None):
        # x in [b n1 c]
        # b, n1, c = x.shape   # coordinate encoding
        # b, n2, c = z.shape   # system encoding
        if self.model_name == 'oformer':
            x = self.cross_attn_module(x, z, x_pos, z_pos) + x
        else:
            v = self.cross_attn_module(x, z, x_pos, z_pos)
            vec = self.randvec()
            x = KINET_DSMC(x, v, vec, training=self.training)

        if self.model_name == 'oformer':
            x = self.ffn(x) + x
        else:
            v = self.ffn(x)
            vec = self.randvec()
            x = KINET_DSMC(x, v, vec, training=self.training)

        return x
########################### Transformer end ###########################


class IrregSTDecoder2D(nn.Module):
    def __init__(self,
                 max_node_type,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 res=200,
                 scale=8,
                 dropout=0.1,
                 model_name='oformer',
                 **kwargs,
                 ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.node_type_embedding = nn.Embedding(max_node_type, latent_channels)

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels // 2, scale=scale),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.combine_layer = nn.Linear(self.latent_channels*2, self.latent_channels, bias=False)

        self.input_dropout = nn.Dropout(dropout)

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                relative_emb=True,
                                                scale=32.,
                                                relative_emb_dim=2,
                                                min_freq=1 / res,
                                                model_name=model_name,
                                                )

        self.mix_layer = LinearAttention(self.latent_channels, 'galerkin',
                                         heads=1, dim_head=self.latent_channels,
                                         relative_emb=True, scale=32,
                                         relative_emb_dim=2,
                                         min_freq=1 / res,
                                         use_ln=False,
                                         )

        self.expand_layer = nn.Linear(self.latent_channels, self.latent_channels*2, bias=False)

        self.propagator = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(self.latent_channels*2),
                           nn.Sequential(
                               nn.Linear(self.latent_channels*3 + 2, self.latent_channels*2, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels*2, self.latent_channels*2, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels*2, self.latent_channels*2, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels * 2, self.latent_channels * 2, bias=False)
                          )])
        ])

        self.out_norm = nn.LayerNorm(self.latent_channels*2)
        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels*3, self.latent_channels*2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.latent_channels, self.out_channels, bias=True))

    def propagate(self, z, z_node, prop_pos):
        for layer in self.propagator:
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), z_node, prop_pos), dim=-1)) + z
        return z

    def decode(self, z, z_node):
        z = self.out_norm(z)
        z = self.to_out(torch.cat((z, z_node), dim=-1))
        return z

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 2]
                prop_node_type,  # [b, n, 1]
                forward_steps,
                input_pos):
        history = []
        x_node = self.node_type_embedding(prop_node_type.squeeze(-1))
        x = self.coordinate_projection.forward(propagate_pos)
        x = self.combine_layer(torch.cat((x, x_node), dim=-1))

        z = self.input_dropout(z)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos) # (bs, n_toke, width)
        z = self.mix_layer.forward(z, propagate_pos) + z
        z = self.expand_layer(z)

        # forward the dynamics in the latent space
        for step in range(forward_steps):
            z = self.propagate(z, x_node, propagate_pos)
            u = self.decode(z, x_node)
            history.append(u)

        history = torch.stack(history, dim=1)  # concatenate in temporal dimension
        return history

    def denormalize(self,
                    x,  # [b, t, n, c]
                    train_set,
                    ):
        # denormalize
        # vel
        x[:, :, :, 0] = x[:, :, :, 0] * train_set.statistics['vel_x_std'] + train_set.statistics['vel_x_mean']
        x[:, :, :, 1] = x[:, :, :, 1] * train_set.statistics['vel_y_std'] + train_set.statistics['vel_y_mean']

        # dns
        x[:, :, :, 2] = x[:, :, :, 2] * train_set.statistics['dns_std'] + train_set.statistics['dns_mean']

        # prs
        x[:, :, :, 3] = x[:, :, :, 3] * train_set.statistics['prs_std'] + train_set.statistics['prs_mean']

        return x
