import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from .attention_module import PreNorm, StandardAttention, FeedForward, LinearAttention, ReLUFeedForward
from .cnn_module import PeriodicConv2d, PeriodicConv3d, UpBlock
from .kinet import rand_vec, KINET_DSMC
#from .gnn_module import SmoothConvEncoder, SmoothConvDecoder, index_points
#from torch_scatter import scatter
# helpers


########################### Transformer start ###########################
class TransformerCatNoCls(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 use_ln=False,
                 scale=16,     # can be list, or an int
                 dropout=0.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 attention_init='orthogonal',
                 init_gain=None,
                 use_relu=False,
                 cat_pos=False,
                 model_name='oformer',
                 ):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth
        assert len(scale) == depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.use_ln = use_ln
        self.model_name = model_name

        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList([
                    PreNorm(dim, StandardAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim,  FeedForward(dim, mlp_dim, dropout=dropout)
                                  if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout))]),
                )
        else:
            for d in range(depth):
                if scale[d] != -1 or not cat_pos:
                    attn_module = LinearAttention(dim, attn_type,
                                                   heads=heads, dim_head=dim_head, dropout=dropout,
                                                   relative_emb=True, scale=scale[d],
                                                   relative_emb_dim=relative_emb_dim,
                                                   min_freq=min_freq,
                                                   init_method=attention_init,
                                                   init_gain=init_gain,
                                                   use_ln=False,
                                                   )
                else:
                    attn_module = LinearAttention(dim, attn_type,
                                                  heads=heads, dim_head=dim_head, dropout=dropout,
                                                  cat_pos=True,
                                                  pos_dim=relative_emb_dim,
                                                  relative_emb=False,
                                                  init_method=attention_init,
                                                  init_gain=init_gain
                                                  )
                if not use_ln:
                    self.layers.append(
                        nn.ModuleList([
                                        attn_module,
                                        FeedForward(dim, mlp_dim, dropout=dropout)
                                        if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout)
                        ]),
                        )
                else:
                    self.layers.append(
                        nn.ModuleList([
                            nn.LayerNorm(dim),
                            attn_module,
                            nn.LayerNorm(dim),
                            FeedForward(dim, mlp_dim, dropout=dropout)
                            if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout),
                        ]),
                    )
        if self.model_name == 'oformer_kinet':
            n_divide = 128
            n = int(128/n_divide)
            self.randvec = rand_vec(16*5233, n_divide, n)

    def forward(self, x, pos_embedding):
        # x in [b n c], pos_embedding in [b n 2]
        b, n, c = x.shape # (bs, n_toke, width)
        for layer_no, attn_layer in enumerate(self.layers):
            [ln1, attn, ln2, ffn] = attn_layer

            x = ln1(x)
            if self.model_name == 'oformer':
                x = attn(x, pos_embedding) + x
            else:
                v = attn(x, pos_embedding)
                vec = self.randvec()
                x = KINET_DSMC(x, v, vec, training=self.training)

            x = ln2(x)
            if self.model_name == 'oformer':
                x = ffn(x) + x
            else:
                v = ffn(x)
                vec = self.randvec()
                x = KINET_DSMC(x, v, vec, training=self.training)
        return x
########################### Transformer end ###########################


class IrregSTEncoder2D(torch.nn.Module):
    # for time dependent airfoil
    def __init__(self,
                 input_channels,  # how many channels
                 time_window,
                 in_emb_dim,  # embedding dim of token                 (how about 512)
                 out_chanels,
                 max_node_type,
                 heads,
                 depth,  # depth of transformer / how many layers of attention    (4)
                 res,
                 use_ln=True,
                 emb_dropout=0.05,  # dropout of embedding
                 model_name='oformer',
                 ):
        super().__init__()
        self.tw = time_window
        # here, assume the input is in the shape [b, t, n, c]
        self.to_embedding = nn.Sequential(
            Rearrange('b t n c -> b c t n'),
            nn.Conv2d(input_channels, in_emb_dim, kernel_size=(self.tw, 1), stride=(self.tw, 1), padding=(0, 0), bias=False),
            nn.GELU(),
            nn.Conv2d(in_emb_dim, in_emb_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Rearrange('b c 1 n -> b n c'),
        )

        self.node_embedding = nn.Embedding(max_node_type, in_emb_dim)

        self.combine_embedding = nn.Linear(in_emb_dim*2, in_emb_dim, bias=False)

        self.dropout = nn.Dropout(emb_dropout)

        if depth > 4:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', use_ln,
                                                     scale=[32, 16, 8, 8] + [1] * (depth - 4),
                                                     min_freq=1/res,
                                                     attention_init='orthogonal',
                                                     model_name=model_name)
        else:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', use_ln,
                                                     scale=[32] + [16] * (depth - 2) + [1],
                                                     min_freq=1 / res,
                                                     attention_init='orthogonal',
                                                     model_name=model_name)

        self.ln = nn.LayerNorm(in_emb_dim)

        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.ReLU(),
            nn.Linear(in_emb_dim, out_chanels, bias=False),
        )

    def forward(self,
                x,  # [b, t, n, c]
                node_type,  # [b, n, 1]
                input_pos,  # [b, n, 2]
                ):
        x = self.to_embedding(x)
        x_node = self.node_embedding(node_type.squeeze(-1))
        x = self.combine_embedding(torch.cat([x, x_node], dim=-1))
        x_skip = x # (bs, n_toke, width) = (16, 5233, 128)

        x = self.dropout(x)

        x = self.s_transformer.forward(x, input_pos)

        x = self.ln(x + x_skip)

        x = self.to_out(x)

        return x