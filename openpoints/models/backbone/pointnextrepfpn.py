"""Official implementation of PointNext
PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies
https://arxiv.org/abs/2206.04670
Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Abed Al Kader Hammoud, Mohamed Elhoseiny, Bernard Ghanem
"""
from re import I
from typing import List, Type
import logging
import torch
import torch.nn as nn
from ..build import MODELS
from ..layers import create_convblock1d, create_convblock2d, create_act, CHANNEL_MAP, \
    create_grouper, furthest_point_sample, random_sample, three_interpolation
import copy

def get_reduction_fn(reduction):
    reduction = 'mean' if reduction.lower() == 'avg' else reduction
    assert reduction in ['sum', 'max', 'mean']
    if reduction == 'max':
        pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
    elif reduction == 'mean':
        pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
    elif reduction == 'sum':
        pool = lambda x: torch.sum(x, dim=-1, keepdim=False)
    return pool


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


class LocalAggregation(nn.Module):
    """Local aggregation layer for a set 
    Set abstraction layer abstracts features from a larger set to a smaller set
    Local aggregation layer aggregates features from the same set
    """

    def __init__(self,
                 channels: List[int],
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 group_args={'NAME': 'ballquery', 'radius': 0.1, 'nsample': 16},
                 conv_args=None,
                 feature_type='dp_fj',
                 reduction='max',
                 last_act=True,
                 **kwargs
                 ):
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        channels1 = channels
        channels2 = copy.copy(channels)
        channels2[0] = 3
        convs1 = []
        convs2 = []
        for i in range(len(channels1) - 1):  # #layers in each blocks
            convs1.append(create_convblock1d(channels1[i], channels1[i + 1],
                                             norm_args=norm_args,
                                            act_args=None if i == (
                                                    len(channels1) - 2) and not last_act else act_args,
                                             **conv_args)
                          )
        for i in range(len(channels2) - 1):  # #layers in each blocks
            convs2.append(create_convblock2d(channels2[i], channels2[i + 1],
                                            norm_args=norm_args,
                                            act_args=None if i == (
                                                    len(channels2) - 2) and not last_act else act_args,
                                            **conv_args)
                         )
        self.convs1 = nn.Sequential(*convs1)
        self.convs2 = nn.Sequential(*convs2)
        self.grouper = create_grouper(group_args)
        self.reduction = reduction.lower()
        self.pool = get_reduction_fn(self.reduction)
        self.feature_type = feature_type

    def forward(self, pf) -> torch.Tensor:
        # p: position, f: feature
        p, f = pf
        # preconv
        f = self.convs1(f)
        # grouping
        dp, fj = self.grouper(p, p, f)
        # conv on neighborhood_dp
        pe = self.convs2(dp)
        # pe + fj 
        f = pe + fj
        f = self.pool(f)
        """ DEBUG neighbor numbers. 
        if f.shape[-1] != 1:
            query_xyz, support_xyz = p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(
                f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
        DEBUG end """
        return f


class SARep(nn.Module):
    """The modified set abstraction module in PointNet++ with residual connection support
    """

    def __init__(self,
                 in_channels, out_channels,
                 layers=1,
                 stride=4,
                 group_args={'NAME': 'ballquery', 'normalize_dp': True, 
                             'radius': 0.1, 'nsample': 32},
                 norm_args={'norm': 'bn'},
                 act_args={'act': 'relu'},
                 conv_args={'order': 'conv-norm-act'},
                 sampler='fps',
                 feature_type='dp_fj',
                 use_res=False,
                 is_head=False,
                 **kwargs, 
                 ):
        super().__init__()
        self.stride = stride
        self.is_head = is_head
        self.all_aggr = not is_head and stride == 1
        self.use_res = use_res and not self.all_aggr and not self.is_head
        self.feature_type = feature_type

        mid_channel = out_channels // 2 if stride > 1 else out_channels
        channels = [in_channels] + [mid_channel] * \
                   (layers - 1) + [out_channels]
        channels[0] = in_channels #if is_head else CHANNEL_MAP[feature_type](channels[0])

        channels1 = channels
        channels2 = copy.copy(channels)
        channels2[0] = 3
        convs1 = []
        convs2 = []

        if self.use_res:
            self.skipconv = create_convblock1d(
                in_channels, channels[-1], norm_args=None, act_args=None) if in_channels != channels[
                -1] else nn.Identity()
            self.act = create_act(act_args)

        # actually, one can use local aggregation layer to replace the following
        for i in range(len(channels1) - 1):  # #layers in each blocks
            convs1.append(create_convblock1d(channels1[i], channels1[i + 1],
                                             norm_args=norm_args if not is_head else None,
                                             act_args=None if i == len(channels) - 2
                                                            and (self.use_res or is_head) else act_args,
                                             **conv_args)
                          )
        self.convs1 = nn.Sequential(*convs1)

        if not is_head:
            for i in range(len(channels2) - 1):  # #layers in each blocks
                convs2.append(create_convblock2d(channels2[i], channels2[i + 1],
                                                 norm_args=norm_args if not is_head else None,
                                                 act_args=None if i == len(channels) - 2
                                                                and (self.use_res or is_head) else act_args,
                                                **conv_args)
                            )
            self.convs2 = nn.Sequential(*convs2)
            if self.all_aggr:
                group_args.nsample = None
                group_args.radius = None
            self.grouper = create_grouper(group_args)
            self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
            if sampler.lower() == 'fps':
                self.sample_fn = furthest_point_sample
            elif sampler.lower() == 'random':
                self.sample_fn = random_sample

    def forward(self, pf):
        p, f = pf
        if self.is_head:
            f = self.convs(f)  # (n, c)
        else:
            if not self.all_aggr:
                idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            else:
                new_p = p
            """ DEBUG neighbor numbers. 
            query_xyz, support_xyz = new_p, p
            radius = self.grouper.radius
            dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
            points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
            logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
            DEBUG end """
            if self.use_res or 'df' in self.feature_type:
                fi = torch.gather(
                    f, -1, idx.unsqueeze(1).expand(-1, f.shape[1], -1))
                if self.use_res:
                    identity = self.skipconv(fi)
            else:
                fi = None
            # preconv
            f = self.convs1(f)
            # grouping
            dp, fj = self.grouper(new_p, p, f)
            # conv on neighborhood_dp
            pe = self.convs2(dp)
            # pe + fj 
            f = pe + fj
            f = self.pool(f)
            if self.use_res:
                f = self.act(f + identity)
            p = new_p
        return p, f


class FeaturePropogation(nn.Module):
    """The Feature Propogation module in PointNet++
    """

    def __init__(self, mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'}
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        if not upsample:
            self.linear2 = nn.Sequential(
                nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
            mlp[1] *= 2
            linear1 = []
            for i in range(1, len(mlp) - 1):
                linear1.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                  norm_args=norm_args, act_args=act_args
                                                  ))
            self.linear1 = nn.Sequential(*linear1)
        else:
            convs = []
            for i in range(len(mlp) - 1):
                convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                norm_args=norm_args, act_args=act_args
                                                ))
            self.convs = nn.Sequential(*convs)

        self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

    def forward(self, pf1, pf2=None):
        # pfb1 is with the same size of upsampled points
        if pf2 is None:
            _, f = pf1  # (B, N, 3), (B, C, N)
            f_global = self.pool(f)
            f = torch.cat(
                (f, self.linear2(f_global).unsqueeze(-1).expand(-1, -1, f.shape[-1])), dim=1)
            f = self.linear1(f)
        else:
            p1, f1 = pf1
            p2, f2 = pf2
            if f1 is not None:
                f = self.convs(
                    torch.cat((f1, three_interpolation(p1, p2, f2)), dim=1))
            else:
                f = self.convs(three_interpolation(p1, p2, f2))
        return f


class InvResMLP(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 num_posconvs=2,
                 less_act=False,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = int(in_channels * expansion)
        self.convs = LocalAggregation([in_channels, in_channels],
                                      norm_args=norm_args, act_args=act_args if num_posconvs > 0 else None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        if num_posconvs < 1:
            channels = []
        elif num_posconvs == 1:
            channels = [in_channels, in_channels]
        else:
            channels = [in_channels, mid_channels, in_channels]
        pwconv = []
        # point wise after depth wise conv (without last layer)
        for i in range(len(channels) - 1):
            pwconv.append(create_convblock1d(channels[i], channels[i + 1],
                                             norm_args=norm_args,
                                             act_args=act_args if
                                             (i != len(channels) - 2) and not less_act else None,
                                             **conv_args)
                          )
        self.pwconv = nn.Sequential(*pwconv)
        self.act = create_act(act_args)

    def forward(self, pf):
        p, f = pf
        identity = f
        f = self.convs([p, f])
        f = self.pwconv(f)
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f]


class ResBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 use_res=True,
                 **kwargs
                 ):
        super().__init__()
        self.use_res = use_res
        mid_channels = in_channels * expansion
        self.convs = LocalAggregation([in_channels, in_channels, mid_channels, in_channels],
                                      norm_args=norm_args, act_args=None,
                                      group_args=group_args, conv_args=conv_args,
                                      **aggr_args, **kwargs)
        self.act = create_act(act_args)

    def forward(self, pf):
        p, f = pf
        identity = f
        f = self.convs([p, f])
        if f.shape[-1] == identity.shape[-1] and self.use_res:
            f += identity
        f = self.act(f)
        return [p, f]


class UpPath(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_stages: int = 4, 
                 **kwargs
                 ):
        super().__init__()
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        if len(skip_channels) < decoder_stages:
            skip_channels.insert(0, kwargs.get('in_channels', 3))
        # the output channel after interpolation
        fp_channels = encoder_channel_list[:decoder_stages]

        n_decoder_stages = len(fp_channels)
        decoder = [[] for _ in range(n_decoder_stages)]
        for i in range(-1, -n_decoder_stages - 1, -1):
            decoder[i] = self._make_dec(
                skip_channels[i], fp_channels[i])
        self.decoder = nn.Sequential(*decoder)
        self.out_channels = fp_channels[-n_decoder_stages]

    def _make_dec(self, skip_channels, fp_channels):
        layers = []
        mlp = [skip_channels + self.in_channels] + \
              [fp_channels] * self.decoder_layers
        layers.append(FeaturePropogation(mlp))
        self.in_channels = fp_channels
        return nn.Sequential(*layers)

    def forward(self, p, f):
        for i in range(-1, -len(self.decoder) - 1, -1):
            f[i - 1] = self.decoder[i][1:](
                [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]   
        return f


class DownPath(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 encoder_layers: int = 2,
                 encoder_stages: int = 4, 
                 **kwargs
                 ):
        super().__init__()
        self.encoder_layers = encoder_layers
        inp_dim_list = encoder_channel_list[:encoder_stages]
        out_dim_list = encoder_channel_list[1:]

        n_encoder_stages = len(out_dim_list)
        encoder = [[] for _ in range(n_encoder_stages)]
        for i in range(n_encoder_stages):
            encoder[i] = self._make_enc(
                inp_dim_list[i], out_dim_list[i])
        self.encoder = nn.Sequential(*encoder)
        self.out_channels = inp_dim_list[0]

    def _make_enc(self, inp_dim, out_dim):
        layers = []
        layers.append(SARep(inp_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, p, f):
        for i in range(len(self.encoder)):
            f[i+2] = f[i+2] + self.encoder[i]([p[i+1], f[i+1]])[1]
            # f[i - 1] = self.decoder[i][1:](
            #     [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]   
        return f


class TransformSize(nn.Module):
    """Transform the channels number and points number
    """

    def __init__(self, mlp,
                 upsample=True,
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'}
                 ):
        """
        Args:
            mlp: [current_channels, next_channels, next_channels]
            out_channels:
            norm_args:
            act_args:
        """
        super().__init__()
        convs = []
        for i in range(len(mlp) - 1):
            convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                            norm_args=norm_args, act_args=act_args
                                            ))
        self.convs = nn.Sequential(*convs)

    def forward(self, pf1, pf2=None):
        # pf1 is with the same size of upsampled points
        # pf2 is the points to be transformed
        p1, _ = pf1
        p2, f2 = pf2
        f2 = self.convs(f2)
        f = three_interpolation(p1, p2, f2)
        return f


@MODELS.register_module()
class FPthenAllConcatDecoder256(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_stages: int = 4, 
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 **kwargs
                 ):
        super().__init__()
        self.up_path = UpPath(encoder_channel_list)
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        if len(skip_channels) < decoder_stages:
            skip_channels.insert(0, kwargs.get('in_channels', 3))
        # the output channel after interpolation
        fp_channels = encoder_channel_list[:decoder_stages]
        n_decoder_stages = len(fp_channels)
        transize = [[] for _ in range(n_decoder_stages)]
        self.out_channels = fp_channels[-n_decoder_stages]
        for i in range(-1, -n_decoder_stages - 1, -1):
            # transize[i] = self._make_TranSize(encoder_channel_list[i], min(encoder_channel_list[i], 256))
            transize[i] = self._make_TranSize(encoder_channel_list[i], min(encoder_channel_list[i], encoder_channel_list[-2]))
        self.transize = nn.Sequential(*transize)
        convs_fuse = []
        # mlp_fuse = [32+64+128+256*2, 64, self.out_channels]
        mlp_fuse = [sum(encoder_channel_list[:-1])+encoder_channel_list[-2], 2*self.out_channels, self.out_channels]
        for i in range(len(mlp_fuse) - 1):
            convs_fuse.append(create_convblock1d(mlp_fuse[i], mlp_fuse[i + 1],
                                            norm_args=norm_args, act_args=act_args
                                            ))
        self.convs_fuse = nn.Sequential(*convs_fuse)

    def _make_TranSize(self, curC, tranC):
        layers = []
        mlp = [curC] + [tranC] 
        layers.append(TransformSize(mlp))
        return nn.Sequential(*layers)

    def forward(self, p, f):
        f = self.up_path(p, f)
        for i in range(-1, -len(self.transize) - 1, -1):
            f[i] = self.transize[i][0]([p[1],f[1]], [p[i],f[i]])
        f_cat = torch.cat(f[1:], dim=1)
        f_out = self.convs_fuse(f_cat)        
        return f_out

@MODELS.register_module()
class UpDownthenAllConcatDecoder256(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_stages: int = 4, 
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
                 **kwargs
                 ):
        super().__init__()
        self.up_path = UpPath(encoder_channel_list)
        self.down_path = DownPath(encoder_channel_list)
        self.decoder_layers = decoder_layers
        self.in_channels = encoder_channel_list[-1]
        skip_channels = encoder_channel_list[:-1]
        if len(skip_channels) < decoder_stages:
            skip_channels.insert(0, kwargs.get('in_channels', 3))
        # the output channel after interpolation
        fp_channels = encoder_channel_list[:decoder_stages]
        n_decoder_stages = len(fp_channels)
        transize = [[] for _ in range(n_decoder_stages)]
        self.out_channels = fp_channels[-n_decoder_stages]
        for i in range(-1, -n_decoder_stages - 1, -1):
            # transize[i] = self._make_TranSize(encoder_channel_list[i], min(encoder_channel_list[i], 256))
            transize[i] = self._make_TranSize(encoder_channel_list[i], min(encoder_channel_list[i], encoder_channel_list[-2]))
        self.transize = nn.Sequential(*transize)
        convs_fuse = []
        # mlp_fuse = [32+64+128+256*2, 64, self.out_channels]
        mlp_fuse = [sum(encoder_channel_list[:-1])+encoder_channel_list[-2], 2*self.out_channels, self.out_channels]
        for i in range(len(mlp_fuse) - 1):
            convs_fuse.append(create_convblock1d(mlp_fuse[i], mlp_fuse[i + 1],
                                            norm_args=norm_args, act_args=act_args
                                            ))
        self.convs_fuse = nn.Sequential(*convs_fuse)

    def _make_TranSize(self, curC, tranC):
        layers = []
        mlp = [curC] + [tranC] 
        layers.append(TransformSize(mlp))
        return nn.Sequential(*layers)

    def forward(self, p, f):
        f = self.up_path(p, f)
        f = self.down_path(p, f)
        for i in range(-1, -len(self.transize) - 1, -1):
            f[i] = self.transize[i][0]([p[1],f[1]], [p[i],f[i]])
        f_cat = torch.cat(f[1:], dim=1)
        f_out = self.convs_fuse(f_cat)        
        return f_out

