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
    create_grouper, furthest_point_sample, random_sample, three_interpolation, create_norm
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


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features, hidden_features=None, out_features=None,
                 act_args={'act': "gelu"}, norm_args=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = create_act(act_args)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        # this is different from what I understand before. 
        # the num_heads here actually works as groups for shared attentions. it partition the channels to different groups.
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple), shape [B, #Heads, N, C]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TRBlock0(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 att_args={'att_type': 'Attention', "num_heads": 6},
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 num_posconvs=2,
                 less_act=False,
                 **kwargs
                 ):
        super().__init__()
        peconv = []
        pe_channels = [3, 32, in_channels]
        for i in range(len(pe_channels) - 1): # last conv has no act
            peconv.append(create_convblock1d(pe_channels[i], pe_channels[i + 1],
                                                norm_args=norm_args,
                                                act_args=act_args if
                                                (i != len(pe_channels) - 2) and not less_act else None,
                                                **conv_args)
                            )
        self.peconv = nn.Sequential(*peconv)
        mid_channels = int(in_channels * expansion)
        AttentionBlock = eval(att_args['att_type'])
        self.attn = AttentionBlock(in_channels, num_heads=att_args['num_heads'])
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
        pe = self.peconv(p.transpose(1,2))
        f = f + pe
        f = f.transpose(1, 2)
        f = f + self.attn(f)
        f = f.transpose(1, 2)
        f = f + self.pwconv(f)
        f = self.act(f)
        return [p, f]


class TRBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 norm_args=None,
                 act_args=None,
                 att_args={'att_type': 'Attention', "num_heads": 6},
                 aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args={'NAME': 'ballquery'},
                 conv_args=None,
                 expansion=1,
                 num_posconvs=2,
                 less_act=False,
                 **kwargs
                 ):
        super().__init__()
        peconv = []
        pe_channels = [3, 32, in_channels]
        for i in range(len(pe_channels) - 1): # last conv has no act
            peconv.append(create_convblock1d(pe_channels[i], pe_channels[i + 1],
                                                norm_args=norm_args,
                                                act_args=act_args if
                                                (i != len(pe_channels) - 2) and not less_act else None,
                                                **conv_args)
                            )
        self.peconv = nn.Sequential(*peconv)
        mid_channels = int(in_channels * expansion)
        AttentionBlock = eval(att_args['att_type'])
        self.attn = AttentionBlock(in_channels, num_heads=att_args['num_heads'])
        self.mlp = Mlp(in_features=in_channels, hidden_features=mid_channels, act_args=act_args)
        norm_args={'norm': 'ln', 'eps': 1.0e-6}
        self.norm1 = create_norm(norm_args, in_channels)
        self.norm2 = create_norm(norm_args, in_channels)

    def forward(self, pf):
        p, f = pf
        pe = self.peconv(p.transpose(1,2))
        f = f + pe
        f = f.transpose(1, 2)
        f = f + self.attn(self.norm1(f))
        f = f + self.mlp(self.norm2(f))
        f = f.transpose(1, 2)
        return [p, f]


@MODELS.register_module()
class LongRange(nn.Module):
    r"""The Encoder for PointNext 
    `"PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies".
    <https://arxiv.org/abs/2206.04670>`_.
    .. note::
        For an example of using :obj:`PointNextEncoder`, see
        `examples/segmentation/main.py <https://github.com/guochengqian/PointNeXt/blob/master/cfgs/s3dis/README.md>`_.
    Args:
        in_channels (int, optional): input channels . Defaults to 4.
        width (int, optional): width of network, the output mlp of the stem MLP. Defaults to 32.
        blocks (List[int], optional): # of blocks per stage (including the SA block). Defaults to [1, 4, 7, 4, 4].
        strides (List[int], optional): the downsampling ratio of each stage. Defaults to [4, 4, 4, 4].
        block (strorType[InvResMLP], optional): the block to use for depth scaling. Defaults to 'InvResMLP'.
        nsample (intorList[int], optional): the number of neighbors to query for each block. Defaults to 32.
        radius (floatorList[float], optional): the initial radius. Defaults to 0.1.
        aggr_args (_type_, optional): the args for local aggregataion. Defaults to {'feature_type': 'dp_fj', "reduction": 'max'}.
        group_args (_type_, optional): the args for grouping. Defaults to {'NAME': 'ballquery'}.
        norm_args (_type_, optional): the args for normalization layer. Defaults to {'norm': 'bn'}.
        act_args (_type_, optional): the args for activation layer. Defaults to {'act': 'relu'}.
        expansion (int, optional): the expansion ratio of the InvResMLP block. Defaults to 4.
        sa_layers (int, optional): the number of MLP layers to use in the SA block. Defaults to 1.
        sa_use_res (bool, optional): wheter to use residual connection in SA block. Set to True only for PointNeXt-S. 
    """

    def __init__(self,
                 in_channels: int = 4,
                 width: int = 32,
                 blocks: List[int] = [1, 4, 7, 4, 4],
                 strides: List[int] = [4, 4, 4, 4],
                 block: str or Type[InvResMLP] = 'InvResMLP',
                 nsample: int or List[int] = 32,
                 radius: float or List[float] = 0.1,
                 att_args: dict = {'att_type': 'Attention', "num_heads": 6},
                 aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                 group_args: dict = {'NAME': 'ballquery'},
                 sa_layers: int = 1,
                 sa_use_res: bool = False,
                 **kwargs
                 ):
        super().__init__()
        if isinstance(block, str):
            block = eval(block)
        self.blocks = blocks
        self.strides = strides
        self.in_channels = in_channels
        self.att_args = att_args
        self.aggr_args = aggr_args
        self.norm_args = kwargs.get('norm_args', {'norm': 'bn'}) 
        self.act_args = kwargs.get('act_args', {'act': 'relu'}) 
        self.conv_args = kwargs.get('conv_args', None)
        self.sampler = kwargs.get('sampler', 'fps')
        self.expansion = kwargs.get('expansion', 4)
        self.sa_layers = sa_layers
        self.sa_use_res = sa_use_res
        self.use_res = kwargs.get('use_res', True)
        radius_scaling = kwargs.get('radius_scaling', 2)
        nsample_scaling = kwargs.get('nsample_scaling', 1)

        self.radii = self._to_full_list(radius, radius_scaling)
        self.nsample = self._to_full_list(nsample, nsample_scaling)
        logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

        # double width after downsampling.
        channels = []
        for stride in strides:
            if stride != 1:
                width *= 2
            channels.append(width)
        encoder = []
        for i in range(len(blocks)):
            group_args.radius = self.radii[i]
            group_args.nsample = self.nsample[i]
            encoder.append(self._make_enc(
                block, channels[i], blocks[i], stride=strides[i], group_args=group_args,
                is_head=i == 0 and strides[i] == 1
            ))
        self.encoder = nn.Sequential(*encoder)
        self.out_channels = channels[-1]
        self.channel_list = channels

    def _to_full_list(self, param, param_scaling=1):
        # param can be: radius, nsample
        param_list = []
        if isinstance(param, List):
            # make param a full list
            for i, value in enumerate(param):
                value = [value] if not isinstance(value, List) else value
                if len(value) != self.blocks[i]:
                    value += [value[-1]] * (self.blocks[i] - len(value))
                param_list.append(value)
        else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
            for i, stride in enumerate(self.strides):
                if stride == 1:
                    param_list.append([param] * self.blocks[i])
                else:
                    param_list.append(
                        [param] + [param * param_scaling] * (self.blocks[i] - 1))
                    param *= param_scaling
        return param_list

    def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
        layers = []
        radii = group_args.radius
        nsample = group_args.nsample
        self.in_channels = channels
        for i in range(0, blocks):
            group_args.radius = radii[i]
            group_args.nsample = nsample[i]
            layers.append(block(self.in_channels,
                                att_args=self.att_args,
                                aggr_args=self.aggr_args,
                                norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                conv_args=self.conv_args, expansion=self.expansion,
                                use_res=self.use_res
                                ))
        return nn.Sequential(*layers)

    def forward_cls_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        for i in range(0, len(self.encoder)):
            p0, f0 = self.encoder[i]([p0, f0])
        return f0.squeeze(-1)

    def forward_seg_feat(self, p0, f0=None):
        if hasattr(p0, 'keys'):
            p0, f0 = p0['pos'], p0.get('x', None)
        if f0 is None:
            f0 = p0.clone().transpose(1, 2).contiguous()
        p, f = [p0], [f0]
        for i in range(0, len(self.encoder)):
            _p, _f = self.encoder[i]([p[-1], f[-1]])
            p.append(_p)
            f.append(_f)
        return p, f

    def forward(self, p0, f0=None):
        return self.forward_seg_feat(p0, f0)


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
class All4ConcatDecoder256_64_LongRange(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_stages: int = 4, 
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
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
        concat_num = 4
        transize = [[] for _ in range(concat_num-1)]
        self.out_channels = fp_channels[-n_decoder_stages]
        for i in range(-1, -concat_num, -1):
            # transize[i] = self._make_TranSize(encoder_channel_list[i], min(encoder_channel_list[i], 256))
            transize[i] = self._make_TranSize(encoder_channel_list[i], min(encoder_channel_list[i], encoder_channel_list[-2]))
        self.transize = nn.Sequential(*transize)
        convs_fuse = []
        # mlp_fuse = [64+128+256*2, 64]
        mlp_fuse = [sum(encoder_channel_list[1:-1])+encoder_channel_list[-2], 2*self.out_channels]
        for i in range(len(mlp_fuse) - 1):
            convs_fuse.append(create_convblock1d(mlp_fuse[i], mlp_fuse[i + 1],
                                            norm_args=norm_args, act_args=act_args
                                            ))
        self.convs_fuse = nn.Sequential(*convs_fuse)
        mlp = [2*self.out_channels+self.out_channels, self.out_channels, self.out_channels]
        self.fp = FeaturePropogation(mlp)

    def _make_TranSize(self, curC, tranC):
        layers = []
        mlp = [curC] + [tranC] #* self.decoder_layers
        layers.append(TransformSize(mlp))
        return nn.Sequential(*layers)

    def forward(self, p, f):
        # allconcat and fuse
        for i in range(-1, -len(self.transize) - 1, -1):
            f[i] = self.transize[i][0]([p[2],f[2]], [p[i],f[i]])
        f_cat = torch.cat(f[2:], dim=1)
        f_fuse = self.convs_fuse(f_cat)
        _, feats = self.longrange(p[2], f_fuse)
        f_fuse = feats[-1]
        # fp
        f_out = self.fp([p[1], f[1]], [p[2], f_fuse])
        return f_out



@MODELS.register_module()
class All3ConcatDecoder256_64_LongRange(nn.Module):
    def __init__(self,
                 encoder_channel_list: List[int],
                 decoder_layers: int = 2,
                 decoder_stages: int = 4, 
                 norm_args={'norm': 'bn1d'},
                 act_args={'act': 'relu'},
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
        concat_num = 3
        transize = [[] for _ in range(concat_num-1)]
        self.out_channels = fp_channels[-n_decoder_stages]
        for i in range(-1, -concat_num, -1):
            # transize[i] = self._make_TranSize(encoder_channel_list[i], min(encoder_channel_list[i], 256))
            transize[i] = self._make_TranSize(encoder_channel_list[i], min(encoder_channel_list[i], encoder_channel_list[-2]))
        self.transize = nn.Sequential(*transize)
        convs_fuse = []
        # mlp_fuse = [128+256*2, 64]
        mlp_fuse = [sum(encoder_channel_list[2:-1])+encoder_channel_list[-2], 2*self.out_channels]
        for i in range(len(mlp_fuse) - 1):
            convs_fuse.append(create_convblock1d(mlp_fuse[i], mlp_fuse[i + 1],
                                            norm_args=norm_args, act_args=act_args
                                            ))
        self.convs_fuse = nn.Sequential(*convs_fuse)
        mlp1 = [2*self.out_channels + encoder_channel_list[1], 2*self.out_channels]
        self.fp1 = FeaturePropogation(mlp1)
        mlp2 = [2*self.out_channels + encoder_channel_list[0], 2*self.out_channels, self.out_channels]
        self.fp2 = FeaturePropogation(mlp2)

    def _make_TranSize(self, curC, tranC):
        layers = []
        mlp = [curC] + [tranC] #* self.decoder_layers
        layers.append(TransformSize(mlp))
        return nn.Sequential(*layers)

    def forward(self, p, f):
        # allconcat and fuse
        for i in range(-1, -len(self.transize) - 1, -1):
            f[i] = self.transize[i][0]([p[3],f[3]], [p[i],f[i]])
        f_cat = torch.cat(f[3:], dim=1)
        f_fuse = self.convs_fuse(f_cat)
        _, feats = self.longrange(p[3], f_fuse)
        f_fuse = feats[-1]
        # fp
        f_out = self.fp1([p[2], f[2]], [p[3], f_fuse])
        f_out = self.fp2([p[1], f[1]], [p[2], f_out])
        return f_out