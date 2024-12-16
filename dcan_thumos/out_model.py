import torch
from torch import nn
import numpy as np


from torch.nn import functional as F



class DConv2D(nn.Module):
    def __init__(self, channel_in, width, channel_out,kernel_size = 7 ,stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        super(DConv2D, self).__init__()
        # self.dconvs = MaskedConv2D(channel_in,channel_out,kernel_size=kernel_size,padding=kernel_size//2)

        self.dcons1 = nn.Conv2d(channel_in*2, width, kernel_size=1,padding=0)
        # self.dcons1 = LAConv2D(channel_in * 2, width, kernel_size=1, padding=0)
        self.dcons2 = nn.Conv2d(width, channel_out, kernel_size=kernel_size, groups=groups, padding=kernel_size//2, dilation=dilation)
        # self.dcons2 = LAConv2D(width, channel_out, kernel_size=kernel_size, groups=groups, padding=kernel_size // 2,
        #                         dilation=dilation)
        # self.dcons3 = nn.Conv2d(width, channel_out, kernel_size=1, padding=0)
        # self.dconvs = nn.Sequential(
        #     nn.Conv2d(channel_in*2, width, kernel_size=1,padding=0), nn.GELU(),
        #     nn.Conv2d(width, channel_out, kernel_size=kernel_size, groups=groups, padding=kernel_size//2, dilation=dilation),
        #     # nn.Conv2d(width, channel_out, kernel_size=1),
        # )  # semantic graph
        # Generating local adaptive weights
        # self.ffn = FFN2D(channel_out,hidden_features=width,out_features=channel_out)

        self.gelu1 = nn.GELU()
        # self.reset_params()
        # self.idx_list = idx
    def reset_params(self):
        # torch.nn.init.xavier_normal_(self.dcons1.weight, a=0, mode='fan_out')

        # torch.nn.init.xavier_normal_(self.dcons2.weight, a=0, mode='fan_out')

        # torch.nn.init.xavier_normal_(self.dcons1.weight)
        # torch.nn.init.kaiming_uniform_()
        # torch.nn.init.xavier_normal_(self.dcons2.weight)
        # torch.nn.init.kaiming_uniform_()
        torch.nn.init.kaiming_normal_(self.dcons1.weight, a=0, mode='fan_out')
        torch.nn.init.kaiming_normal_(self.dcons2.weight, a=0, mode='fan_out')
    def forward(self,x):
        # print(x.shape)
        x = self.gelu1(self.dcons1(x))
        # x = self.dcons1(x)
        # print(x.shape)
        # x = self.gelu2(self.dcons2(x))
        x = self.dcons2(x)
        return x




class TConv1D(nn.Module):
    def __init__(self, channel_in, width, channel_out, kernel_size = 5,stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        super(TConv1D, self).__init__()
        self.tconvs1 = nn.Conv1d(channel_in, width, kernel_size=1,padding=0)
        self.tconvs2 = nn.Conv1d(width, channel_out, kernel_size=kernel_size, groups=groups, padding=kernel_size // 2,
                                 dilation=dilation)
        # self.tconvs3 = MaskedConv1D(width, channel_out, kernel_size=1, padding=0)
        # self.tconvs1 = nn.Conv1d(channel_in, channel_out, kernel_size=1,padding=0)
        # self.tconvs2 = nn.Conv1d(width, channel_out, kernel_size=kernel_size, groups=groups, padding=kernel_size//2,dilation=dilation)
        # self.tconvs = nn.Sequential(
        #
        #     nn.Conv1d(channel_in, width, kernel_size=1,padding=0),nn.GELU(),
        #     nn.Conv1d(width, channel_out, kernel_size=kernel_size, groups=groups, padding=kernel_size//2,dilation=dilation),
        #     # nn.Conv1d(width, channel_out, kernel_size=1),
        # )  # semantic graph
        # Generating local adaptive weights
        # self.reset_params(init_conv_vars=init_conv_vars)
        # self.gelu = nn.GELU()
        # self.reset_params()

        # self.idx_list = idx
        self.gelu = nn.GELU()
    def reset_params(self):
        torch.nn.init.kaiming_normal_(self.tconvs1.weight, a=0, mode='fan_out')

        torch.nn.init.kaiming_normal_(self.tconvs2.weight, a=0, mode='fan_out')
        # torch.nn.init.kaiming_normal_(self.tconvs3.weight, a=0, mode='fan_out')
        # torch.nn.init.xavier_normal_(self.tcons1.weight)
        # torch.nn.init.xavier_normal_(self.tcons2.weight)

    def forward(self, x):
        x  = self.tconvs1(x)
        x = self.gelu(x)
        # x = self.gelu(self.tconvs1(x))
        x = self.tconvs2(x)
        # x, mask = self.tconvs3(x, mask)
        return x


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0

        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype), size=out_conv.size(-1), mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output

class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
        self,
        num_channels,
        eps=1e-5,
        affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


class TemporalMaxer(nn.Module):
    def __init__(
            self,
            kernel_size,
            stride,
            padding,
            n_embd):
        super().__init__()

        self.ds_pooling = nn.AvgPool1d(
            kernel_size, stride=stride, padding=padding)

        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
    def forward(self, x):
        # print(self.inputsize)
        # out, out_mask = self.channel_att(x, mask)

        # if self.stride > 1:
        #     # downsample the mask using nearest neighbor
        #     out_mask = F.interpolate(
        #         mask.to(x.dtype), size=x.size(-1)//self.stride, mode='nearest')
        # else:
        #     # masking out the features
        #     out_mask = mask

        out = self.ds_pooling(x)

        return out


def poincare_distance(u, v):
    """
    Calculates the hyperbolic distance between two vectors in the Poincare ball model.

    Args:
    - u: A torch.Tensor representing the first vector. Shape: (batch_size, embedding_dim)
    - v: A torch.Tensor representing the second vector. Shape: (batch_size, embedding_dim)

    Returns:
    - A torch.Tensor representing the distance between the two vectors. Shape: (batch_size,)
    """
    epsilon = 1e-10
    # print(u.shape,'shape')
    # print(v.shape)
    # Euclidean norm of the input vectors
    norm_u = torch.norm(u, dim=1, keepdim=True)
    norm_v = torch.norm(v, dim=1, keepdim=True)
    # print(norm_v.shape,'norm_v.shape')
    # print(norm_u.shape,'norm_u')
    # Calculate the Poincare ball radius
    radius = 1 - epsilon * epsilon

    # Calculate the magnitude of the embedded vectors in the hyperbolic space
    magnitude_u = torch.sqrt(torch.sum(u ** 2, dim=1, keepdim=True) + epsilon ** 2)
    magnitude_v = torch.sqrt(torch.sum(v ** 2, dim=1, keepdim=True) + epsilon ** 2)
    # print(magnitude_u.shape,'magnitude_u')
    # print(magnitude_v.shape,'magnitude_v')
    # Calculate the dot product of the embedded vectors in the hyperbolic space
    dot_product = torch.sum(u * v, dim=1, keepdim=True)

    # Compute the hyperbolic distance between the vectors
    cosh_distance = 1 + 2 * (torch.norm(u - v, dim=1, keepdim=True) ** 2) / (
                (1 - magnitude_u ** 2) * (1 - magnitude_v.transpose(2, 1) ** 2))
    distance = 1 / epsilon * torch.acosh(cosh_distance)
    # print(distance.shape,'distance.shape')
    return distance



def knn(x, y=None, k=10,type='dis'):
    """
    :param x: BxCxN
    :param y: BxCxM
    :param k: scalar
    :return: BxMxk
    """
    if y is None:
        y = x

    #distant = poincare_distance(x,y)
   # print(distant)
    #distant = pairwise_distances(x,y)
    xx_p = torch.sum(x, dim=1, keepdim=True)
    yy_p = torch.sum(y, dim=1, keepdim=True)

    #print(distant.shape)
    # logging.info('Size in KNN: {} - {}'.format(x.size(), y.size()))
    inner = -2 * torch.matmul(y.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)

    _, idx_dis = pairwise_distance.topk(k=k, dim=-1)
    # print(idx_dis,'idx_dis')
    # print(idx_dis.shape, 'idx_dis')
    # k1 = torch.Tensor(1).cuda()

    distant = poincare_distance(x, y)
    # print(torch.min(distant), 'min')
    _, idx_poi = distant.topk(k=k, dim=-1)
    # print(idx_poi, 'idx_poi')
    # print(idx_poi.shape, 'idx_poi')

    x_cos = F.normalize(x, dim=1)
    y_cos = F.normalize(y, dim=1)

    cos_distant = x_cos.transpose(2, 1) @ y_cos
    _, idx_cos = cos_distant.topk(k=k, dim=-1)
    # print(idx_cos)
    if type == 'dis':
        return idx_dis
    elif type == 'cos':
        return idx_cos
    else:
        return idx_poi


class PoincareEmbedding(torch.nn.Module):
    def __init__(self, num_features, embedding_dim):
        super(PoincareEmbedding, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(num_features, embedding_dim)
        self.epsilon = 1e-5

    def forward(self, x):
        # Euclidean norm of the input feature vector
        norm_x = torch.norm(x, dim=1, keepdim=True)

        # Calculate the Poincare ball radius
        radius = 1 - self.epsilon * self.epsilon

        # Calculate the magnitude of the embedded vector in the hyperbolic space
        magnitude = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + self.epsilon ** 2)

        # Project the vector into the Poincare ball
        u = x / magnitude
        u = u * torch.sqrt(1 - norm_x ** 2 / radius ** 2)
        # print(u.shape,'u')
        # Compute the hyperbolic distance between the origin and the embedded vector
        # distance = 1 / self.epsilon * torch.acosh(1 + 2 * (torch.norm(u, dim=1, keepdim=True) ** 2) / (
        #             (1 - torch.norm(u, dim=1, keepdim=True) ** 2) * (1 - radius)))
        #
        # # Apply the embedding matrix
        # out = self.embedding(torch.arange(self.num_features, device=x.device))
        # out = out.view(1, self.num_features, self.embedding_dim)
        # out = out.expand(x.shape[0], self.num_features, self.embedding_dim)
        # out = torch.sum(out * u.unsqueeze(1), dim=2)

        return u




class GCNeXt(nn.Module):
    def __init__(self,channel_in,
                 n_embd,
                 channel_out,
                 n_embd_ks=3,
                 n_mha_win_size=19,
                 arch = (2, 5),
                 k=[9,7,7,5,5,3],
                 norm_layer=None,
                 groups=32,
                 width_group=4,
                 with_ln = False,
                 idx=None,
                 init_conv_vars=1,
                 scale_factor=2):
        super(GCNeXt, self).__init__()
        self.k = k
        self.groups = groups
        self.channel_in = channel_in
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        # width = width_group * groups
        self.tconvs = nn.ModuleList()
        self.dconvs = nn.ModuleList()
        # self.pconvs = nn.ModuleList()
        # self.cconvs = nn.ModuleList()
        self.samping = nn.ModuleList()
        self.poi = nn.ModuleList()

        # self.ffn = nn.ModuleList()
        # self.ln_before = nn.ModuleList()
        # self.poiembed = PoincareEmbedding(n_embd,2304)
        # self.mask_conv = nn.ModuleList()
        # self.ln_after = nn.ModuleList()
        # self.mlpModul = nn.ModuleList()
        # self.alpha = list()
        # self.conv1 = MaskedConv1D(in_channels=channel_in,out_channels=n_embd,kernel_size=1)
        self.n_in = channel_in
        if isinstance(channel_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(
                channel_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(channel_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None
        for i in range(arch[1]):
            self.tconvs.append(TConv1D(n_embd, n_embd,channel_out,groups=groups))
            self.dconvs.append(DConv2D(n_embd, n_embd,channel_out,groups=groups))
            # self.pconvs.append(PConv2D(n_embd, n_embd, channel_out, groups=groups))
            # self.cconvs.append(CConv2D(n_embd, n_embd, channel_out, groups=groups))
            # self.ffn.append(FFN1D(channel_out,hidden_features=channel_out*4,out_features=channel_out))
            # self.ffn.append(SGPBlock(n_embd,1,1,n_hidden=768, k=5, init_conv_vars=1))
            # self.ln_before.append(LayerNorm(channel_out))
            self.poi.append(PoincareEmbedding(n_embd,n_embd))
            # self.mlpModul.append(MLP_Our(n_embd))
            # self.alpha.append(nn.Parameter(torch.zeros([1]),requires_grad=True))
            # self.mask_conv.append(nn.Conv1d(n_embd,n_embd,kernel_size=1))
            # self.ln_after.append(LayerNorm(channel_out))
        # self.tranconv = nn.ConvTranspose1d(in_channels=n_embd,out_channels=n_embd,kernel_size=3,stride=1,padding=0)
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()

        # self.relu_list = nn.ModuleList()
        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size] * (1 + arch[-1])
        else:
            assert len(n_mha_win_size) == (1 + arch[-1])
            self.mha_win_size = n_mha_win_size

        for idx in range(arch[0]):
            if idx == 0:
                in_channels = channel_in
            else:
                in_channels = n_embd
            # self.embd.append(
            # MaskedConv1D(in_channels, n_embd, kernel_size=n_embd_ks, padding=n_embd_ks // 2, bias=(not with_ln)))
            self.embd.append(nn.Conv1d(
                in_channels, n_embd, n_embd_ks,
                stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
            ))
            # self.embd.append(SGPBlock(in_channels, 1, 1, n_hidden=768, k=5, init_conv_vars=init_conv_vars))



            if with_ln:
                self.embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm.append(nn.Identity())

        for i in range(arch[1]):
            self.samping.append(TemporalMaxer(kernel_size=3,
                                             stride=scale_factor,
                                             padding=1,
                                             n_embd=n_embd))
            # self.relu_list.append(nn.ReLU(True))
        # self.relu = nn.ReLU(True)
    def get_mask(self,x):
        print(x.shape,'xx')
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in x])
        print(feats_lens.shape,'feats_lens.shape')
        max_len = feats_lens.max(0).values.item()
        print(max_len,'len(max_len)')
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]
        return batched_masks
    def forward(self, x):
        # print(x.shape,'x1')
        #x =self.tconvs(x)
        # if isinstance(self.n_in, (list, tuple)):
        #     x = torch.cat(
        #         [proj(s, mask)[0]
        #             for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
        #          ], dim=1
        #     )
        # x ,mask = self.conv1(x,mask)
        # mask = self.get_mask(x)
        # print(mask.shape)
        device = x.device
        for idx in range(len(self.embd)):
            x = self.embd[idx](x)

            # print(x.shape,'x.shape')
            # x = self.relu(x)
        # x,mask = self.conv1(x,mask)
        out_feat = (x,)
        # out_mask = (mask,)
        samp_x = x
        input_size = x.shape[-1]
        # masks = F.interpolate(
        #         mask.to(x.dtype), size=x.size(-1)//self.stride, mode='nearest')
        for i in range(len(self.tconvs)):

            # x = self.ln_before[i](x)
            x = self.samping[i](samp_x)
            samp_x = x
            identity = x  # residual
            tout = self.tconvs[i](x)

            x_f, idx_dis = get_distant_graph_feature(x, k=self.k[i], style=0)  # (bs,ch,100) -> (bs, 2ch, 100, k)
            dout = self.dconvs[i](x_f)
           # print(tout.shape,'tout.shape')
           # print(type(tout),'tout.type')
        #    print(x.shape)
           # print(tout.shape,'tout.shape')
           # tout = self.tcn(x)x
           #  x_a = x.unsqueeze(3)
           #  poi_in,_ = self.poiembed(tout)
           #  print(tout.shape==x.shape)
            x_p = self.poi[i](x)
            x_f, idx_poi = get_po_graph_feature(x,x_p, k=self.k[i], style=0)  # (bs,ch,100) -> (bs, 2ch, 100, k)
            pout = self.dconvs[i](x_f)  # conv on semantic graph


            # x_f, idx_dis = get_cos_graph_feature(x, k=self.k[i], style=0)  # (bs,ch,100) -> (bs, 2ch, 100, k)
            # cout = self.dconvs[i](x_f)

            #print(sout.shape)
            dout = dout.max(dim=-1, keepdim=False)[0]  # (bs, ch, 100, k) -> (bs, ch, 100)
            pout = pout.max(dim=-1, keepdim=False)[0]
            # cout = cout.max(dim=-1, keepdim=False)[0]
            #print(sout.shape)
            # tout = self.ffn[i](tout)
            # dout = self.ffn[i](dout)
            # pout = self.ffn[i](pout)
            # tout = self.ffn[i](tout)

            # identity = self.ffn[i](identity)
            # else:
            x = tout + dout + identity +pout
            # x,mask = self.ffn[i](x,mask)

            # x = tout + identity


            # x = self.ffn[i](x)
            # x = self.relu(out)
            # x = self.ln_after[i](x)
            # out = out*masks[i]


            # x = x * masks[i].to(x.dtype)
            # mask_x = self.mask_conv[i](x)
            # x = x * mask_x
            # print(x.shape)
            # out = self.tranconv(x)
            out = F.interpolate(x, size=input_size, mode='nearest')
            # print(out.shape,'out.shape')
            out_feat += (out,)


            # out_mask += (mask,)

            # if i < len(self.tconvs)-1:
            #     x, mask = self.samping[i](x, mask)

            #
            # if out_mask is None:
            #     out_mask =(masks, )
            # else:
            #     out_mask += (masks,)
            # mask = self.samping[i-1](mask)
            # out+=(x,)
            # if not self.idx_list is None:
            #     self.idx_list.append(idx)
        x_out = out_feat[0]
        for i in range(len(out_feat)):
            if i > 0:
                x_out = x_out+  out_feat[i]
        return x_out
        # return out_feat,masks.bool()

def get_po_graph_feature(x,x_p,prev_x=None, k=20, idx_poi_knn=None, r=-1, style=1):
    """
    :param x:
    :param prev_x:
    :param k:
    :param idx:
    :param r: output downsampling factor (-1 for no downsampling)
    :param style: method to get graph feature
    :return:
    """
    batch_size = x.size(0)
    num_points = x.size(2)  # if prev_x is None else prev_x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx_poi_knn is None:
        idx_poi_knn = knn(x=x_p, y=prev_x, k=k,type='poi')  # (batch_size, num_points, k)
    else:
        k = idx_poi_knn.shape[-1]
    # print(idx_knn.shape)
    device = x.device  # torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_poi_knn + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if style == 0:  # use offset as feature
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    elif style == 1:  # use feature as feature
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    else: # style == 2:
        feature = feature.permute(0,3,1,2)
    # downsample if needed
    if r != -1:
        select_idx = torch.from_numpy(np.random.choice(feature.size(2), feature.size(2) // r,
                                                       replace=False)).to(device=device)
        feature = feature[:, :, select_idx, :]
    return feature, idx_poi_knn

def get_distant_graph_feature(x, prev_x=None, k=20, idx_dis_knn=None, r=-1, style=1):
    """
    :param x:
    :param prev_x:
    :param k:
    :param idx:
    :param r: output downsampling factor (-1 for no downsampling)
    :param style: method to get graph feature
    :return:
    """
    batch_size = x.size(0)
    num_points = x.size(2)  # if prev_x is None else prev_x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx_dis_knn is None:
        idx_dis_knn = knn(x=x, y=prev_x, k=k,type='dis')  # (batch_size, num_points, k)
    else:
        k = idx_dis_knn.shape[-1]
    # print(idx_knn.shape)
    device = x.device  # torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_dis_knn + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if style == 0:  # use offset as feature
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    elif style == 1:  # use feature as feature
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    else: # style == 2:
        feature = feature.permute(0,3,1,2)
    # downsample if needed
    if r != -1:
        select_idx = torch.from_numpy(np.random.choice(feature.size(2), feature.size(2) // r,
                                                       replace=False)).to(device=device)
        feature = feature[:, :, select_idx, :]
    return feature, idx_dis_knn

def get_cos_graph_feature(x, prev_x=None, k=20, idx_cos_knn=None, r=-1, style=1):
    """
    :param x:
    :param prev_x:
    :param k:
    :param idx:
    :param r: output downsampling factor (-1 for no downsampling)
    :param style: method to get graph feature
    :return:
    """
    batch_size = x.size(0)
    num_points = x.size(2)  # if prev_x is None else prev_x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx_cos_knn is None:
        idx_cos_knn = knn(x=x, y=prev_x, k=k,type='cos')  # (batch_size, num_points, k)
    else:
        k = idx_cos_knn.shape[-1]
    # print(idx_knn.shape)
    device = x.device  # torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_cos_knn + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if style == 0:  # use offset as feature
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    elif style == 1:  # use feature as feature
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    else: # style == 2:
        feature = feature.permute(0,3,1,2)
    # downsample if needed
    if r != -1:
        select_idx = torch.from_numpy(np.random.choice(feature.size(2), feature.size(2) // r,
                                                       replace=False)).to(device=device)
        feature = feature[:, :, select_idx, :]
    return feature, idx_cos_knn
