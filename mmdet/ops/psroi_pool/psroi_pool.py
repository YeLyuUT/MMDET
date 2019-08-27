import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from . import psroi_pool_cuda

class PSRoIPoolingFunction(Function):

    @staticmethod
    def forward(ctx, features, rois, out_size, spatial_scale):
        assert features.is_cuda
        out_h, out_w = _pair(out_size)
        assert isinstance(out_h, int) and isinstance(out_w, int) and out_h==out_w
        num_channels = features.size(1)
        num_rois = rois.size(0)
        output_dim = int(num_channels // (out_h * out_w))
        out_size = (num_rois, output_dim, out_h, out_w)
        output = features.new_zeros(out_size)
        mappingchannel = features.new_zeros(out_size, dtype=torch.int32)
        group_size = out_h
        psroi_pool_cuda.forward(out_h, out_w, spatial_scale, group_size, output_dim, features, rois,
                                                 output, mappingchannel)
        feature_size = features.size()
        ctx.save_for_backward(rois, mappingchannel)
        ctx.feature_size = feature_size
        ctx.spatial_scale = spatial_scale
        ctx.out_size = out_size
        ctx.out_dim = output_dim
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois = ctx.saved_tensors[0]
        mappingchannel = ctx.saved_tensors[1]
        num_rois, output_dim, out_h, out_w = ctx.out_size
        feature_size = ctx.feature_size
        assert (feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = feature_size
        grad_input = grad_output.new_zeros(batch_size, num_channels, data_height, data_width)
        psroi_pool_cuda.backward(out_h, out_w, ctx.spatial_scale, output_dim,
                                               grad_output, rois, grad_input, mappingchannel)
        return grad_input, None, None, None

psroi_pool = PSRoIPoolingFunction.apply

class PSRoIPool(nn.Module):
    def __init__(self, out_size, spatial_scale):
        super(PSRoIPool, self).__init__()
        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        return psroi_pool(features, rois, self.out_size, self.spatial_scale)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(out_size={}, spatial_scale={}'.format(
            self.out_size, self.spatial_scale)
        return format_str

class PSRoIPoolAfterPointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels_cls, out_channels_box, out_size, spatial_scale):
        super(PSRoIPoolAfterPointwiseConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels_cls = out_channels_cls
        self.out_channels_box = out_channels_box
        self.ps_pool = PSRoIPool(out_size, spatial_scale)
        self.pointWiseConv_cls = nn.Conv2d(self.in_channels, self.out_channels_cls, kernel_size=1, stride=1)
        self.pointWiseConv_box = nn.Conv2d(self.in_channels, self.out_channels_box, kernel_size=1, stride=1)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.pointWiseConv_cls.weight, 0, 0.01)
        nn.init.constant_(self.pointWiseConv_cls.bias, 0)
        nn.init.normal_(self.pointWiseConv_box.weight, 0, 0.001)
        nn.init.constant_(self.pointWiseConv_box.bias, 0)

    def forward(self, features, rois):
        features_cls = self.pointWiseConv_cls(features)
        features_box = self.pointWiseConv_box(features)
        return self.ps_pool(features_cls, rois), self.ps_pool(features_box, rois)

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += '(in_channels={}, out_channels_cls={}, out_channels_box={}, out_size={}, spatial_scale={}'.format(
            self.in_channels, self.out_channels_cls, self.out_channels_box, self.out_size, self.spatial_scale)
        return format_str