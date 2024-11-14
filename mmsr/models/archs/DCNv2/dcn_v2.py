#!/usr/bin/env python

import logging
import math

# import _ext as _backend
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
# from torch.autograd import Function
# from torch.autograd.function import once_differentiable
# from torch.nn.modules.utils import _pair

logger = logging.getLogger('base')




# class DCNv2Layer(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, deformable_groups=1):
#         super(DCNv2Layer, self).__init__()
#         self.offset_mask_conv = nn.Conv2d(in_channels, deformable_groups * 3 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
#         self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, deformable_groups=deformable_groups)
        
#     def forward(self, x):
#         out = self.offset_mask_conv(x)
#         offset, mask = torch.split(out, out.size(1) // 3 * 2, dim=1), torch.sigmoid(out[:, out.size(1) // 3 * 2:])
#         return self.deform_conv(x, offset, mask)


# class _DCNv2(Function):

#     @staticmethod
#     def forward(ctx, input, offset, mask, weight, bias, stride, padding,
#                 dilation, deformable_groups):
#         ctx.stride = _pair(stride)
#         ctx.padding = _pair(padding)
#         ctx.dilation = _pair(dilation)
#         ctx.kernel_size = _pair(weight.shape[2:4])
#         ctx.deformable_groups = deformable_groups
#         output = _backend.dcn_v2_forward(
#             input, weight, bias, offset, mask, ctx.kernel_size[0],
#             ctx.kernel_size[1], ctx.stride[0], ctx.stride[1], ctx.padding[0],
#             ctx.padding[1], ctx.dilation[0], ctx.dilation[1],
#             ctx.deformable_groups)
#         ctx.save_for_backward(input, offset, mask, weight, bias)
#         return output

#     @staticmethod
#     @once_differentiable
#     def backward(ctx, grad_output):
#         input, offset, mask, weight, bias = ctx.saved_tensors
#         grad_input, grad_offset, grad_mask, grad_weight, grad_bias = \
#             _backend.dcn_v2_backward(input, weight,
#                                      bias,
#                                      offset, mask,
#                                      grad_output,
#                                      ctx.kernel_size[0], ctx.kernel_size[1],
#                                      ctx.stride[0], ctx.stride[1],
#                                      ctx.padding[0], ctx.padding[1],
#                                      ctx.dilation[0], ctx.dilation[1],
#                                      ctx.deformable_groups)

#         return grad_input, grad_offset, grad_mask, grad_weight, grad_bias,\
#             None, None, None, None,


# dcn_v2_conv = _DCNv2.apply


# class DCNv2(nn.Module):

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride,
#                  padding,
#                  dilation=1,
#                  deformable_groups=1):
#         super(DCNv2, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = _pair(kernel_size)
#         self.stride = _pair(stride)
#         self.padding = _pair(padding)
#         self.dilation = _pair(dilation)
#         self.deformable_groups = deformable_groups

#         self.weight = nn.Parameter(
#             torch.Tensor(out_channels, in_channels, *self.kernel_size))
#         self.bias = nn.Parameter(torch.Tensor(out_channels))
#         self.reset_parameters()

#     def reset_parameters(self):
#         n = self.in_channels
#         for k in self.kernel_size:
#             n *= k
#         stdv = 1. / math.sqrt(n)
#         self.weight.data.uniform_(-stdv, stdv)
#         self.bias.data.zero_()

#     def forward(self, input, offset, mask):
#         assert 2 * self.deformable_groups * self.kernel_size[
#             0] * self.kernel_size[1] == offset.shape[1]
#         assert self.deformable_groups * self.kernel_size[0] * self.kernel_size[
#             1] == mask.shape[1]
#         return dcn_v2_conv(input, offset, mask, self.weight, self.bias,
#                            self.stride, self.padding, self.dilation,
#                            self.deformable_groups)


# class DCN(DCNv2):

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride,
#                  padding,
#                  dilation=1,
#                  deformable_groups=1):
#         super(DCN, self).__init__(in_channels, out_channels, kernel_size,
#                                   stride, padding, dilation, deformable_groups)

#         channels_ = self.deformable_groups * 3 * self.kernel_size[
#             0] * self.kernel_size[1]
#         self.conv_offset_mask = nn.Conv2d(
#             self.in_channels,
#             channels_,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             padding=self.padding,
#             bias=True)
#         self.init_offset()

#     def init_offset(self):
#         self.conv_offset_mask.weight.data.zero_()
#         self.conv_offset_mask.bias.data.zero_()

#     def forward(self, input):
#         out = self.conv_offset_mask(input)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)
#         return dcn_v2_conv(input, offset, mask, self.weight, self.bias,
#                            self.stride, self.padding, self.dilation,
#                            self.deformable_groups)


# class DCN_sep(DCNv2):
#     '''Use other features to generate offsets and masks'''

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride,
#                  padding,
#                  dilation=1,
#                  deformable_groups=1,
#                  extra_offset_mask=True):
#         super(DCN_sep,
#               self).__init__(in_channels, out_channels, kernel_size, stride,
#                              padding, dilation, deformable_groups)
#         self.extra_offset_mask = extra_offset_mask
#         channels_ = self.deformable_groups * 3 * self.kernel_size[
#             0] * self.kernel_size[1]
#         self.conv_offset_mask = nn.Conv2d(
#             self.in_channels,
#             channels_,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             padding=self.padding,
#             bias=True)
#         self.init_offset()

#     def init_offset(self):
#         self.conv_offset_mask.weight.data.zero_()
#         self.conv_offset_mask.bias.data.zero_()

#     def forward(self, x):
#         if self.extra_offset_mask:
#             # x = [input, features]
#             out = self.conv_offset_mask(x[1])
#             x = x[0]
#         else:
#             out = self.conv_offset_mask(x)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         mask = torch.sigmoid(mask)

#         offset_mean = torch.mean(torch.abs(offset))
#         if offset_mean > 100:
#             logger.warning(
#                 'Offset mean is {}, larger than 100.'.format(offset_mean))
#         return dcn_v2_conv(x, offset, mask, self.weight, self.bias,
#                            self.stride, self.padding, self.dilation,
#                            self.deformable_groups)


# class DCN_sep_pre_multi_offset(DCNv2):
#     '''
#     Use other features to generate offsets and masks.

#     Intialized the offset with precomputed non-local offset.
#     '''

#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  kernel_size,
#                  stride,
#                  padding,
#                  dilation=1,
#                  deformable_groups=1,
#                  extra_offset_mask=True):
#         super(DCN_sep_pre_multi_offset,
#               self).__init__(in_channels, out_channels, kernel_size, stride,
#                              padding, dilation, deformable_groups)
#         self.extra_offset_mask = extra_offset_mask
#         channels_ = self.deformable_groups * 3 * self.kernel_size[
#             0] * self.kernel_size[1]
#         self.conv_offset_mask = nn.Conv2d(
#             self.in_channels,
#             channels_,
#             kernel_size=self.kernel_size,
#             stride=self.stride,
#             padding=self.padding,
#             bias=True)
#         self.init_offset()

#     def init_offset(self):
#         self.conv_offset_mask.weight.data.zero_()
#         self.conv_offset_mask.bias.data.zero_()

#     def forward(self, x, pre_offset):
#         '''
#         Args:
#             pre_offset: precomputed_offset. Size: [b, 9, h, w, 2]
#         '''
#         if self.extra_offset_mask:
#             # x = [input, features]
#             out = self.conv_offset_mask(x[1])
#             x = x[0]
#         else:
#             out = self.conv_offset_mask(x)
#         o1, o2, mask = torch.chunk(out, 3, dim=1)
#         offset = torch.cat((o1, o2), dim=1)
#         # repeat pre_offset along dim1, shape: [b, 9*groups, h, w, 2]
#         pre_offset = pre_offset.repeat([1, self.deformable_groups, 1, 1, 1])
#         # the order of offset is [y, x, y, x, ..., y, x]
#         pre_offset_reorder = torch.zeros_like(offset)
#         # add pre_offset on y-axis
#         pre_offset_reorder[:, 0::2, :, :] = pre_offset[:, :, :, :, 1]
#         # add pre_offset on x-axis
#         pre_offset_reorder[:, 1::2, :, :] = pre_offset[:, :, :, :, 0]
#         offset = offset + pre_offset_reorder
#         # print(offset.size())
#         mask = torch.sigmoid(mask)

#         offset_mean = torch.mean(torch.abs(offset - pre_offset_reorder))
#         if offset_mean > 100:
#             logger.warning(
#                 'Offset mean is {}, larger than 100.'.format(offset_mean))
#         return dcn_v2_conv(x, offset, mask, self.weight, self.bias,
#                            self.stride, self.padding, self.dilation,
#                            self.deformable_groups)



class DCN_sep_pre_multi_offset(nn.Module):
    '''
    Use other features to generate offsets and masks.
    Initialized the offset with precomputed non-local offset.
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 deformable_groups=1,
                 extra_offset_mask=True):
        super(DCN_sep_pre_multi_offset, self).__init__()
        self.extra_offset_mask = extra_offset_mask
        self.deformable_groups = deformable_groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.out_channels = out_channels

        channels_ = self.deformable_groups * 3 * self.kernel_size * self.kernel_size
        self.conv_offset_mask = nn.Conv2d(
            in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, x, pre_offset):
        '''
        Args:
            pre_offset: precomputed_offset. Size: [b, 9, h, w, 2]
        '''
        if self.extra_offset_mask:
            # x = [input, features]
            out = self.conv_offset_mask(x[1])
            x = x[0]
        else:
            out = self.conv_offset_mask(x)
        
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        
        # repeat pre_offset along dim1, shape: [b, 9*groups, h, w, 2]
        pre_offset = pre_offset.repeat([1, self.deformable_groups, 1, 1, 1])
        
        # the order of offset is [y, x, y, x, ..., y, x]
        pre_offset_reorder = torch.zeros_like(offset)

        # add pre_offset on y-axis
        pre_offset_reorder[:, 0::2, :, :] = pre_offset[:, :, :, :, 1]
        
        # add pre_offset on x-axis
        pre_offset_reorder[:, 1::2, :, :] = pre_offset[:, :, :, :, 0]
        
        offset = offset + pre_offset_reorder
        mask = torch.sigmoid(mask)
        
        offset_mean = torch.mean(torch.abs(offset - pre_offset_reorder))
        if offset_mean > 100:
            print(f'Offset mean is {offset_mean}, larger than 100.')
        
        return self.deform_conv(x, offset, mask)

