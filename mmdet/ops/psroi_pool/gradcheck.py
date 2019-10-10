import os.path as osp
import sys

import torch
from torch.autograd import gradcheck

sys.path.append(osp.abspath(osp.join(__file__, '../../')))
from .psroi_pool import PSRoIPool  # noqa: E402, isort:skip

feat = torch.randn(4, 32, 15, 15, requires_grad=True).type(torch.float64).cuda()
rois = torch.Tensor([[0, 0, 0, 50, 50], [0, 10, 30, 43, 55],
                     [1, 67, 40, 110, 120]]).type(torch.float64).cuda()
inputs = (feat, rois)
print('Gradcheck for psroi pooling...')
test = gradcheck(PSRoIPool(4, 1.0 / 8, 2), inputs, eps=1e-5, atol=1e-3)
print(test)
